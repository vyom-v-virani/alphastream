import time
from datetime import datetime, timedelta

import pandas as pd
import requests
import yfinance as yf
from sqlalchemy.orm import Session

from backend.db.database import SessionLocal
from backend.db.models import Signal, SignalLabel, Ticker
from backend.pipeline.earnings_backfill import (
    calculate_dynamic_thresholds,
    calculate_price_labels,
    HORIZONS,
)
from backend.pipeline.weather import (
    CORN_BELT_STATES,
    NOAA_API_KEY,
    NOAA_CDO_URL,
    aggregate_state_weather,
    calculate_corn_impact_score,
    get_or_create_ticker,
    store_signal,
)

BACKFILL_YEARS = 5              # how far back to scan for monthly weather data
VOLATILITY_LOOKBACK_DAYS = 504  # ~2 trading years, matches earnings_backfill's lookback window
REQUEST_DELAY_SECONDS = 0.25    # stay within NOAA's 5 req/sec limit (same as weather.py)


# ── NOAA fetch (parameterised by date range instead of "last N days") ───────

def fetch_weather_data_range(start_date, end_date) -> dict:
    """
    Fetch TMAX, TMIN, PRCP from NOAA CDO (GHCND) for corn belt states over an
    arbitrary [start_date, end_date] window (weather.py's fetch_weather_data
    only supports "last N days from now", which can't reach historical months).

    Same endpoint, auth, and datatypes as weather.py's fetch_weather_data.
    Returns dict keyed by state name with list of raw station records.
    """
    if not NOAA_API_KEY:
        raise ValueError("NOAA_API_KEY not found in environment variables")

    headers = {"token": NOAA_API_KEY}
    results = {}

    for state_name, fips_id in CORN_BELT_STATES.items():
        print(f"    Fetching NOAA data for {state_name} ({fips_id})...")
        params = {
            "datasetid": "GHCND",
            "locationid": fips_id,
            "datatypeid": "TMAX,TMIN,PRCP",
            "startdate": start_date.isoformat(),
            "enddate": end_date.isoformat(),
            "limit": 1000,
            "offset": 1,
        }

        response = requests.get(NOAA_CDO_URL, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()

        results[state_name] = payload.get("results", [])
        time.sleep(REQUEST_DELAY_SECONDS)

    return results


def fetch_monthly_corn_signals(years_back: int = BACKFILL_YEARS) -> list[dict]:
    """
    Walk the backfill window one calendar month at a time, fetching NOAA data
    for each month and scoring it with the same aggregate_state_weather and
    calculate_corn_impact_score functions weather.py's live pipeline uses.

    Returns one record per month, oldest first, anchored on the last calendar
    day of that month (capped at today for the current, still-in-progress month).
    """
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=years_back * 365)

    month_starts = pd.date_range(
        start=pd.Timestamp(start_date).replace(day=1), end=end_date, freq="MS"
    )

    monthly: list[dict] = []
    for month_start_ts in month_starts:
        month_start = month_start_ts.date()
        month_end = min(
            (month_start_ts + pd.offsets.MonthEnd(0)).date(),
            end_date,
        )
        print(f"  Fetching NOAA window {month_start} to {month_end}...")

        raw_data = fetch_weather_data_range(month_start, month_end)
        aggregated = aggregate_state_weather(raw_data)
        total_obs = sum(len(v) for v in aggregated.values())
        if total_obs == 0:
            print("    No observations for this month — skipping")
            continue

        crop_score, crop_label = calculate_corn_impact_score(aggregated, month=month_start.month)
        # Invert: favorable growing conditions (positive crop score) = bearish ZC price
        signal_score = -crop_score
        direction = "bullish" if signal_score > 0.15 else ("bearish" if signal_score < -0.15 else "neutral")

        monthly.append({
            "anchor_date": datetime.combine(month_end, datetime.min.time()),
            "score": signal_score,
            "direction": direction,
            "crop_score": crop_score,
            "crop_label": crop_label,
        })

    return monthly


# ── ZC futures price fetch ────────────────────────────────────────────────────

def fetch_zc_prices(ticker: str = "ZC=F") -> pd.Series:
    """
    Download ZC (corn futures) daily close prices covering the full backfill
    window plus the volatility-lookback buffer, via yfinance.
    Returns a tz-naive pd.Series indexed by normalised date, sorted oldest-first.
    """
    extra = (VOLATILITY_LOOKBACK_DAYS // 252) + 1
    start = datetime.now() - timedelta(days=(BACKFILL_YEARS + extra) * 365)
    df = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
        multi_level_index=False,
    )
    if df.empty:
        raise ValueError(f"No price data returned for {ticker}")

    prices: pd.Series = df["Close"].squeeze()
    prices.index = pd.to_datetime(prices.index).tz_localize(None).normalize()
    return prices


# ── database helpers (mirrors earnings_backfill.py / google_trends_backfill.py) ──

def store_signal_label(
    db: Session,
    signal: Signal,
    ticker: Ticker,
    filing_date: datetime,
    price_labels: dict,
) -> SignalLabel | None:
    """
    Persist outcome labels for a weather (ZC) signal to signal_labels.
    transcript_id is left null — there is no transcript for this source.
    Idempotent: returns the existing row if signal_id is already present.
    """
    if not price_labels:
        return None

    existing = db.query(SignalLabel).filter(SignalLabel.signal_id == signal.id).first()
    if existing:
        return existing

    row = SignalLabel(
        signal_id=signal.id,
        transcript_id=None,
        ticker_id=ticker.id,
        filing_date=filing_date,
        price_at_filing=price_labels.get("price_at_filing"),
        price_5d=price_labels.get("price_5d"),
        return_5d=price_labels.get("return_5d"),
        label_5d=price_labels.get("label_5d"),
        threshold_5d=price_labels.get("threshold_5d"),
        price_30d=price_labels.get("price_30d"),
        return_30d=price_labels.get("return_30d"),
        label_30d=price_labels.get("label_30d"),
        threshold_30d=price_labels.get("threshold_30d"),
        price_90d=price_labels.get("price_90d"),
        return_90d=price_labels.get("return_90d"),
        label_90d=price_labels.get("label_90d"),
        threshold_90d=price_labels.get("threshold_90d"),
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def label_already_stored(db: Session, ticker: Ticker, anchor_date: datetime) -> bool:
    """
    Idempotency check: has this month (identified by ticker + anchor date)
    already been backfilled for the weather source? There is no natural
    document key here (unlike accession_number for earnings), so filing_date
    + source stands in as the unique key for a given month.
    """
    existing = (
        db.query(SignalLabel)
        .join(Signal, Signal.id == SignalLabel.signal_id)
        .filter(
            SignalLabel.ticker_id == ticker.id,
            SignalLabel.filing_date == anchor_date,
            Signal.source == "weather",
        )
        .first()
    )
    return existing is not None


# ── orchestrator ─────────────────────────────────────────────────────────────

def run_weather_backfill_pipeline() -> None:
    """
    Backfill ZC weather-derived signals and price-outcome labels for the last
    BACKFILL_YEARS years, one calendar month at a time.

    Steps:
      1. Fetch NOAA GHCND data for Iowa, Illinois, Indiana, Nebraska one month
         at a time across the backfill window
      2. Aggregate and score each month with aggregate_state_weather /
         calculate_corn_impact_score (same functions the live weather.py
         pipeline uses), inverted to a ZC price-direction signal
      3. Download ZC=F daily close prices via yfinance
      4. For each month, oldest-first:
         a. Skip if a signal_labels row already exists for this month (idempotent)
         b. Store the weather-derived ZC Signal
         c. Compute per-horizon 1-sigma thresholds from ZC's own rolling return
            distribution (same logic as earnings_backfill.py)
         d. Calculate 5/30/90 trading-day price outcomes and volatility-relative labels
         e. Store SignalLabel linked to the Signal via signal_id FK
    """
    print(f"Starting ZC weather backfill — {BACKFILL_YEARS} years, monthly")
    db = SessionLocal()

    try:
        print("Step 1-2: Fetching and scoring NOAA weather data by month...")
        monthly_signals = fetch_monthly_corn_signals(BACKFILL_YEARS)
        print(f"  {len(monthly_signals)} months of corn-impact data")
        if not monthly_signals:
            print("  No monthly signals derived — aborting")
            return

        print("Step 3: Downloading ZC=F daily close prices via yfinance...")
        prices = fetch_zc_prices("ZC=F")
        print(f"  {len(prices)} trading days loaded "
              f"({prices.index[0].date()} to {prices.index[-1].date()})")

        ticker_obj = get_or_create_ticker(db, "ZC")
        processed = skipped = failed = 0

        print(f"Step 4: Processing {len(monthly_signals)} months...")
        for i, month in enumerate(monthly_signals, 1):
            anchor_date = month["anchor_date"]
            anchor_ts = pd.Timestamp(anchor_date)
            print(f"  [{i:03d}/{len(monthly_signals)}] month ending {anchor_date.date()}")

            if label_already_stored(db, ticker_obj, anchor_date):
                print("    Already processed — skipping")
                skipped += 1
                continue

            try:
                score, direction = month["score"], month["direction"]
                signal = store_signal(db, ticker_obj, score, direction)

                thresholds = calculate_dynamic_thresholds(prices, anchor_ts)
                price_labels = calculate_price_labels(prices, anchor_ts, thresholds)

                if not price_labels:
                    print("    No price data available for this month — skipping label")
                    db.rollback()
                    skipped += 1
                    continue

                store_signal_label(
                    db,
                    signal=signal,
                    ticker=ticker_obj,
                    filing_date=anchor_date,
                    price_labels=price_labels,
                )

                def _fmt(h: int) -> str:
                    ret = price_labels.get(f"return_{h}d")
                    lbl = price_labels.get(f"label_{h}d", "?")
                    thr = price_labels.get(f"threshold_{h}d", 0)
                    if ret is None:
                        return f"{h}d: N/A"
                    return f"{h}d: {ret:+.2%} [{lbl}, sigma={thr:.2%}]"

                print(f"    Crop={month['crop_score']:+.3f} ({month['crop_label']}) | "
                      f"ZC signal={score:+.3f} ({direction}) | "
                      f"{' | '.join(_fmt(h) for h in HORIZONS)}")
                processed += 1

            except Exception as exc:
                print(f"    ERROR: {exc} — skipping this month")
                db.rollback()
                failed += 1

        print()
        print(f"Backfill complete — processed={processed}  skipped={skipped}  failed={failed}")

    except Exception as exc:
        print(f"Fatal error: {exc}")
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    run_weather_backfill_pipeline()
