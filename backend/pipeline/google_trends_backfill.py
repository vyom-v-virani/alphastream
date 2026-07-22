import time
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf
from pytrends.request import TrendReq
from sqlalchemy.orm import Session

from backend.db.database import SessionLocal
from backend.db.models import Signal, SignalLabel, Ticker
from backend.pipeline.earnings_backfill import (
    calculate_dynamic_thresholds,
    calculate_price_labels,
    HORIZONS,
)
from backend.pipeline.google_trends import (
    calculate_sentiment_score,
    get_or_create_ticker,
    store_signal,
)

BACKFILL_YEARS = 5              # how far back to scan for weekly trend data
MONTH_CHUNK_DAYS = 30            # pytrends is fetched one ~month window at a time to avoid rate limiting
REQUEST_DELAY_SECONDS = 1.5      # pause between pytrends requests
RETRY_DELAY_SECONDS = 60         # backoff before a single retry on a rate-limited request

VOLATILITY_LOOKBACK_DAYS = 504   # ~2 trading years, matches earnings_backfill's lookback window


# ── google trends fetch (chunked monthly to avoid rate limiting) ────────────

def fetch_google_trends_backfill(keyword: str = "AAPL", years_back: int = BACKFILL_YEARS) -> pd.DataFrame:
    """
    Fetch interest-over-time data for `keyword` across the full backfill window,
    one ~month-long window at a time (pytrends aggressively rate-limits large
    single requests). Returns a combined, date-sorted DataFrame with a single
    interest-value column (isPartial dropped), ready to be split into weekly
    buckets by fetch_weekly_trend_signals.
    """
    pytrends = TrendReq(hl="en-US", tz=360)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years_back * 365)

    chunks: list[pd.DataFrame] = []
    window_start = start_date

    while window_start < end_date:
        window_end = min(window_start + timedelta(days=MONTH_CHUNK_DAYS), end_date)
        timeframe = f"{window_start.strftime('%Y-%m-%d')} {window_end.strftime('%Y-%m-%d')}"
        print(f"  Fetching Google Trends window {timeframe}...")

        data = None
        for attempt in (1, 2):
            try:
                pytrends.build_payload(kw_list=[keyword], timeframe=timeframe)
                data = pytrends.interest_over_time()
                break
            except Exception as exc:
                if attempt == 1:
                    print(f"    Rate limited / error ({exc}) — backing off {RETRY_DELAY_SECONDS}s and retrying once")
                    time.sleep(RETRY_DELAY_SECONDS)
                else:
                    print(f"    Failed twice ({exc}) — skipping this window")

        if data is not None and not data.empty:
            if "isPartial" in data.columns:
                data = data.drop("isPartial", axis=1)
            chunks.append(data)

        time.sleep(REQUEST_DELAY_SECONDS)
        window_start = window_end

    if not chunks:
        return pd.DataFrame()

    combined = pd.concat(chunks).sort_index()
    combined = combined[~combined.index.duplicated(keep="first")]
    return combined


def fetch_weekly_trend_signals(daily_trend_data: pd.DataFrame) -> list[dict]:
    """
    Split the combined trend series into calendar weeks and run
    calculate_sentiment_score (from google_trends.py, unmodified) over each
    week's slice. Returns one record per week, oldest first, anchored on the
    last date present in that week.
    """
    if daily_trend_data.empty:
        return []

    weekly: list[dict] = []
    for _, week_df in daily_trend_data.groupby(pd.Grouper(freq="W")):
        if week_df.empty:
            continue
        score, direction = calculate_sentiment_score(week_df)
        anchor_date = week_df.index[-1].to_pydatetime()
        weekly.append({"anchor_date": anchor_date, "score": score, "direction": direction})

    return weekly


# ── price fetch (same source as earnings_backfill.py / options_flow.py) ─────

def fetch_aapl_prices(ticker: str = "AAPL") -> pd.Series:
    """
    Download split/dividend-adjusted daily close prices covering the full
    backfill window plus the volatility-lookback buffer, via yfinance.

    Alpha Vantage's TIME_SERIES_DAILY?outputsize=full is a premium-only
    endpoint as of testing, so this mirrors earnings_backfill.py's price
    source instead. Returns a tz-naive pd.Series indexed by normalised date,
    sorted oldest-first.
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


# ── database helpers (mirrors earnings_backfill.py) ──────────────────────────

def store_signal_label(
    db: Session,
    signal: Signal,
    ticker: Ticker,
    filing_date: datetime,
    price_labels: dict,
) -> SignalLabel | None:
    """
    Persist outcome labels for a Google Trends signal to signal_labels.
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
    Idempotency check: has this week (identified by ticker + anchor date) already
    been backfilled for the google_trends source? There is no natural document
    key here (unlike accession_number for earnings), so filing_date + source
    stands in as the unique key for a given week.
    """
    existing = (
        db.query(SignalLabel)
        .join(Signal, Signal.id == SignalLabel.signal_id)
        .filter(
            SignalLabel.ticker_id == ticker.id,
            SignalLabel.filing_date == anchor_date,
            Signal.source == "google_trends",
        )
        .first()
    )
    return existing is not None


# ── orchestrator ─────────────────────────────────────────────────────────────

def run_google_trends_backfill_pipeline() -> None:
    """
    Backfill AAPL Google Trends signals and price-outcome labels for the last
    BACKFILL_YEARS years, one calendar week at a time.

    Steps:
      1. Fetch AAPL Google Trends interest-over-time data in monthly chunks
         (pytrends rate-limits large single requests)
      2. Split into calendar weeks; score each week with calculate_sentiment_score
         (same function used by the live google_trends.py pipeline)
      3. Download AAPL daily close prices via yfinance
      4. For each week, oldest-first:
         a. Skip if a signal_labels row already exists for this week (idempotent)
         b. Store the Google Trends Signal
         c. Compute per-horizon 1-sigma thresholds from AAPL's own rolling return
            distribution (same logic as earnings_backfill.py)
         d. Calculate 5/30/90 trading-day price outcomes and volatility-relative labels
         e. Store SignalLabel linked to the Signal via signal_id FK
    """
    print(f"Starting AAPL Google Trends backfill — {BACKFILL_YEARS} years, weekly")
    db = SessionLocal()

    try:
        print("Step 1: Fetching AAPL Google Trends data (monthly chunks)...")
        daily_trend_data = fetch_google_trends_backfill("AAPL", BACKFILL_YEARS)
        if daily_trend_data.empty:
            print("  No Google Trends data returned — aborting")
            return

        print("Step 2: Splitting into weekly sentiment signals...")
        weekly_signals = fetch_weekly_trend_signals(daily_trend_data)
        print(f"  {len(weekly_signals)} weeks of trend data")
        if not weekly_signals:
            print("  No weekly signals derived — aborting")
            return

        print("Step 3: Downloading AAPL daily close prices via yfinance...")
        prices = fetch_aapl_prices("AAPL")
        print(f"  {len(prices)} trading days loaded "
              f"({prices.index[0].date()} to {prices.index[-1].date()})")

        ticker_obj = get_or_create_ticker(db, "AAPL")
        processed = skipped = failed = 0

        print(f"Step 4: Processing {len(weekly_signals)} weeks...")
        for i, week in enumerate(weekly_signals, 1):
            anchor_date = week["anchor_date"]
            anchor_ts = pd.Timestamp(anchor_date)
            print(f"  [{i:03d}/{len(weekly_signals)}] week ending {anchor_date.date()}")

            if label_already_stored(db, ticker_obj, anchor_date):
                print("    Already processed — skipping")
                skipped += 1
                continue

            try:
                score, direction = week["score"], week["direction"]
                signal = store_signal(db, ticker_obj, score, direction)

                thresholds = calculate_dynamic_thresholds(prices, anchor_ts)
                price_labels = calculate_price_labels(prices, anchor_ts, thresholds)

                if not price_labels:
                    print("    No price data available for this week — skipping label")
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

                print(f"    Trends={score:+.3f} ({direction}) | "
                      f"{' | '.join(_fmt(h) for h in HORIZONS)}")
                processed += 1

            except Exception as exc:
                print(f"    ERROR: {exc} — skipping this week")
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
    run_google_trends_backfill_pipeline()
