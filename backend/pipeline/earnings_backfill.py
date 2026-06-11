import time
from datetime import datetime, timedelta

import pandas as pd
import requests
import yfinance as yf
from sqlalchemy.orm import Session

from backend.db.database import SessionLocal
from backend.db.models import EarningsTranscript, Signal, SignalLabel, Ticker
from backend.pipeline.earnings import (
    AAPL_CIK,
    HEADERS,
    SUBMISSIONS_URL,
    calculate_sentiment_score,
    fetch_filing_document,
    parse_and_clean_transcript,
    store_transcript,
)

BACKFILL_YEARS = 5        # how far back to scan for filings
VOLATILITY_LOOKBACK = 504  # ~2 trading years of history for rolling std calculation
SIGMA_MULTIPLIER = 1.0    # label threshold = 1 standard deviation of own return dist
HORIZONS = (5, 30, 90)    # trading-day horizons to measure and label


# ── filing discovery ────────────────────────────────────────────────────────

def fetch_all_earnings_8ks(cik: str = AAPL_CIK, years_back: int = BACKFILL_YEARS) -> list[dict]:
    """
    Return every 8-K with Item 2.02 (Results of Operations) filed in the last N years.

    Queries the EDGAR submissions API and follows archive-file links for filings
    that predate the most-recent 1 000 entries. Returns dicts with accession_no
    and filing_date, sorted oldest-first.
    """
    cutoff = datetime.now() - timedelta(days=years_back * 365)

    resp = requests.get(SUBMISSIONS_URL.format(cik=cik), headers=HEADERS, timeout=15)
    resp.raise_for_status()
    time.sleep(0.15)
    sub = resp.json()

    def _parse_block(block: dict) -> list[dict]:
        results = []
        for form, acc, date_str, items in zip(
            block.get("form", []),
            block.get("accessionNumber", []),
            block.get("filingDate", []),
            block.get("items", []),
        ):
            if form != "8-K" or "2.02" not in (items or ""):
                continue
            try:
                fd = datetime.strptime(date_str, "%Y-%m-%d")
            except ValueError:
                continue
            if fd >= cutoff:
                results.append({"accession_no": acc, "filing_date": date_str})
        return results

    all_filings = _parse_block(sub.get("filings", {}).get("recent", {}))

    for arc in sub.get("filings", {}).get("files", []):
        from_str = arc.get("filingFrom", "")
        try:
            arc_from = datetime.strptime(from_str, "%Y-%m-%d")
        except ValueError:
            continue
        if arc_from < cutoff:
            continue  # entire archive block predates our window

        arc_url = f"https://data.sec.gov/submissions/{arc['name']}"
        arc_resp = requests.get(arc_url, headers=HEADERS, timeout=15)
        arc_resp.raise_for_status()
        time.sleep(0.15)
        all_filings.extend(_parse_block(arc_resp.json()))

    # Deduplicate and sort chronologically
    seen: set[str] = set()
    unique: list[dict] = []
    for f in all_filings:
        if f["accession_no"] not in seen:
            seen.add(f["accession_no"])
            unique.append(f)
    return sorted(unique, key=lambda x: x["filing_date"])


# ── price data and labelling ────────────────────────────────────────────────

def fetch_historical_prices(ticker: str, retries: int = 5, base_delay: float = 30.0) -> pd.Series:
    """
    Download split/dividend-adjusted daily close prices for the full backfill
    window plus the volatility-lookback buffer. Returns a tz-naive pd.Series
    indexed by normalised date. Retries on rate-limit errors with exponential backoff.
    Tries yf.download first; falls back to Ticker.history on failure.
    """
    extra = (VOLATILITY_LOOKBACK // 252) + 1
    start_dt = datetime.now() - timedelta(days=(BACKFILL_YEARS + extra) * 365)
    start_str = start_dt.strftime("%Y-%m-%d")

    last_exc: Exception | None = None
    for attempt in range(retries):
        if attempt > 0:
            delay = base_delay * (2 ** (attempt - 1))
            print(f"  yfinance rate limited — retrying in {delay:.0f}s (attempt {attempt + 1}/{retries})...")
            time.sleep(delay)
        # Use a browser-like session to avoid Yahoo Finance rate-limit fingerprinting
        session = requests.Session()
        session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/125.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        })

        try:
            # Primary: yf.download with custom session
            df = yf.download(
                ticker,
                start=start_str,
                auto_adjust=True,
                progress=False,
                multi_level_index=False,
                session=session,
            )
            if not df.empty:
                prices: pd.Series = df["Close"].squeeze()
                prices.index = pd.to_datetime(prices.index).tz_localize(None).normalize()
                return prices
        except Exception as exc:
            last_exc = exc

        # Fallback: Ticker.history with the same session
        try:
            hist = yf.Ticker(ticker, session=session).history(start=start_str, auto_adjust=True)
            if not hist.empty:
                prices = hist["Close"].squeeze()
                prices.index = pd.to_datetime(prices.index).tz_localize(None).normalize()
                return prices
            last_exc = ValueError(f"No price data returned for {ticker}")
        except Exception as exc:
            last_exc = exc

    raise ValueError(f"Failed to download prices for {ticker} after {retries} attempts: {last_exc}")


def calculate_dynamic_thresholds(prices: pd.Series, filing_ts: pd.Timestamp) -> dict[int, float]:
    """
    Compute per-horizon 1-sigma return thresholds from AAPL's own rolling return
    distribution in the VOLATILITY_LOOKBACK window ending on the filing date.

    Because volatility changes over time, this gives each filing its own calibrated
    threshold rather than a constant that could misrepresent low- or high-vol regimes.
    """
    hist = prices[prices.index <= filing_ts].tail(VOLATILITY_LOOKBACK + max(HORIZONS))
    thresholds: dict[int, float] = {}
    for h in HORIZONS:
        rolling = hist.pct_change(periods=h).dropna()
        thresholds[h] = float(rolling.std() * SIGMA_MULTIPLIER) if len(rolling) >= 20 else 0.02
    return thresholds


def calculate_price_labels(
    prices: pd.Series,
    filing_ts: pd.Timestamp,
    thresholds: dict[int, float],
) -> dict:
    """
    Measure 5/30/90 trading-day price outcomes and classify each relative to threshold.

    Base price = first available close on or after the filing date (handles weekends /
    holidays transparently). The Nth-day price is the Nth trading-day close after that
    base date. Return = (price_Nd - base) / base; label = bullish | bearish | neutral.
    """
    base_candidates = prices[prices.index >= filing_ts]
    if base_candidates.empty:
        return {}

    base_price = float(base_candidates.iloc[0])
    future = prices[prices.index > base_candidates.index[0]]

    result: dict = {"price_at_filing": base_price}
    for h in HORIZONS:
        thresh = thresholds.get(h, 0.02)
        if len(future) < h:
            result.update({
                f"price_{h}d": None,
                f"return_{h}d": None,
                f"label_{h}d": None,
                f"threshold_{h}d": float(thresh),
            })
            continue

        target = float(future.iloc[h - 1])   # h-1 because iloc is 0-indexed
        ret = (target - base_price) / base_price

        if ret > thresh:
            label = "bullish"
        elif ret < -thresh:
            label = "bearish"
        else:
            label = "neutral"

        result.update({
            f"price_{h}d": target,
            f"return_{h}d": float(ret),
            f"label_{h}d": label,
            f"threshold_{h}d": float(thresh),
        })

    return result


# ── database helpers (same structure as all other pipelines) ────────────────

def get_or_create_ticker(db: Session, symbol: str) -> Ticker:
    """Get existing ticker or create a new one."""
    ticker = db.query(Ticker).filter(Ticker.symbol == symbol).first()
    if not ticker:
        ticker = Ticker(symbol=symbol)
        db.add(ticker)
        db.commit()
        db.refresh(ticker)
    return ticker


def store_signal(
    db: Session, ticker: Ticker, score: float, direction: str, source: str = "earnings"
) -> Signal:
    """Store the earnings sentiment signal in the signals table."""
    signal = Signal(
        ticker_id=ticker.id,
        source=source,
        score=score,
        direction=direction,
        timestamp=datetime.utcnow(),
    )
    db.add(signal)
    db.commit()
    db.refresh(signal)
    return signal


def store_signal_label(
    db: Session,
    signal: Signal,
    transcript: EarningsTranscript,
    ticker: Ticker,
    filing_date: datetime,
    price_labels: dict,
) -> SignalLabel | None:
    """
    Persist outcome labels for a signal to the signal_labels table.
    Idempotent: returns the existing row if signal_id is already present.
    Returns None if price_labels is empty.
    """
    if not price_labels:
        return None

    existing = db.query(SignalLabel).filter(SignalLabel.signal_id == signal.id).first()
    if existing:
        return existing

    row = SignalLabel(
        signal_id=signal.id,
        transcript_id=transcript.id,
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


# ── orchestrator ────────────────────────────────────────────────────────────

def run_earnings_backfill_pipeline() -> None:
    """
    Backfill AAPL earnings signals and price-outcome labels for the last N years.

    Steps:
      1. Discover every AAPL earnings 8-K (Item 2.02) filed in the past BACKFILL_YEARS years
      2. Download AAPL adjusted close prices covering the full period + lookback buffer
      3. For each filing, oldest-first:
         a. Skip if EarningsTranscript + SignalLabel already exist (idempotent)
         b. Fetch and parse the SEC filing document
         c. VADER sentiment → Signal + EarningsTranscript
         d. Compute per-horizon 1-sigma thresholds from AAPL's own rolling return distribution
         e. Calculate 5/30/90 trading-day price outcomes and volatility-relative labels
         f. Store SignalLabel linked to the Signal via signal_id FK
    """
    print(f"Starting AAPL earnings backfill — {BACKFILL_YEARS} years of 8-K filings")
    db = SessionLocal()

    try:
        print("Step 1: Discovering AAPL earnings 8-K filings from SEC EDGAR...")
        filings = fetch_all_earnings_8ks(AAPL_CIK, BACKFILL_YEARS)
        print(f"  {len(filings)} earnings filings found")
        if not filings:
            print("  No filings in range — aborting")
            return

        print("Step 2: Downloading AAPL adjusted close prices...")
        prices = fetch_historical_prices("AAPL")
        print(f"  {len(prices)} trading days loaded "
              f"({prices.index[0].date()} → {prices.index[-1].date()})")

        ticker_obj = get_or_create_ticker(db, "AAPL")
        processed = skipped = failed = 0

        print(f"Step 3: Processing {len(filings)} filings...")
        for i, filing in enumerate(filings, 1):
            acc = filing["accession_no"]
            date_str = filing["filing_date"]
            filing_date = datetime.strptime(date_str, "%Y-%m-%d")
            filing_ts = pd.Timestamp(date_str)

            print(f"  [{i:02d}/{len(filings)}] {date_str}  {acc}")

            # Idempotency: skip if both transcript and label already stored
            existing_tx = (
                db.query(EarningsTranscript)
                .filter(EarningsTranscript.accession_number == acc)
                .first()
            )
            if existing_tx:
                existing_lbl = (
                    db.query(SignalLabel)
                    .filter(SignalLabel.transcript_id == existing_tx.id)
                    .first()
                )
                if existing_lbl:
                    print("    Already processed — skipping")
                    skipped += 1
                    continue

            try:
                # Fetch and parse document
                html, doc_url = fetch_filing_document(AAPL_CIK, acc)
                if not html:
                    print("    Could not retrieve document — skipping")
                    failed += 1
                    continue

                parsed = parse_and_clean_transcript(html)
                score, direction = calculate_sentiment_score(parsed)

                # Persist transcript and signal
                transcript = store_transcript(
                    db,
                    ticker=ticker_obj,
                    filing_date=filing_date,
                    accession_number=acc,
                    document_url=doc_url,
                    cleaned_text=parsed["cleaned_text"],
                    ceo_cfo_text=parsed["ceo_cfo_text"],
                )
                signal = store_signal(db, ticker_obj, score, direction)

                # Dynamic thresholds — calibrated to AAPL's own vol at this point in time
                thresholds = calculate_dynamic_thresholds(prices, filing_ts)
                price_labels = calculate_price_labels(prices, filing_ts, thresholds)

                store_signal_label(
                    db,
                    signal=signal,
                    transcript=transcript,
                    ticker=ticker_obj,
                    filing_date=filing_date,
                    price_labels=price_labels,
                )

                # Human-readable summary row
                def _fmt(h: int) -> str:
                    ret = price_labels.get(f"return_{h}d")
                    lbl = price_labels.get(f"label_{h}d", "?")
                    thr = price_labels.get(f"threshold_{h}d", 0)
                    if ret is None:
                        return f"{h}d: N/A"
                    return f"{h}d: {ret:+.2%} [{lbl}, σ={thr:.2%}]"

                print(f"    VADER={score:+.3f} ({direction}) | "
                      f"{_fmt(5)} | {_fmt(30)} | {_fmt(90)}")
                processed += 1

            except Exception as exc:
                print(f"    ERROR: {exc} — skipping this filing")
                db.rollback()
                failed += 1

        print()
        print(f"Backfill complete — "
              f"processed={processed}  skipped={skipped}  failed={failed}")

    except Exception as exc:
        print(f"Fatal error: {exc}")
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    run_earnings_backfill_pipeline()
