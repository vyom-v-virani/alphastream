import time
from datetime import datetime

import requests
from sqlalchemy.orm import Session

from backend.db.database import SessionLocal
from backend.db.models import Signal, Ticker

# ApeWisdom public API — no key required
APEWISDOM_FILTERS = {
    "all-stocks": "https://apewisdom.io/api/v1.0/filter/all-stocks",
    "all-crypto": "https://apewisdom.io/api/v1.0/filter/all-crypto",
    "all": "https://apewisdom.io/api/v1.0/filter/all",
}

MIN_MENTIONS = 5        # ignore tickers with negligible activity
MAX_PAGES_PER_FILTER = 20  # safety cap: 20 pages × 25 items = 500 items per filter

# Score weights: how much each momentum signal contributes
MENTION_WEIGHT = 0.40   # volume of discussion
UPVOTE_WEIGHT = 0.30    # quality / approval of that discussion
RANK_WEIGHT = 0.30      # relative trending position vs. other tickers


def fetch_apewisdom_data() -> list[dict]:
    """
    Fetch retail sentiment data from all three ApeWisdom filters.

    Paginates through every available page of all-stocks, all-crypto, and all.
    Deduplicates by ticker symbol — the first filter to report a ticker wins
    (all-stocks → all-crypto → all), so each ticker appears only once.
    Returns a flat list of raw item dicts enriched with a '_filter' key.
    """
    seen: set[str] = set()
    results: list[dict] = []

    for filter_name, url in APEWISDOM_FILTERS.items():
        page = 1
        while page <= MAX_PAGES_PER_FILTER:
            response = requests.get(url, params={"page": page}, timeout=15)
            response.raise_for_status()
            data = response.json()

            for item in data.get("results", []):
                ticker = (item.get("ticker") or "").upper().strip()
                if not ticker or ticker in seen:
                    continue
                seen.add(ticker)
                results.append({**item, "_filter": filter_name})

            nb_pages = int(data.get("nb_pages") or 1)
            if page >= nb_pages:
                break
            page += 1
            time.sleep(0.1)  # be polite to a free public API

    return results


def calculate_sentiment_score(item: dict) -> tuple[float, str]:
    """
    Compute a retail sentiment score from a single ApeWisdom ticker item.

    Three momentum signals are weighted and summed:
    - Mention momentum  (40%): is discussion volume rising vs. 24 h ago?
    - Upvote momentum   (30%): is content quality / approval rising?
    - Rank momentum     (30%): is the ticker climbing the trending rankings?

    Each component is clamped to [-1, 1] before weighting.
    Returns (score, direction) where score is -1.0 (cooling off) to 1.0 (surging).
    """
    mentions = item.get("mentions") or 0
    mentions_24h = item.get("mentions_24h_ago") or 0
    upvotes = item.get("upvotes") or 0
    upvotes_24h = item.get("upvotes_24h_ago") or 0
    rank = item.get("rank") or 0
    rank_24h = item.get("rank_24h_ago") or 0

    # Mention momentum: positive = more people talking about it
    if mentions_24h > 0:
        mention_delta = (mentions - mentions_24h) / mentions_24h
    elif mentions > 0:
        mention_delta = 1.0  # appeared from nothing → maximum buzz
    else:
        mention_delta = 0.0

    # Upvote momentum: positive = discussion quality improving
    if upvotes_24h > 0:
        upvote_delta = (upvotes - upvotes_24h) / upvotes_24h
    elif upvotes > 0:
        upvote_delta = 1.0
    else:
        upvote_delta = 0.0

    # Rank momentum: lower rank number = more popular, so rising = rank_24h > rank
    if rank_24h > 0 and rank > 0:
        rank_delta = (rank_24h - rank) / rank_24h
    elif rank_24h > 0 and rank == 0:
        rank_delta = -1.0  # dropped off the list entirely
    else:
        rank_delta = 0.0

    mention_delta = max(-1.0, min(1.0, mention_delta))
    upvote_delta = max(-1.0, min(1.0, upvote_delta))
    rank_delta = max(-1.0, min(1.0, rank_delta))

    score = (MENTION_WEIGHT * mention_delta
             + UPVOTE_WEIGHT * upvote_delta
             + RANK_WEIGHT * rank_delta)
    score = max(-1.0, min(1.0, score))

    if score > 0.1:
        direction = "bullish"
    elif score < -0.1:
        direction = "bearish"
    else:
        direction = "neutral"

    return float(score), direction


def get_or_create_ticker(db: Session, symbol: str) -> Ticker:
    """Get existing ticker or create a new one."""
    ticker = db.query(Ticker).filter(Ticker.symbol == symbol).first()
    if not ticker:
        ticker = Ticker(symbol=symbol)
        db.add(ticker)
        db.commit()
        db.refresh(ticker)
    return ticker


def store_signal(db: Session, ticker: Ticker, score: float, direction: str, source: str = "apewisdom") -> Signal:
    """Store the retail sentiment signal in the signals table."""
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


def run_apewisdom_pipeline() -> None:
    """
    Execute the full ApeWisdom retail sentiment pipeline.

    Steps:
      1. Fetch all-stocks, all-crypto, and all filters (paginated, deduplicated by ticker)
      2. Drop tickers below the minimum mentions threshold
      3. Compute mention / upvote / rank momentum score for every active ticker
      4. Store one signal per ticker in the signals table with source="apewisdom"
    """
    print("Starting ApeWisdom retail sentiment pipeline...")
    db = SessionLocal()

    try:
        print("Step 1: Fetching ApeWisdom data across all filters...")
        items = fetch_apewisdom_data()
        print(f"  {len(items)} unique tickers fetched")

        print(f"Step 2: Filtering to tickers with >= {MIN_MENTIONS} mentions...")
        active = [i for i in items if (i.get("mentions") or 0) >= MIN_MENTIONS]
        print(f"  {len(active)} active tickers retained")

        if not active:
            print("  No active tickers — aborting pipeline")
            return

        print("Step 3: Computing scores and storing signals...")
        n_bullish = n_bearish = n_neutral = 0

        for item in active:
            symbol = item["ticker"].upper()
            score, direction = calculate_sentiment_score(item)
            ticker_obj = get_or_create_ticker(db, symbol)
            store_signal(db, ticker_obj, score, direction)

            if direction == "bullish":
                n_bullish += 1
            elif direction == "bearish":
                n_bearish += 1
            else:
                n_neutral += 1

        print(
            f"  {len(active)} signals stored | "
            f"bullish={n_bullish} bearish={n_bearish} neutral={n_neutral}"
        )

    except Exception as e:
        print(f"Error in ApeWisdom pipeline: {e}")
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    run_apewisdom_pipeline()
