from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf
from sqlalchemy.orm import Session

from backend.db.database import SessionLocal
from backend.db.models import Signal, Ticker

# Classification thresholds
MIN_VOLUME = 100            # minimum contracts traded to be "notable"
LARGE_ORDER_VOLUME = 500    # contracts threshold for a "large order"
UNUSUAL_VOI_RATIO = 1.5     # vol/OI where positioning is clearly fresh
SWEEP_VOI_RATIO = 2.0       # higher vol/OI required for sweep classification
SWEEP_MAX_DTE = 30          # sweeps are short-dated (days to expiry)
SWEEP_DELTA_LOW = 0.25      # sweep contracts are near ATM
SWEEP_DELTA_HIGH = 0.75
OTM_DELTA_THRESHOLD = 0.30  # |delta| < this = far out-of-the-money speculation

# Conviction multipliers applied when computing weighted score
MULTIPLIER_SWEEP = 2.0
MULTIPLIER_LARGE_ORDER = 1.5
MULTIPLIER_UNUSUAL_VOI = 1.2
MULTIPLIER_OTM_SPEC = 0.8   # reduce weight for speculative far-OTM trades



def fetch_options_chain(ticker: str = "AAPL") -> list[dict]:
    """
    Fetch current options chain for the given ticker using yfinance.

    Collects all expirations within the next 90 days, fetches calls and puts
    for each expiry, and returns a flat list of contract dicts in the same
    nested format clean_options_data expects.
    Note: yfinance does not provide Greeks; delta will be None for all contracts,
    so is_sweep will never fire (requires delta). All other flags still work.
    """
    today = datetime.now().date()
    expiry_max = today + timedelta(days=90)

    yf_ticker = yf.Ticker(ticker)

    underlying_price = None
    try:
        underlying_price = yf_ticker.fast_info["last_price"]
    except Exception:
        pass

    contracts = []

    for expiry_str in yf_ticker.options:
        try:
            expiry_date = datetime.strptime(expiry_str, "%Y-%m-%d").date()
        except ValueError:
            continue

        if expiry_date > expiry_max:
            break  # yfinance returns expirations in ascending order

        chain = yf_ticker.option_chain(expiry_str)

        for contract_type, df in [("call", chain.calls), ("put", chain.puts)]:
            for _, row in df.iterrows():
                contracts.append({
                    "day": {
                        "volume": int(row["volume"]) if pd.notna(row.get("volume")) else None,
                        "vwap": None,
                    },
                    "details": {
                        "contract_type": contract_type,
                        "strike_price": float(row["strike"]),
                        "expiration_date": expiry_str,
                        "ticker": row.get("contractSymbol", ""),
                    },
                    "greeks": {"delta": None},
                    "implied_volatility": float(row["impliedVolatility"]) if pd.notna(row.get("impliedVolatility")) else None,
                    "open_interest": int(row["openInterest"]) if pd.notna(row.get("openInterest")) else None,
                    "underlying_asset": {"price": underlying_price},
                })

    return contracts


def clean_options_data(raw_contracts: list[dict]) -> list[dict]:
    """
    Filter and normalize raw Polygon options snapshots.

    Drops contracts with zero volume, missing core fields, or already-expired expirations.
    Normalises each contract into a flat dict with consistent field names.
    """
    cleaned = []
    today = datetime.now().date()

    for c in raw_contracts:
        day = c.get("day") or {}
        details = c.get("details") or {}
        greeks = c.get("greeks") or {}
        underlying = c.get("underlying_asset") or {}

        volume = day.get("volume") or 0
        open_interest = c.get("open_interest") or 0
        contract_type = details.get("contract_type")
        strike = details.get("strike_price")
        expiry_str = details.get("expiration_date")
        delta = greeks.get("delta")
        iv = c.get("implied_volatility")
        underlying_price = underlying.get("price")

        if volume == 0 or not expiry_str or not contract_type or strike is None:
            continue

        try:
            expiry_date = datetime.strptime(expiry_str, "%Y-%m-%d").date()
        except ValueError:
            continue

        dte = (expiry_date - today).days
        if dte < 0:
            continue

        # vol/OI ratio: float("inf") when OI is 0 (all volume is brand-new positioning)
        voi_ratio = volume / open_interest if open_interest > 0 else float("inf")

        cleaned.append({
            "contract_ticker": details.get("ticker", ""),
            "contract_type": contract_type,       # "call" | "put"
            "strike": float(strike),
            "expiry": expiry_str,
            "dte": dte,
            "volume": int(volume),
            "open_interest": int(open_interest),
            "voi_ratio": voi_ratio,
            "delta": delta,
            "iv": iv,
            "underlying_price": underlying_price,
            "vwap": day.get("vwap"),
        })

    return cleaned


def classify_options_flow(contracts: list[dict]) -> list[dict]:
    """
    Attach classification flags and a directional sentiment to each cleaned contract.

    Flags:
    - large_order    : volume >= LARGE_ORDER_VOLUME (institutional-scale)
    - unusual_voi    : vol/OI ratio >= UNUSUAL_VOI_RATIO (fresh aggressive positioning)
    - sweep          : high vol/OI + near-ATM delta + short DTE — urgent directional bet.
                       True sweeps require trade-level data; this is a snapshot heuristic.
    - otm_spec       : |delta| < OTM_DELTA_THRESHOLD — speculative low-probability trade

    sentiment: +1 for calls (bullish), -1 for puts (bearish)
    """
    classified = []

    for c in contracts:
        delta = c["delta"]
        abs_delta = abs(delta) if delta is not None else None

        is_large = c["volume"] >= LARGE_ORDER_VOLUME
        is_unusual_voi = c["voi_ratio"] >= UNUSUAL_VOI_RATIO

        is_sweep = (
            c["voi_ratio"] >= SWEEP_VOI_RATIO
            and c["dte"] <= SWEEP_MAX_DTE
            and abs_delta is not None
            and SWEEP_DELTA_LOW <= abs_delta <= SWEEP_DELTA_HIGH
        )

        is_otm_spec = abs_delta is not None and abs_delta < OTM_DELTA_THRESHOLD

        sentiment = 1 if c["contract_type"] == "call" else -1

        classified.append({
            **c,
            "is_large_order": is_large,
            "is_unusual_voi": is_unusual_voi,
            "is_sweep": is_sweep,
            "is_otm_spec": is_otm_spec,
            "sentiment": sentiment,
        })

    return classified


def calculate_sentiment_score(classified: list[dict]) -> tuple[float, str]:
    """
    Compute a volume-weighted directional score from classified options flow.

    Only "notable" contracts contribute: those meeting LARGE_ORDER_VOLUME or
    UNUSUAL_VOI_RATIO thresholds. Sweeps and large orders receive higher conviction
    multipliers; far-OTM speculation is down-weighted.

    Returns (score, direction) where score is -1.0 (strongly bearish) to 1.0 (strongly bullish).
    """
    notable = [c for c in classified if c["is_large_order"] or c["is_unusual_voi"]]

    if not notable:
        return 0.0, "neutral"

    weighted_score = 0.0
    total_weight = 0.0

    for c in notable:
        base_weight = float(c["volume"])

        multiplier = 1.0
        if c["is_sweep"]:
            multiplier *= MULTIPLIER_SWEEP
        if c["is_large_order"]:
            multiplier *= MULTIPLIER_LARGE_ORDER
        if c["is_unusual_voi"]:
            multiplier *= MULTIPLIER_UNUSUAL_VOI
        if c["is_otm_spec"]:
            multiplier *= MULTIPLIER_OTM_SPEC

        effective_weight = base_weight * multiplier
        weighted_score += c["sentiment"] * effective_weight
        total_weight += effective_weight

    if total_weight == 0:
        return 0.0, "neutral"

    score = max(-1.0, min(1.0, weighted_score / total_weight))

    if score > 0.15:
        direction = "bullish"
    elif score < -0.15:
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


def store_signal(db: Session, ticker: Ticker, score: float, direction: str, source: str = "options_flow") -> Signal:
    """Store the options flow signal in the signals table."""
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


def run_options_flow_pipeline() -> None:
    """
    Execute the full options flow → classification → signal pipeline for AAPL.

    Steps:
      1. Fetch AAPL options chain from yfinance (next 90 days of expirations)
      2. Clean: drop zero-volume / incomplete contracts, normalise fields
      3. Classify: flag large orders, unusual vol/OI, sweeps, OTM speculation
      4. Score: volume-weighted directional sentiment across notable contracts
      5. Store signal in the signals table with source="options_flow"
    """
    print("Starting options flow pipeline for AAPL...")
    db = SessionLocal()

    try:
        print("Step 1: Fetching AAPL options chain from yfinance...")
        raw = fetch_options_chain("AAPL")
        print(f"  {len(raw)} raw contracts fetched")

        print("Step 2: Cleaning options data...")
        cleaned = clean_options_data(raw)
        print(f"  {len(cleaned)} contracts after cleaning")

        if not cleaned:
            print("  No valid contracts — aborting pipeline")
            return

        print("Step 3: Classifying options flow...")
        classified = classify_options_flow(cleaned)
        n_sweep = sum(1 for c in classified if c["is_sweep"])
        n_large = sum(1 for c in classified if c["is_large_order"])
        n_unusual = sum(1 for c in classified if c["is_unusual_voi"])
        n_notable = sum(1 for c in classified if c["is_large_order"] or c["is_unusual_voi"])
        print(f"  Notable: {n_notable} | Sweeps: {n_sweep} | Large orders: {n_large} | Unusual vol/OI: {n_unusual}")

        print("Step 4: Calculating sentiment score...")
        score, direction = calculate_sentiment_score(classified)
        print(f"  Score: {score:.4f} | Direction: {direction}")

        print("Step 5: Storing signal...")
        ticker = get_or_create_ticker(db, "AAPL")
        signal = store_signal(db, ticker, score, direction)
        print(f"  Signal stored: ID={signal.id}, Ticker=AAPL, Score={score:.4f}, Direction={direction}")

    except Exception as e:
        print(f"Error in options flow pipeline: {e}")
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    run_options_flow_pipeline()
