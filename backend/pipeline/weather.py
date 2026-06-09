import os
import time
from collections import defaultdict
from datetime import datetime, timedelta

import requests
from dotenv import load_dotenv
from sqlalchemy.orm import Session

from backend.db.database import SessionLocal
from backend.db.models import Signal, Ticker, WeatherObservation

load_dotenv()

NOAA_API_KEY = os.getenv("NOAA_API_KEY")
NOAA_CDO_URL = "https://www.ncdc.noaa.gov/cdo-web/api/v2/data"

CORN_BELT_STATES = {
    "Iowa": "FIPS:19",
    "Illinois": "FIPS:17",
    "Indiana": "FIPS:18",
    "Nebraska": "FIPS:31",
}

# Approximate share of 4-state corn production (USDA)
STATE_PRODUCTION_WEIGHTS = {
    "Iowa": 0.30,
    "Illinois": 0.27,
    "Indiana": 0.18,
    "Nebraska": 0.25,
}

# Growing phase weights: (temperature_weight, precipitation_weight) per month.
# July (7) = pollination — heat stress dominates; temperature weight is highest.
PHASE_WEIGHTS = {
    5: (0.40, 0.60),
    6: (0.40, 0.60),
    7: (0.70, 0.30),
    8: (0.40, 0.60),
    9: (0.30, 0.70),
}


def fetch_weather_data(days: int = 30) -> dict:
    """
    Fetch last N days of TMAX, TMIN, PRCP from NOAA CDO (GHCND) for corn belt states.

    Returns dict keyed by state name with list of raw station records.
    GHCND stores values in tenths: TMAX/TMIN in tenths of °C, PRCP in tenths of mm.
    """
    if not NOAA_API_KEY:
        raise ValueError("NOAA_API_KEY not found in environment variables")

    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days)
    headers = {"token": NOAA_API_KEY}
    results = {}

    for state_name, fips_id in CORN_BELT_STATES.items():
        print(f"  Fetching NOAA data for {state_name} ({fips_id})...")
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
        time.sleep(0.25)  # stay within NOAA's 5 req/sec limit

    return results


def aggregate_state_weather(raw_data: dict) -> dict:
    """
    Average TMAX, TMIN, PRCP across all stations per state per calendar day.

    GHCND stores values in tenths: divides by 10.0 to yield °C and mm.
    Returns dict[state_name][YYYY-MM-DD] = {tmax_c, tmin_c, prcp_mm}.
    """
    aggregated = {}

    for state, records in raw_data.items():
        buckets: dict[str, dict[str, list]] = defaultdict(lambda: {"TMAX": [], "TMIN": [], "PRCP": []})

        for rec in records:
            date = rec["date"][:10]
            dtype = rec["datatype"]
            if dtype in buckets[date]:
                buckets[date][dtype].append(rec["value"])

        state_daily = {}
        for date, readings in buckets.items():
            state_daily[date] = {
                "tmax_c": (sum(readings["TMAX"]) / len(readings["TMAX"]) / 10.0) if readings["TMAX"] else None,
                "tmin_c": (sum(readings["TMIN"]) / len(readings["TMIN"]) / 10.0) if readings["TMIN"] else None,
                "prcp_mm": (sum(readings["PRCP"]) / len(readings["PRCP"]) / 10.0) if readings["PRCP"] else None,
            }

        aggregated[state] = state_daily

    return aggregated


def _score_temperature(tmax_c: float, tmin_c: float, month: int) -> float:
    """
    Score daily temperature for corn. Returns -1.0 to 1.0.

    Frost (<= 0°C) during growing season is catastrophic.
    Heat stress (> 35°C tmax) destroys pollen viability.
    Optimal average temperature: 18–28°C.
    """
    if tmax_c is None or tmin_c is None:
        return 0.0

    tavg = (tmax_c + tmin_c) / 2.0

    if tmin_c <= 0.0 and 5 <= month <= 9:
        return -1.0

    if tmax_c > 35.0:
        return -min((tmax_c - 35.0) / 8.0, 1.0)

    if 18.0 <= tavg <= 28.0:
        return 0.8
    if 12.0 <= tavg < 18.0:
        return 0.3
    if 28.0 < tavg <= 35.0:
        return 0.4
    return -0.2


def _score_precipitation(prcp_mm: float, month: int) -> float:
    """
    Score average daily precipitation for corn. Returns -1.0 to 1.0.

    Optimal: 4–8 mm/day. Drought (< 2 mm/day) and flooding (> 15 mm/day) are penalized.
    """
    if prcp_mm is None or month not in PHASE_WEIGHTS:
        return 0.0

    if 4.0 <= prcp_mm <= 8.0:
        return 0.9
    if 2.0 <= prcp_mm < 4.0:
        return 0.3
    if 8.0 < prcp_mm <= 15.0:
        return 0.2
    if prcp_mm < 2.0:
        return -0.7
    return -0.5  # > 15 mm/day: flooding / excess moisture risk


def calculate_corn_impact_score(aggregated: dict) -> tuple[float, str]:
    """
    Compute weighted crop-quality score from aggregated state weather data.

    Weights state production share against monthly growing-phase weights
    (temperature vs. precipitation importance shifts through the season).

    Returns (crop_score, label) where crop_score is -1.0 (severe damage) to
    1.0 (ideal conditions). A high crop score implies abundant supply →
    bearish ZC futures price; caller should negate for a price-direction signal.
    """
    current_month = datetime.now().month
    temp_weight, prcp_weight = PHASE_WEIGHTS.get(current_month, (0.50, 0.50))

    weighted_crop_score = 0.0
    total_weight = 0.0

    for state, daily in aggregated.items():
        if not daily:
            continue

        state_weight = STATE_PRODUCTION_WEIGHTS[state]
        temp_scores = []
        prcp_values = []

        for date_str, readings in daily.items():
            month = int(date_str[5:7])
            temp_scores.append(_score_temperature(readings["tmax_c"], readings["tmin_c"], month))
            if readings["prcp_mm"] is not None:
                prcp_values.append(readings["prcp_mm"])

        if not temp_scores:
            continue

        avg_temp_score = sum(temp_scores) / len(temp_scores)
        avg_prcp_mm = sum(prcp_values) / len(prcp_values) if prcp_values else 0.0
        prcp_score = _score_precipitation(avg_prcp_mm, current_month)

        state_score = (temp_weight * avg_temp_score) + (prcp_weight * prcp_score)
        weighted_crop_score += state_score * state_weight
        total_weight += state_weight

    if total_weight == 0:
        return 0.0, "neutral"

    crop_score = max(-1.0, min(1.0, weighted_crop_score / total_weight))
    label = "favorable" if crop_score > 0.15 else ("unfavorable" if crop_score < -0.15 else "neutral")
    return float(crop_score), label


def get_or_create_ticker(db: Session, symbol: str) -> Ticker:
    """Get existing ticker or create a new one."""
    ticker = db.query(Ticker).filter(Ticker.symbol == symbol).first()
    if not ticker:
        ticker = Ticker(symbol=symbol)
        db.add(ticker)
        db.commit()
        db.refresh(ticker)
    return ticker


def store_weather_observations(db: Session, aggregated: dict) -> int:
    """Persist daily averaged station readings to the weather_observations table."""
    count = 0
    fetched_at = datetime.utcnow()

    for state, daily in aggregated.items():
        fips = CORN_BELT_STATES[state]
        for date_str, readings in daily.items():
            obs = WeatherObservation(
                state_fips=fips,
                state_name=state,
                date=datetime.strptime(date_str, "%Y-%m-%d"),
                tmax_celsius=readings["tmax_c"],
                tmin_celsius=readings["tmin_c"],
                prcp_mm=readings["prcp_mm"],
                fetched_at=fetched_at,
            )
            db.add(obs)
            count += 1

    db.commit()
    return count


def store_signal(db: Session, ticker: Ticker, score: float, direction: str, source: str = "weather") -> Signal:
    """Store the ZC weather-derived signal in the signals table."""
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


def run_weather_pipeline() -> None:
    """
    Execute the full weather → corn impact → ZC signal pipeline.

    Steps:
      1. Fetch NOAA GHCND data for Iowa, Illinois, Indiana, Nebraska
      2. Aggregate station readings by state and calendar day
      3. Score crop impact (temperature + precipitation, phase-weighted)
      4. Store raw observations in weather_observations
      5. Invert crop score to ZC price-direction signal and store in signals
    """
    print("Starting weather pipeline for ZC (corn futures)...")
    db = SessionLocal()

    try:
        print("Step 1: Fetching NOAA weather data...")
        raw_data = fetch_weather_data(days=30)

        print("Step 2: Aggregating station readings by state and day...")
        aggregated = aggregate_state_weather(raw_data)
        total_obs = sum(len(v) for v in aggregated.values())
        print(f"  {total_obs} state-day observations aggregated")

        print("Step 3: Calculating corn crop-impact score...")
        crop_score, crop_label = calculate_corn_impact_score(aggregated)
        # Invert: favorable growing conditions (positive crop score) = bearish ZC price
        signal_score = -crop_score
        direction = "bullish" if signal_score > 0.15 else ("bearish" if signal_score < -0.15 else "neutral")
        print(f"  Crop quality: {crop_score:.4f} ({crop_label}) → ZC signal: {signal_score:.4f} ({direction})")

        print("Step 4: Storing weather observations...")
        obs_count = store_weather_observations(db, aggregated)
        print(f"  {obs_count} rows written to weather_observations")

        print("Step 5: Storing ZC signal...")
        ticker = get_or_create_ticker(db, "ZC")
        signal = store_signal(db, ticker, signal_score, direction)
        print(f"  Signal stored: ID={signal.id}, Ticker=ZC, Score={signal_score:.4f}, Direction={direction}")

    except Exception as e:
        print(f"Error in weather pipeline: {e}")
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    run_weather_pipeline()
