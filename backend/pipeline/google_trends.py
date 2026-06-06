from datetime import datetime, timedelta

from pytrends.request import TrendReq
from sqlalchemy.orm import Session

from backend.db.database import SessionLocal
from backend.db.models import Signal, Ticker


def fetch_google_trends_data(keyword: str = "AAPL", days: int = 90) -> dict:
    """Fetch Google Trends data for a keyword over the last N days."""
    pytrends = TrendReq(hl="en-US", tz=360)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    timeframe = f"{start_date.strftime('%Y-%m-%d')} {end_date.strftime('%Y-%m-%d')}"

    pytrends.build_payload(kw_list=[keyword], timeframe=timeframe)
    data = pytrends.interest_over_time()

    if data.empty:
        return None

    return data


def calculate_sentiment_score(data) -> tuple[float, str]:
    """
    Calculate sentiment score (-1 to 1) based on trend direction.
    Returns (score, direction).
    """
    if data is None or data.empty:
        return 0.0, "neutral"

    # Remove the 'isPartial' column if it exists
    if "isPartial" in data.columns:
        data = data.drop("isPartial", axis=1)

    # Get the interest values
    values = data.iloc[:, 0].values
    start_value = values[0]
    end_value = values[-1]

    # Calculate trend direction
    trend_change = end_value - start_value
    avg_value = values.mean()

    # Normalize score between -1 and 1
    if avg_value > 0:
        score = trend_change / avg_value
    else:
        score = 0.0

    score = max(-1.0, min(1.0, score))

    # Determine direction
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


def store_signal(db: Session, ticker: Ticker, score: float, direction: str, source: str = "google_trends"):
    """Store the signal in the database."""
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


def run_google_trends_pipeline():
    """Main pipeline function."""
    db = SessionLocal()
    try:
        # Fetch data
        data = fetch_google_trends_data("AAPL", days=90)
        if data is None:
            print("No Google Trends data available")
            return

        # Calculate sentiment
        score, direction = calculate_sentiment_score(data)
        print(f"Google Trends Score: {score:.2f}, Direction: {direction}")

        # Get or create ticker
        ticker = get_or_create_ticker(db, "AAPL")

        # Store signal
        signal = store_signal(db, ticker, score, direction)
        print(f"Signal stored: ID={signal.id}, Ticker={ticker.symbol}, Score={score:.2f}, Direction={direction}")

    except Exception as e:
        print(f"Error in Google Trends pipeline: {e}")
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    run_google_trends_pipeline()
