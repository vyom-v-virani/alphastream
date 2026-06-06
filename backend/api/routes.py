from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.db.database import get_db
from backend.db.models import Signal, Ticker

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


class SignalResponse(BaseModel):
    id: int
    ticker: str
    source: str
    score: float
    direction: str
    timestamp: str

    class Config:
        from_attributes = True


@app.get("/signals/{ticker}", response_model=SignalResponse)
def get_latest_signal(ticker: str, db: Session = Depends(get_db)):
    ticker_row = db.query(Ticker).filter(Ticker.symbol == ticker.upper()).first()
    if not ticker_row:
        raise HTTPException(status_code=404, detail=f"Ticker {ticker.upper()} not found")

    signal = (
        db.query(Signal)
        .filter(Signal.ticker_id == ticker_row.id)
        .order_by(Signal.timestamp.desc())
        .first()
    )
    if not signal:
        raise HTTPException(status_code=404, detail=f"No signals found for {ticker.upper()}")

    return SignalResponse(
        id=signal.id,
        ticker=ticker_row.symbol,
        source=signal.source,
        score=signal.score,
        direction=signal.direction,
        timestamp=signal.timestamp.isoformat(),
    )
