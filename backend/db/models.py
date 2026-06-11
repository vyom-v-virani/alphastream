from datetime import datetime

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class Ticker(Base):
    __tablename__ = "tickers"

    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), unique=True, nullable=False)

    signals = relationship("Signal", back_populates="ticker")


class Signal(Base):
    __tablename__ = "signals"

    id = Column(Integer, primary_key=True)
    ticker_id = Column(Integer, ForeignKey("tickers.id"), nullable=False)
    source = Column(String(50), nullable=False)
    score = Column(Float, nullable=False)
    direction = Column(String(20), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)

    ticker = relationship("Ticker", back_populates="signals")
    label = relationship("SignalLabel", back_populates="signal", uselist=False)


class WeatherObservation(Base):
    __tablename__ = "weather_observations"

    id = Column(Integer, primary_key=True)
    state_fips = Column(String(10), nullable=False)
    state_name = Column(String(50), nullable=False)
    date = Column(DateTime, nullable=False)
    tmax_celsius = Column(Float)
    tmin_celsius = Column(Float)
    prcp_mm = Column(Float)
    fetched_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class EarningsTranscript(Base):
    __tablename__ = "earnings_transcripts"

    id = Column(Integer, primary_key=True)
    ticker_id = Column(Integer, ForeignKey("tickers.id"), nullable=False)
    filing_date = Column(DateTime, nullable=False)
    accession_number = Column(String(50), nullable=False, unique=True)
    document_url = Column(String(500))
    cleaned_text = Column(Text, nullable=False)
    ceo_cfo_text = Column(Text)
    fetched_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class SignalLabel(Base):
    """
    Ground-truth outcome labels for an earnings-derived Signal.

    Links a Signal to its measured stock-price outcomes at 5, 30, and 90
    trading days after the filing date. Thresholds are computed from the
    stock's own rolling return distribution (not a fixed constant) so that
    'significant' is calibrated to AAPL's actual volatility at the time.
    """
    __tablename__ = "signal_labels"

    id = Column(Integer, primary_key=True)
    signal_id = Column(Integer, ForeignKey("signals.id"), nullable=False, unique=True)
    transcript_id = Column(Integer, ForeignKey("earnings_transcripts.id"), nullable=True, unique=True)
    ticker_id = Column(Integer, ForeignKey("tickers.id"), nullable=False)
    filing_date = Column(DateTime, nullable=False)
    price_at_filing = Column(Float)           # first close on or after filing date

    # ── 5 trading-day outcome ───────────────────────────────────────────────
    price_5d = Column(Float)                  # close price 5 trading days later
    return_5d = Column(Float)                 # (price_5d - price_at_filing) / price_at_filing
    label_5d = Column(String(10))             # "bullish" | "bearish" | "neutral"
    threshold_5d = Column(Float)              # 1-sigma threshold used for this filing

    # ── 30 trading-day outcome ──────────────────────────────────────────────
    price_30d = Column(Float)
    return_30d = Column(Float)
    label_30d = Column(String(10))
    threshold_30d = Column(Float)

    # ── 90 trading-day outcome ──────────────────────────────────────────────
    price_90d = Column(Float)
    return_90d = Column(Float)
    label_90d = Column(String(10))
    threshold_90d = Column(Float)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    signal = relationship("Signal", back_populates="label")
