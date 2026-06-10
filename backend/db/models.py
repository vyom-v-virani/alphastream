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
