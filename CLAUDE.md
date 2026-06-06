# AlphaStream

## What this is

A quantitative alternative data platform that aggregates multiple signal sources (earnings call transcripts, Reddit/forums, weather data, Google Trends, options flow) and uses ML/NLP to generate trade signals and conviction scores for options and futures markets.

## Architecture

- **Frontend**: Next.js + TypeScript + Tailwind, deployed on Vercel
- **API Layer**: FastAPI (Python)
- **Databases**: PostgreSQL (structured data), InfluxDB (time series), pgvector (RAG embeddings)
- **Pipeline**: Kafka + Python consumers per data source
- **ML Layer**: FinBERT (earnings/Fed), VADER (Reddit), XGBoost (weather/trends), LSTM meta model
- **LLM Layer**: LangChain + RAG for grounded trade narrative generation
- **Deployment**: AWS (EC2, RDS, S3), Vercel (frontend)

## Current Phase

Phase 1 — Building one signal end to end:
Reddit sentiment for AAPL → Python processing → PostgreSQL → FastAPI → Next.js dashboard

## Project Structure (target)

alphastream/
├── backend/
│ ├── api/ # FastAPI routes
│ ├── pipeline/ # Kafka consumers per source
│ ├── models/ # ML models
│ ├── db/ # Database connections and schemas
│ └── llm/ # LangChain + RAG layer
├── frontend/ # Next.js app
└── docker-compose.yml

## Conventions

- Python 3.11+
- All backend in Python
- Async FastAPI endpoints
- Pydantic models for all data validation
- SQLAlchemy for PostgreSQL ORM
- Environment variables via .env file, never hardcoded
- Each pipeline step (fetch, clean, transform, store) in separate functions
