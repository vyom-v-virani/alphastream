"""
LangChain pipeline that turns a ticker symbol into a plain-English trade
narrative, combining every signal source with the most recent earnings-call
transcript.

Pipeline, given a ticker symbol:
  1. Most recent earnings transcript for the ticker (earnings_transcripts table)
  2. Latest signal (score + direction) per source from the signals table
  3. LSTM meta-model prediction for the ticker (backend/models/lstm_meta.py)
  4. LangChain chain: summarize each signal source individually, then combine
     everything into one trade thesis narrative, via Anthropic claude-sonnet-4-6

Saved artefacts: none — this pipeline only reads and returns narrative text;
there is no model to persist.
"""

from datetime import datetime

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.db.database import SessionLocal
from backend.db.models import EarningsTranscript, Signal, Ticker
from backend.models import lstm_meta

load_dotenv()

NARRATIVE_MODEL = "claude-sonnet-4-6"
NARRATIVE_TEMPERATURE = 0.3   # low but non-zero — plain-English narrative, not deterministic extraction
MAX_TRANSCRIPT_CHARS = 6000   # keeps the raw transcript excerpt a reasonable size inside the prompt


# ── earnings transcript ────────────────────────────────────────────────────────

def fetch_latest_transcript(db: Session, ticker_id: int) -> EarningsTranscript | None:
    """Most recent earnings transcript on file for this ticker, if any."""
    return (
        db.query(EarningsTranscript)
        .filter(EarningsTranscript.ticker_id == ticker_id)
        .order_by(EarningsTranscript.filing_date.desc())
        .first()
    )


# ── signals + LSTM meta model ─────────────────────────────────────────────────

def fetch_latest_signals_by_source(db: Session, ticker_id: int) -> dict[str, dict]:
    """Latest (score, direction, timestamp) per source for this ticker."""
    latest_per_source = (
        db.query(Signal.source, func.max(Signal.timestamp).label("max_ts"))
        .filter(Signal.ticker_id == ticker_id)
        .group_by(Signal.source)
        .subquery()
    )
    rows = (
        db.query(Signal)
        .join(
            latest_per_source,
            (Signal.source == latest_per_source.c.source) & (Signal.timestamp == latest_per_source.c.max_ts),
        )
        .filter(Signal.ticker_id == ticker_id)
        .all()
    )
    return {
        row.source: {"score": row.score, "direction": row.direction, "timestamp": row.timestamp}
        for row in rows
    }


def fetch_lstm_meta_output(latest_signals: dict[str, dict]) -> dict | None:
    """
    Run the LSTM meta model (backend/models/lstm_meta.py) on the current
    per-source scores. Returns None if no trained model exists yet or none
    of the current signals overlap with the sources it was trained on.
    """
    scores = {
        source: info["score"]
        for source, info in latest_signals.items()
        if source in lstm_meta.SOURCES
    }
    if not scores:
        return None

    try:
        return lstm_meta.predict(scores)
    except FileNotFoundError:
        return None


# ── LangChain prompts ──────────────────────────────────────────────────────────

SOURCE_SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a quantitative analyst summarizing one alternative-data signal "
        "source for a trader. Be concise: 1-2 plain-English sentences, no jargon, "
        "no restating the raw numbers verbatim.",
    ),
    (
        "human",
        "Ticker: {ticker}\n"
        "Source: {source}\n"
        "Raw score: {score:+.3f} (range -1 bearish to +1 bullish)\n"
        "Direction: {direction}\n\n"
        "Explain in 1-2 sentences what this source is indicating about {ticker} "
        "and why a trader should care.",
    ),
])

COMBINE_NARRATIVE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a senior quant strategist writing a trade narrative for a "
        "trader. Combine the latest earnings-call context, the individual "
        "signal-source summaries, and the meta-model output into one coherent "
        "thesis. Explain the overall market context, what each data source is "
        "saying, and how they connect into a conviction-weighted view. Plain "
        "English, no academic tone, 3-5 short paragraphs.",
    ),
    (
        "human",
        "Ticker: {ticker}\n\n"
        "Most recent earnings-call context:\n{transcript_context}\n\n"
        "Per-source signal summaries:\n{source_summaries}\n\n"
        "LSTM meta-model output: {meta_summary}\n\n"
        "Write the unified trade thesis narrative.",
    ),
])


def _format_transcript_context(transcript: EarningsTranscript | None) -> str:
    if transcript is None:
        return "No earnings transcripts available yet for this ticker."
    text = (transcript.ceo_cfo_text or transcript.cleaned_text or "").strip()
    if not text:
        return "No earnings transcripts available yet for this ticker."
    return f"[{transcript.filing_date.date()}] {text[:MAX_TRANSCRIPT_CHARS]}"


def _format_source_summaries(summaries: dict[str, str]) -> str:
    if not summaries:
        return "No signal sources available yet for this ticker."
    return "\n".join(f"- {source}: {summary}" for source, summary in summaries.items())


def _format_meta_summary(meta_output: dict | None) -> str:
    if meta_output is None:
        return "Not available (LSTM meta model not yet trained, or no overlapping sources)."
    return (
        f"direction={meta_output['direction']}, "
        f"win_probability={meta_output['win_probability']:.2%}, "
        f"conviction={meta_output['conviction']:+.4f}"
    )


# ── orchestrator ──────────────────────────────────────────────────────────────

def generate_trade_narrative(ticker_symbol: str, db: Session) -> dict:
    """
    End-to-end: given a ticker symbol, gather the latest transcript, signals,
    and LSTM meta-model output, and run the LangChain summarize-then-combine chain.
    """
    ticker_symbol = ticker_symbol.upper()
    ticker = db.query(Ticker).filter(Ticker.symbol == ticker_symbol).first()
    if not ticker:
        raise ValueError(f"Ticker {ticker_symbol} not found in the database")

    llm = ChatAnthropic(model=NARRATIVE_MODEL, temperature=NARRATIVE_TEMPERATURE, max_tokens=1024)
    summary_chain = SOURCE_SUMMARY_PROMPT | llm | StrOutputParser()
    combine_chain = COMBINE_NARRATIVE_PROMPT | llm | StrOutputParser()

    # Step 1: most recent earnings transcript
    latest_transcript = fetch_latest_transcript(db, ticker.id)

    # Step 2: latest signal per source
    latest_signals = fetch_latest_signals_by_source(db, ticker.id)

    # Step 3: LSTM meta-model output
    meta_output = fetch_lstm_meta_output(latest_signals)

    # Step 4a: summarize each source individually
    source_summaries: dict[str, str] = {}
    for source, info in latest_signals.items():
        source_summaries[source] = summary_chain.invoke({
            "ticker": ticker_symbol,
            "source": source,
            "score": info["score"],
            "direction": info["direction"],
        }).strip()

    # Step 4b: combine into one trade thesis narrative
    narrative = combine_chain.invoke({
        "ticker": ticker_symbol,
        "transcript_context": _format_transcript_context(latest_transcript),
        "source_summaries": _format_source_summaries(source_summaries),
        "meta_summary": _format_meta_summary(meta_output),
    }).strip()

    return {
        "ticker": ticker_symbol,
        "latest_transcript": latest_transcript,
        "latest_signals": latest_signals,
        "source_summaries": source_summaries,
        "lstm_meta": meta_output,
        "narrative": narrative,
        "generated_at": datetime.utcnow().isoformat(),
    }


def run_narrative_pipeline(ticker_symbol: str = "AAPL") -> None:
    """
    End-to-end narrative generation pipeline for a single ticker.

    Steps:
      1. Fetch the most recent earnings transcript on file for this ticker
      2. Fetch the latest signal per source and the LSTM meta-model output
      3. Summarize each source, then combine into a unified trade narrative
    """
    print(f"Starting narrative pipeline for {ticker_symbol}...")
    db = SessionLocal()
    try:
        result = generate_trade_narrative(ticker_symbol, db)

        print(f"\n=== {result['ticker']} Trade Thesis ===")
        transcript = result["latest_transcript"]
        print(f"Latest transcript: {transcript.filing_date.date() if transcript else 'none on file'}")
        print(f"Signal sources: {list(result['source_summaries'])}")
        print(f"LSTM meta: {_format_meta_summary(result['lstm_meta'])}")
        print("\n--- Narrative ---")
        print(result["narrative"])

    except Exception as exc:
        print(f"Error in narrative pipeline: {exc}")
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    run_narrative_pipeline()
