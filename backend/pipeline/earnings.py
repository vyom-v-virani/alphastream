import re
import time
from datetime import datetime

import nltk
import requests
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sqlalchemy.orm import Session

from backend.db.database import SessionLocal
from backend.db.models import EarningsTranscript, Signal, Ticker

nltk.download("vader_lexicon", quiet=True)

# SEC EDGAR requires a descriptive User-Agent with a contact email
HEADERS = {"User-Agent": "AlphaStream/1.0 vyomvirani@gmail.com"}

AAPL_CIK = "0000320193"
SEC_BASE_URL = "https://www.sec.gov"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"

# Speaker weighting for VADER scoring
WEIGHT_EXEC = 2.0      # CEO / CFO — primary signal source
WEIGHT_ANALYST = 0.5   # buy-side / sell-side questions — lower conviction
WEIGHT_DEFAULT = 1.0   # operator, moderator, unattributed text

# Apple known executive names (lowercase for matching)
APPLE_EXEC_NAMES = {"tim cook", "luca maestri", "kevan parekh"}

# Matches title-case or ALL-CAPS speaker headers such as:
#   "Tim Cook:"  |  "Tim Cook - CEO:"  |  "OPERATOR:"  |  "John Smith - Goldman Sachs:"
_SPEAKER_RE = re.compile(
    r"^([A-Z][A-Za-z\-\.']+(?:\s+[A-Z][A-Za-z\-\.']+){0,4})"
    r"(?:\s*[\-–]\s*[A-Za-z ,\.]+)?\s*:\s*$"
)


def _classify_speaker_weight(speaker: str) -> float:
    """Return the conviction weight for a speaker string."""
    lower = speaker.lower()
    if any(name in lower for name in APPLE_EXEC_NAMES):
        return WEIGHT_EXEC
    if re.search(r"\b(ceo|chief executive|cfo|chief financial)\b", lower):
        return WEIGHT_EXEC
    if re.search(r"\b(analyst|equity|portfolio|capital|bank|securities|research)\b", lower):
        return WEIGHT_ANALYST
    return WEIGHT_DEFAULT


def fetch_latest_earnings_8k(cik: str = AAPL_CIK) -> dict | None:
    """
    Find the most recent 8-K with earnings content for the given CIK.

    Queries the SEC EDGAR submissions API and returns the first 8-K whose
    items list contains 2.02 (Results of Operations). Falls back to the
    most recent 8-K if no 2.02 filing is found within the recent batch.
    Returns dict with accession_no and filing_date, or None.
    """
    url = SUBMISSIONS_URL.format(cik=cik)
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    time.sleep(0.15)

    recent = resp.json().get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accessions = recent.get("accessionNumber", [])
    dates = recent.get("filingDate", [])
    items_list = recent.get("items", [])

    # Prefer 8-K with Item 2.02 (Results of Operations = earnings release date)
    for form, acc, date, items in zip(forms, accessions, dates, items_list):
        if form == "8-K" and "2.02" in (items or ""):
            return {"accession_no": acc, "filing_date": date}

    # Fallback: most recent 8-K of any type
    for form, acc, date in zip(forms, accessions, dates):
        if form == "8-K":
            return {"accession_no": acc, "filing_date": date}

    return None


def fetch_filing_document(cik: str, accession_no: str) -> tuple[str, str]:
    """
    Download the most relevant HTML document from an EDGAR 8-K filing.

    Fetches the filing index and selects:
      transcript exhibit  >  any EX-99 exhibit  >  8-K primary doc.
    Returns (html_text, document_url). Returns ("", "") on failure.
    """
    acc_clean = accession_no.replace("-", "")
    cik_plain = cik.lstrip("0") or cik  # Archives URLs use unpadded CIK

    index_url = (
        f"{SEC_BASE_URL}/Archives/edgar/data/{cik_plain}"
        f"/{acc_clean}/{accession_no}-index.htm"
    )

    resp = requests.get(index_url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    time.sleep(0.15)

    soup = BeautifulSoup(resp.text, "html.parser")

    docs: list[dict] = []
    for tr in soup.find_all("tr"):
        tds = tr.find_all("td")
        a = tr.find("a", href=True)
        if not a or len(tds) < 2:
            continue
        docs.append({
            "href": a["href"],
            "type": tds[0].get_text(strip=True).upper(),
            "desc": tds[-1].get_text(strip=True).lower(),
        })

    # Priority: transcript > any EX-99 exhibit > 8-K main doc > any .htm link
    target_url = None
    for matcher in [
        lambda d: "transcript" in d["desc"],
        lambda d: d["type"].startswith("EX-99"),
        lambda d: d["type"] == "8-K",
        lambda d: d["href"].lower().endswith(".htm"),
    ]:
        for doc in docs:
            if matcher(doc):
                href = doc["href"]
                target_url = href if href.startswith("http") else f"{SEC_BASE_URL}{href}"
                break
        if target_url:
            break

    if not target_url:
        return "", ""

    doc_resp = requests.get(target_url, headers=HEADERS, timeout=30)
    doc_resp.raise_for_status()
    time.sleep(0.15)

    return doc_resp.text, target_url


def parse_and_clean_transcript(html: str) -> dict:
    """
    Parse SEC filing HTML into structured speaker sections.

    Strips boilerplate (scripts, nav, styles), then splits the body text
    by speaker header lines. Each section is tagged with the speaker name
    and a conviction weight (exec / analyst / default).
    Returns dict with keys: cleaned_text (str), sections (list[dict]), ceo_cfo_text (str).
    """
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()

    raw_lines = [
        line.strip()
        for line in soup.get_text(separator="\n").splitlines()
        if line.strip()
    ]

    sections: list[dict] = []
    current_speaker: str | None = None
    current_weight: float = WEIGHT_DEFAULT
    current_lines: list[str] = []

    for line in raw_lines:
        m = _SPEAKER_RE.match(line)
        if m:
            if current_lines:
                sections.append({
                    "speaker": current_speaker,
                    "weight": current_weight,
                    "text": " ".join(current_lines),
                })
            current_speaker = m.group(1).strip()
            current_weight = _classify_speaker_weight(current_speaker)
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections.append({
            "speaker": current_speaker,
            "weight": current_weight,
            "text": " ".join(current_lines),
        })

    # If no speakers were detected treat the whole doc as one unattributed block
    if not any(s["speaker"] for s in sections):
        sections = [{"speaker": None, "weight": WEIGHT_DEFAULT, "text": " ".join(raw_lines)}]

    cleaned_text = "\n\n".join(
        (f"[{s['speaker']}]\n" if s["speaker"] else "") + s["text"]
        for s in sections if s["text"]
    )
    ceo_cfo_text = "\n\n".join(
        s["text"] for s in sections
        if s["weight"] >= WEIGHT_EXEC and s["text"]
    )

    return {"cleaned_text": cleaned_text, "sections": sections, "ceo_cfo_text": ceo_cfo_text}


def calculate_sentiment_score(parsed: dict) -> tuple[float, str]:
    """
    VADER sentiment with speaker-weighted averaging.

    Each section's compound score is weighted by speaker conviction × character count
    so longer executive speeches dominate. Analyst questions are down-weighted.
    Returns (score, direction) where score is -1.0 to 1.0.
    """
    sia = SentimentIntensityAnalyzer()
    sections = parsed.get("sections", [])

    if not sections:
        compound = sia.polarity_scores(parsed.get("cleaned_text", ""))["compound"]
    else:
        weighted_sum = 0.0
        total_weight = 0.0
        for section in sections:
            text = section.get("text", "")
            if len(text) < 20:
                continue
            compound = sia.polarity_scores(text)["compound"]
            effective_weight = section["weight"] * len(text)
            weighted_sum += compound * effective_weight
            total_weight += effective_weight
        compound = (weighted_sum / total_weight) if total_weight > 0 else 0.0

    score = max(-1.0, min(1.0, compound))

    # Earnings calls use formal language; VADER scores tend to be muted — use tight threshold
    if score > 0.05:
        direction = "bullish"
    elif score < -0.05:
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


def store_transcript(
    db: Session,
    ticker: Ticker,
    filing_date: datetime,
    accession_number: str,
    document_url: str,
    cleaned_text: str,
    ceo_cfo_text: str,
) -> EarningsTranscript:
    """
    Persist the cleaned filing text to earnings_transcripts.
    Idempotent: returns existing row if accession_number already stored.
    """
    existing = (
        db.query(EarningsTranscript)
        .filter(EarningsTranscript.accession_number == accession_number)
        .first()
    )
    if existing:
        return existing

    transcript = EarningsTranscript(
        ticker_id=ticker.id,
        filing_date=filing_date,
        accession_number=accession_number,
        document_url=document_url,
        cleaned_text=cleaned_text,
        ceo_cfo_text=ceo_cfo_text,
    )
    db.add(transcript)
    db.commit()
    db.refresh(transcript)
    return transcript


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


def run_earnings_pipeline() -> None:
    """
    Execute the full SEC EDGAR → VADER → signal pipeline for AAPL.

    Steps:
      1. Find the most recent AAPL earnings 8-K (Item 2.02) via EDGAR submissions API
      2. Fetch the filing document HTML (prefers transcript exhibit > EX-99 > primary)
      3. Parse HTML: strip boilerplate, split by speaker, tag exec vs analyst sections
      4. VADER with CEO/CFO sections at 2× weight, analysts at 0.5× weight
      5. Store cleaned transcript in earnings_transcripts; signal in signals table
    """
    print("Starting earnings sentiment pipeline for AAPL...")
    db = SessionLocal()

    try:
        print("Step 1: Fetching latest AAPL earnings 8-K from SEC EDGAR...")
        filing = fetch_latest_earnings_8k(AAPL_CIK)
        if not filing:
            print("  No 8-K filing found")
            return
        print(f"  Filing: {filing['accession_no']} | Date: {filing['filing_date']}")

        print("Step 2: Fetching filing document HTML...")
        html, doc_url = fetch_filing_document(AAPL_CIK, filing["accession_no"])
        if not html:
            print("  Could not retrieve filing document")
            return
        print(f"  Document URL: {doc_url}")

        print("Step 3: Parsing and cleaning transcript...")
        parsed = parse_and_clean_transcript(html)
        n_exec = sum(1 for s in parsed["sections"] if s["weight"] >= WEIGHT_EXEC)
        n_analyst = sum(1 for s in parsed["sections"] if s["weight"] == WEIGHT_ANALYST)
        print(f"  {len(parsed['sections'])} sections | {n_exec} exec | {n_analyst} analyst")

        print("Step 4: Calculating VADER sentiment score...")
        score, direction = calculate_sentiment_score(parsed)
        print(f"  Score: {score:.4f} | Direction: {direction}")

        print("Step 5: Storing transcript and signal...")
        filing_date = datetime.strptime(filing["filing_date"], "%Y-%m-%d")
        ticker_obj = get_or_create_ticker(db, "AAPL")

        transcript = store_transcript(
            db,
            ticker=ticker_obj,
            filing_date=filing_date,
            accession_number=filing["accession_no"],
            document_url=doc_url,
            cleaned_text=parsed["cleaned_text"],
            ceo_cfo_text=parsed["ceo_cfo_text"],
        )
        signal = store_signal(db, ticker_obj, score, direction)
        print(
            f"  Transcript ID={transcript.id} | "
            f"Signal ID={signal.id}, Score={score:.4f}, Direction={direction}"
        )

    except Exception as e:
        print(f"Error in earnings pipeline: {e}")
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    run_earnings_pipeline()
