"""
Fine-tunes ProsusAI/finbert on AAPL earnings transcripts labelled with actual
5-day price-outcome directions from signal_labels.

With ~20 labelled examples the entire transformer backbone is frozen — only the
3-class classification head is trained.  This calibrates FinBERT's existing
financial-language embeddings toward observed market outcomes (bullish / bearish /
neutral) rather than generic sentiment positivity.

Score convention (consistent with VADER pipelines):
    score     = p_bullish − p_bearish   →  [−1.0, +1.0]
    direction = argmax{bullish, bearish, neutral}

Saved artefacts (backend/models/saved/finbert/):
    config.json, pytorch_model.bin (or safetensors), tokenizer files,
    training_config.json  — all loadable by FastAPI via load_finbert().
"""

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sqlalchemy.orm import Session
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from backend.db.database import SessionLocal
from backend.db.models import EarningsTranscript, Signal, SignalLabel


# ── constants ─────────────────────────────────────────────────────────────────

MODEL_NAME = "ProsusAI/finbert"
SAVE_DIR = Path(__file__).parent / "saved" / "finbert"

MAX_LENGTH = 512          # FinBERT hard limit; long transcripts are front-truncated
                          # (prepared remarks appear first, so the first 512 tokens
                          #  capture CEO / CFO commentary before the Q&A section)

LABEL_MAP: dict[str, int] = {"bullish": 0, "bearish": 1, "neutral": 2}
ID_TO_LABEL: dict[int, str] = {v: k for k, v in LABEL_MAP.items()}
LABEL_NAMES = list(LABEL_MAP)          # ["bullish", "bearish", "neutral"]

# Hyper-parameters tuned for head-only fine-tuning on a small dataset
TRAIN_EPOCHS = 10
BATCH_SIZE = 4
LEARNING_RATE = 2e-4      # higher than typical full fine-tune; backbone is frozen
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
VAL_SPLIT = 0.2


# ── data loading ──────────────────────────────────────────────────────────────

def load_training_data(db: Session) -> list[dict]:
    """
    Join earnings_transcripts × signal_labels × signals and return every row
    that has a valid 5-day price label (the primary training target).

    ceo_cfo_text is preferred over cleaned_text: executive commentary carries
    more signal and fits more comfortably within the 512-token limit.
    """
    rows = (
        db.query(EarningsTranscript, SignalLabel, Signal)
        .join(SignalLabel, SignalLabel.transcript_id == EarningsTranscript.id)
        .join(Signal, Signal.id == SignalLabel.signal_id)
        .filter(SignalLabel.label_5d.isnot(None))
        .order_by(SignalLabel.filing_date)
        .all()
    )

    records: list[dict] = []
    for transcript, label, signal in rows:
        text = (transcript.ceo_cfo_text or "").strip() or (transcript.cleaned_text or "").strip()
        if not text:
            continue
        records.append({
            "text": text,
            "label": label.label_5d,          # primary training target
            "label_30d": label.label_30d,
            "label_90d": label.label_90d,
            "vader_score": signal.score,
            "vader_direction": signal.direction,
            "filing_date": str(label.filing_date.date()),
            "return_5d": label.return_5d,
        })

    return records


# ── dataset ───────────────────────────────────────────────────────────────────

class EarningsDataset(torch.utils.data.Dataset):
    """Thin wrapper that presents tokenized encodings + integer labels to Trainer."""

    def __init__(self, encodings: dict[str, list], labels: list[int]) -> None:
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ── model helpers ─────────────────────────────────────────────────────────────

def _freeze_backbone(model) -> None:
    """
    Freeze every parameter except the pooler and classifier head.

    With ~16 training examples unfreezing even one transformer block risks
    catastrophic forgetting and overfitting.  The frozen backbone retains
    FinBERT's financial-language representations; only the output head is
    re-oriented toward actual price-outcome labels.
    """
    for name, param in model.named_parameters():
        trainable = name.startswith("classifier") or name.startswith("pooler")
        param.requires_grad = trainable


def _compute_metrics(eval_pred) -> dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    accuracy = float((preds == labels).sum()) / len(labels)
    return {"accuracy": accuracy}


# ── training ──────────────────────────────────────────────────────────────────

def train_finbert(db: Session) -> None:
    """
    Fine-tune the FinBERT classification head on labelled AAPL earnings data.

    Steps:
      1. Pull labelled rows (earnings_transcripts × signal_labels × signals)
      2. Stratified 80/20 train/val split
      3. Tokenize with truncation to MAX_LENGTH
      4. Load ProsusAI/finbert; freeze backbone; wire bullish/bearish/neutral head
      5. HuggingFace Trainer: AdamW + linear warmup, eval each epoch, best-model restore
      6. Print per-class report and VADER baseline side-by-side on the val set
      7. Persist model, tokenizer, and training metadata to SAVE_DIR
    """
    print("Step 1: Loading labelled training data...")
    records = load_training_data(db)
    n = len(records)
    print(f"  {n} labelled examples")
    if n < 4:
        raise ValueError(
            f"Only {n} labelled examples — need >= 4.  "
            "Run the earnings backfill pipeline first to populate signal_labels."
        )

    texts = [r["text"] for r in records]
    int_labels = [LABEL_MAP[r["label"]] for r in records]
    dist = {ID_TO_LABEL[i]: int_labels.count(i) for i in range(3)}
    print(f"  Label distribution: {dist}")

    # ── split ────────────────────────────────────────────────────────────────
    print("Step 2: Train/val split...")
    try:
        train_idx, val_idx = train_test_split(
            range(n), test_size=VAL_SPLIT, random_state=42, stratify=int_labels
        )
    except ValueError:
        # Stratification impossible when a class has only 1 sample
        train_idx, val_idx = train_test_split(range(n), test_size=VAL_SPLIT, random_state=42)

    train_texts = [texts[i] for i in train_idx]
    train_labels = [int_labels[i] for i in train_idx]
    val_texts = [texts[i] for i in val_idx]
    val_labels = [int_labels[i] for i in val_idx]
    val_records = [records[i] for i in val_idx]
    print(f"  train={len(train_texts)}  val={len(val_texts)}")

    # ── model ─────────────────────────────────────────────────────────────────
    print("Step 3: Loading FinBERT...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label=ID_TO_LABEL,
        label2id=LABEL_MAP,
        ignore_mismatched_sizes=True,   # replaces ProsusAI head labels with ours
    )
    _freeze_backbone(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} params ({100 * trainable / total:.2f}%)")

    # ── tokenize ──────────────────────────────────────────────────────────────
    print("Step 4: Tokenizing...")
    def _enc(batch: list[str]) -> dict[str, list]:
        out = tokenizer(batch, truncation=True, padding=True, max_length=MAX_LENGTH)
        return {k: list(v) for k, v in out.items()}

    train_dataset = EarningsDataset(_enc(train_texts), train_labels)
    val_dataset = EarningsDataset(_enc(val_texts), val_labels)

    # ── train ─────────────────────────────────────────────────────────────────
    print(f"Step 5: Fine-tuning classification head ({TRAIN_EPOCHS} epochs)...")
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(SAVE_DIR / "checkpoints"),
        num_train_epochs=TRAIN_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_steps=1,
        report_to="none",
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=_compute_metrics,
    )

    trainer.train()

    # ── evaluate ──────────────────────────────────────────────────────────────
    print("Step 6: Evaluating...")
    preds_out = trainer.predict(val_dataset)
    val_preds = np.argmax(preds_out.predictions, axis=-1)
    true_names = [ID_TO_LABEL[l] for l in val_labels]
    finbert_pred_names = [ID_TO_LABEL[p] for p in val_preds]
    vader_pred_names = [r["vader_direction"] for r in val_records]

    print("\n  FinBERT (fine-tuned) on val set:")
    print(classification_report(true_names, finbert_pred_names, labels=LABEL_NAMES, zero_division=0))

    print("  VADER baseline on same val set:")
    print(classification_report(true_names, vader_pred_names, labels=LABEL_NAMES, zero_division=0))

    print("  Filing-level detail (date | actual | FinBERT | VADER | 5d return):")
    for rec, fb, vd in zip(val_records, finbert_pred_names, vader_pred_names):
        ret = f"{rec['return_5d']:+.2%}" if rec.get("return_5d") is not None else "N/A"
        print(f"    {rec['filing_date']}  actual={rec['label']:<8} finbert={fb:<8} vader={vd:<8} {ret}")

    # ── save ──────────────────────────────────────────────────────────────────
    print(f"\nStep 7: Saving to {SAVE_DIR}...")
    model.save_pretrained(str(SAVE_DIR))
    tokenizer.save_pretrained(str(SAVE_DIR))
    (SAVE_DIR / "training_config.json").write_text(json.dumps({
        "base_model": MODEL_NAME,
        "training_horizon": "label_5d",
        "label_map": LABEL_MAP,
        "id_to_label": {str(k): v for k, v in ID_TO_LABEL.items()},
        "max_length": MAX_LENGTH,
        "train_samples": len(train_texts),
        "val_samples": len(val_texts),
        "label_distribution": dist,
        "epochs": TRAIN_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "frozen_backbone": True,
    }, indent=2))
    print("  Done.")


# ── inference ─────────────────────────────────────────────────────────────────

def load_finbert() -> tuple:
    """
    Load the fine-tuned model and tokenizer from SAVE_DIR.
    Call once at FastAPI startup and pass the returned (model, tokenizer) to predict().
    """
    if not SAVE_DIR.exists():
        raise FileNotFoundError(
            f"No fine-tuned model at {SAVE_DIR}. "
            "Run run_finbert_pipeline() to train and save first."
        )
    tokenizer = AutoTokenizer.from_pretrained(str(SAVE_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(SAVE_DIR))
    model.eval()
    return model, tokenizer


def predict(text: str, model=None, tokenizer=None) -> dict:
    """
    Score a single earnings text with the fine-tuned FinBERT model.

    Returns
    -------
    score       float   p_bullish − p_bearish  →  [−1.0, +1.0], consistent with VADER scale
    direction   str     "bullish" | "bearish" | "neutral"
    confidence  float   max softmax probability (how certain the model is)
    label_probs dict    per-class softmax probabilities

    Pass pre-loaded model/tokenizer for repeated calls (e.g. FastAPI endpoint);
    omit both to auto-load from disk (convenient for one-off scripts).
    """
    if model is None or tokenizer is None:
        model, tokenizer = load_finbert()

    inputs = tokenizer(
        text, return_tensors="pt",
        truncation=True, padding=True, max_length=MAX_LENGTH,
    )

    with torch.no_grad():
        logits = model(**inputs).logits

    probs: list[float] = torch.softmax(logits, dim=-1).squeeze().tolist()
    p_bullish, p_bearish, p_neutral = probs

    return {
        "score": round(p_bullish - p_bearish, 6),
        "direction": ID_TO_LABEL[int(np.argmax(probs))],
        "confidence": round(max(probs), 4),
        "label_probs": {
            "bullish": round(p_bullish, 4),
            "bearish": round(p_bearish, 4),
            "neutral": round(p_neutral, 4),
        },
    }


# ── orchestrator ──────────────────────────────────────────────────────────────

def run_finbert_pipeline() -> None:
    """
    End-to-end FinBERT fine-tuning pipeline.

    Steps:
      1. Load labelled data from DB (requires earnings_backfill to have run)
      2. Fine-tune classification head on label_5d targets
      3. Evaluate and compare against VADER baseline
      4. Save model + tokenizer + config to backend/models/saved/finbert/

    The saved model is loaded by FastAPI on startup via load_finbert() so that
    the /signals endpoint can return FinBERT scores alongside VADER scores.
    """
    print("Starting FinBERT fine-tuning pipeline...")
    db = SessionLocal()
    try:
        train_finbert(db)
    except Exception as exc:
        print(f"Fatal error: {exc}")
        raise
    finally:
        db.close()

    print("\nSmoke-testing saved model...")
    result = predict(
        "Revenue grew 8% year over year with strong iPhone demand and record Services revenue. "
        "We are guiding for continued growth in the next quarter."
    )
    print(
        f"  score={result['score']:+.4f}  "
        f"direction={result['direction']}  "
        f"confidence={result['confidence']:.2%}"
    )
    print(f"  label_probs={result['label_probs']}")
    print("\nFinBERT fine-tuning complete.")


if __name__ == "__main__":
    run_finbert_pipeline()
