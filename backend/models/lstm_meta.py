"""
Trains an LSTM "meta model" that combines every source's signals over time —
google_trends, weather, apewisdom, earnings — into a single 5-day directional
prediction per ticker, the final layer described in the platform architecture
(FinBERT / VADER / XGBoost feed individual sources; this model learns how they
interact across time).

Each Signal row (one source, one point in time) is encoded as a small numeric
feature vector (raw score + one-hot direction + one-hot source). For every
signal that has a labelled 5-day outcome, the model is trained on the sequence
of up to MAX_SEQ_LEN prior signals for that ticker (across all sources,
chronologically ordered) ending at that signal — i.e. "given everything we
knew up to this point, what happened 5 days later?"

Score convention (consistent with VADER / FinBERT / XGBoost pipelines):
    direction        = argmax{bullish, bearish, neutral}
    win_probability  = softmax probability of the predicted class
    conviction       = p_bullish − p_bearish  →  [−1.0, +1.0]

Saved artefacts (backend/models/saved/lstm/):
    model.pt     — state_dict for the trained SignalLSTM
    config.json  — architecture dims, label/source maps, training metadata
"""

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sqlalchemy.orm import Session

from backend.db.database import SessionLocal
from backend.db.models import Signal, SignalLabel


# ── constants ─────────────────────────────────────────────────────────────────

SAVE_DIR = Path(__file__).parent / "saved" / "lstm"

SOURCES = ["google_trends", "weather", "apewisdom", "earnings"]
SOURCE_INDEX = {s: i for i, s in enumerate(SOURCES)}

LABEL_MAP: dict[str, int] = {"bullish": 0, "bearish": 1, "neutral": 2}
ID_TO_LABEL: dict[int, str] = {v: k for k, v in LABEL_MAP.items()}
LABEL_NAMES = list(LABEL_MAP)          # ["bullish", "bearish", "neutral"]

FEATURE_DIM = 1 + len(LABEL_NAMES) + len(SOURCES)   # score + direction one-hot + source one-hot
MAX_SEQ_LEN = 10          # longest window of prior cross-source signals fed to the LSTM
MIN_SAMPLES = 4           # below this, a train/val split is meaningless

# Small, shallow network — sequences are short (<= MAX_SEQ_LEN) and the labelled
# dataset is tiny (a handful to a few dozen examples), so a large model would just overfit.
HIDDEN_SIZE = 16
NUM_LAYERS = 1
DROPOUT = 0.2
EPOCHS = 30
LEARNING_RATE = 0.01
VAL_SPLIT = 0.2

# Fallback threshold used only to derive a direction for predict()'s single-snapshot
# input, where no direction is supplied — pipelines each use slightly different
# thresholds (0.05–0.15); this is a simplification since direction is a secondary
# feature alongside the raw score, which is passed through unmodified either way.
INFERENCE_DIRECTION_THRESHOLD = 0.15


# ── feature encoding ──────────────────────────────────────────────────────────

def _encode_signal(source: str, score: float, direction: str) -> list[float]:
    """One timestep's feature vector: [score, *direction_one_hot, *source_one_hot]."""
    direction_onehot = [1.0 if direction == d else 0.0 for d in LABEL_NAMES]
    source_onehot = [1.0 if source == s else 0.0 for s in SOURCES]
    return [float(score)] + direction_onehot + source_onehot


def _direction_from_score(score: float) -> str:
    if score > INFERENCE_DIRECTION_THRESHOLD:
        return "bullish"
    if score < -INFERENCE_DIRECTION_THRESHOLD:
        return "bearish"
    return "neutral"


# ── data loading ──────────────────────────────────────────────────────────────

def load_training_data(db: Session) -> list[dict]:
    """
    Build one training example per labelled signal: the sequence of up to
    MAX_SEQ_LEN prior signals (this ticker, all SOURCES, chronologically
    ordered) ending at that signal, paired with its label_5d target.
    """
    signals = (
        db.query(Signal)
        .filter(Signal.source.in_(SOURCES))
        .order_by(Signal.ticker_id, Signal.timestamp)
        .all()
    )

    label_by_signal_id = {
        row.signal_id: row.label_5d
        for row in db.query(SignalLabel).filter(SignalLabel.label_5d.isnot(None)).all()
    }

    by_ticker: dict[int, list[Signal]] = defaultdict(list)
    for sig in signals:
        by_ticker[sig.ticker_id].append(sig)

    records: list[dict] = []
    for ticker_id, ticker_signals in by_ticker.items():
        for i, anchor in enumerate(ticker_signals):
            label = label_by_signal_id.get(anchor.id)
            if label is None:
                continue

            window = ticker_signals[max(0, i - MAX_SEQ_LEN + 1): i + 1]
            sequence = [_encode_signal(s.source, s.score, s.direction) for s in window]

            records.append({
                "ticker_id": ticker_id,
                "sequence": sequence,
                "label": LABEL_MAP[label],
                "anchor_timestamp": str(anchor.timestamp),
            })

    return records


# ── model ─────────────────────────────────────────────────────────────────────

class SignalLSTM(nn.Module):
    """Many-to-one LSTM: a variable-length sequence of signal features → 3-class logits."""

    def __init__(
        self,
        input_size: int = FEATURE_DIM,
        hidden_size: int = HIDDEN_SIZE,
        num_layers: int = NUM_LAYERS,
        num_classes: int = 3,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (1, seq_len, input_size) → logits: (1, num_classes)."""
        _, (h_n, _) = self.lstm(x)
        last_hidden = self.dropout(h_n[-1])
        return self.classifier(last_hidden)


# ── training ──────────────────────────────────────────────────────────────────

def train_lstm_meta(db: Session) -> dict:
    """
    Train the LSTM meta model on labelled cross-source signal sequences.

    Steps:
      1. Build (sequence, label_5d) pairs from signals × signal_labels
      2. Stratified 80/20 train/val split
      3. Manual training loop (batch size 1 — sequences are variable-length
         and the dataset is small, so no padding/packing is needed)
      4. Evaluate: classification report on the val set
      5. Persist state_dict + config to SAVE_DIR
    """
    print("Step 1: Loading labelled cross-source signal sequences...")
    records = load_training_data(db)
    n = len(records)
    print(f"  {n} labelled sequences")
    if n < MIN_SAMPLES:
        raise ValueError(
            f"Only {n} labelled sequences — need >= {MIN_SAMPLES}. "
            "Run the earnings/google_trends/weather backfill pipelines first."
        )

    labels = [r["label"] for r in records]
    dist = {ID_TO_LABEL[i]: labels.count(i) for i in range(3)}
    print(f"  Label distribution: {dist}")

    # ── split ────────────────────────────────────────────────────────────────
    print("Step 2: Train/val split...")
    try:
        train_idx, val_idx = train_test_split(
            range(n), test_size=VAL_SPLIT, random_state=42, stratify=labels
        )
    except ValueError:
        train_idx, val_idx = train_test_split(range(n), test_size=VAL_SPLIT, random_state=42)

    train_records = [records[i] for i in train_idx]
    val_records = [records[i] for i in val_idx]
    print(f"  train={len(train_records)}  val={len(val_records)}")

    # ── train ─────────────────────────────────────────────────────────────────
    print(f"Step 3: Training SignalLSTM ({EPOCHS} epochs)...")
    model = SignalLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    rng = np.random.default_rng(42)
    model.train()
    for epoch in range(1, EPOCHS + 1):
        order = rng.permutation(len(train_records))
        total_loss = 0.0
        for idx in order:
            rec = train_records[idx]
            x = torch.tensor([rec["sequence"]], dtype=torch.float32)   # (1, seq_len, FEATURE_DIM)
            y = torch.tensor([rec["label"]], dtype=torch.long)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch == 1 or epoch % 5 == 0 or epoch == EPOCHS:
            print(f"  epoch {epoch:02d}/{EPOCHS}  avg_loss={total_loss / max(len(train_records), 1):.4f}")

    # ── evaluate ──────────────────────────────────────────────────────────────
    print("Step 4: Evaluating...")
    model.eval()
    val_preds, val_true = [], []
    with torch.no_grad():
        for rec in val_records:
            x = torch.tensor([rec["sequence"]], dtype=torch.float32)
            logits = model(x)
            pred = int(torch.argmax(logits, dim=-1).item())
            val_preds.append(pred)
            val_true.append(rec["label"])

    if val_true:
        true_names = [ID_TO_LABEL[i] for i in val_true]
        pred_names = [ID_TO_LABEL[i] for i in val_preds]
        print("\n  SignalLSTM on val set:")
        print(classification_report(true_names, pred_names, labels=LABEL_NAMES, zero_division=0))

    # ── save ──────────────────────────────────────────────────────────────────
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Step 5: Saving to {SAVE_DIR}...")

    torch.save(model.state_dict(), SAVE_DIR / "model.pt")
    config = {
        "input_size": FEATURE_DIM,
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "num_classes": 3,
        "dropout": DROPOUT,
        "sources": SOURCES,
        "label_map": LABEL_MAP,
        "id_to_label": {str(k): v for k, v in ID_TO_LABEL.items()},
        "max_seq_len": MAX_SEQ_LEN,
        "training_horizon": "label_5d",
        "train_samples": len(train_records),
        "val_samples": len(val_records),
        "label_distribution": dist,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    (SAVE_DIR / "config.json").write_text(json.dumps(config, indent=2))
    print("  Done.")

    return config


# ── inference ─────────────────────────────────────────────────────────────────

def load_lstm_meta() -> tuple:
    """
    Load the trained model + config from SAVE_DIR.
    Call once at FastAPI startup and pass the returned (model, config) to predict().
    """
    config_path = SAVE_DIR / "config.json"
    model_path = SAVE_DIR / "model.pt"
    if not config_path.exists() or not model_path.exists():
        raise FileNotFoundError(
            f"No trained LSTM meta model at {SAVE_DIR}. "
            "Run run_lstm_pipeline() to train and save first."
        )

    config = json.loads(config_path.read_text())
    model = SignalLSTM(
        input_size=config["input_size"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        num_classes=config["num_classes"],
        dropout=config["dropout"],
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model, config


def predict(scores: dict[str, float], model=None, config: dict | None = None) -> dict:
    """
    Predict a 5-day directional outcome from a snapshot of current signal scores.

    Parameters
    ----------
    scores  dict[str, float]   e.g. {"google_trends": 0.4, "weather": -0.2, "earnings": 0.6}
                                Keys must be a subset of SOURCES; direction per source is
                                derived from its score (see INFERENCE_DIRECTION_THRESHOLD).

    Returns
    -------
    direction        str    "bullish" | "bearish" | "neutral"
    win_probability  float  softmax probability of the predicted class
    conviction       float  p_bullish − p_bearish  →  [−1.0, +1.0]
    label_probs      dict   per-class probabilities

    Pass a pre-loaded (model, config) for repeated calls (e.g. FastAPI endpoint);
    omit both to auto-load from disk (convenient for one-off scripts).
    """
    if model is None or config is None:
        model, config = load_lstm_meta()

    ordered_sources = [s for s in SOURCES if s in scores]
    if not ordered_sources:
        raise ValueError(f"scores must contain at least one of: {SOURCES}")

    sequence = [
        _encode_signal(source, scores[source], _direction_from_score(scores[source]))
        for source in ordered_sources
    ]
    x = torch.tensor([sequence], dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=-1).squeeze(0).tolist()

    p_bullish, p_bearish, p_neutral = probs

    return {
        "direction": ID_TO_LABEL[int(np.argmax(probs))],
        "win_probability": round(max(probs), 4),
        "conviction": round(p_bullish - p_bearish, 6),
        "label_probs": {
            "bullish": round(p_bullish, 4),
            "bearish": round(p_bearish, 4),
            "neutral": round(p_neutral, 4),
        },
    }


# ── orchestrator ──────────────────────────────────────────────────────────────

def run_lstm_pipeline() -> None:
    """
    End-to-end LSTM meta-model training pipeline.

    Steps:
      1. Load labelled cross-source signal sequences from DB (requires the
         earnings/google_trends/weather backfill pipelines to have run)
      2. Train the LSTM on label_5d targets
      3. Evaluate on a held-out validation split
      4. Save model + config to backend/models/saved/lstm/

    The saved model is loaded by FastAPI on startup via load_lstm_meta() so the
    /signals endpoint can return a combined meta-prediction alongside each
    individual source's score.
    """
    print("Starting LSTM meta-model training pipeline...")
    db = SessionLocal()
    try:
        train_lstm_meta(db)
    except Exception as exc:
        print(f"Fatal error: {exc}")
        raise
    finally:
        db.close()

    print("\nSmoke-testing saved model...")
    result = predict({
        "google_trends": 0.4,
        "weather": -0.1,
        "apewisdom": 0.2,
        "earnings": 0.6,
    })
    print(
        f"  direction={result['direction']}  "
        f"win_probability={result['win_probability']:.2%}  "
        f"conviction={result['conviction']:+.4f}"
    )
    print(f"  label_probs={result['label_probs']}")
    print("\nLSTM meta-model training complete.")


if __name__ == "__main__":
    run_lstm_pipeline()
