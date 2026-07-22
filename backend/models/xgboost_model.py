"""
Trains a separate XGBoost classifier per numerical signal source — google_trends,
weather, options_flow — to refine each source's raw heuristic score against actual
5-day price-outcome labels from signal_labels.

Each classifier takes a single feature (the source's raw Signal.score) and predicts
the same 3-class target used across the platform (bullish / bearish / neutral),
calibrating the raw heuristic score to what the market actually did next.

Score convention (consistent with VADER / FinBERT pipelines):
    score     = p_bullish − p_bearish   →  [−1.0, +1.0]
    direction = argmax{bullish, bearish, neutral}

Saved artefacts (backend/models/saved/xgboost/):
    {source}.joblib  — one self-contained file per source (model + label maps +
    training metadata), loadable via load_xgboost_model(source).
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sqlalchemy.orm import Session
from xgboost import XGBClassifier

from backend.db.database import SessionLocal
from backend.db.models import Signal, SignalLabel


# ── constants ─────────────────────────────────────────────────────────────────

SOURCES = ["google_trends", "weather", "options_flow"]
SAVE_DIR = Path(__file__).parent / "saved" / "xgboost"

LABEL_MAP: dict[str, int] = {"bullish": 0, "bearish": 1, "neutral": 2}
ID_TO_LABEL: dict[int, str] = {v: k for k, v in LABEL_MAP.items()}
LABEL_NAMES = list(LABEL_MAP)          # ["bullish", "bearish", "neutral"]

MIN_SAMPLES = 4           # below this, stratified split / training is meaningless
VAL_SPLIT = 0.2

# Small, shallow trees — each model has exactly one input feature (the raw score)
# and typically very few labelled rows, so a large ensemble would just overfit.
N_ESTIMATORS = 50
MAX_DEPTH = 3
LEARNING_RATE = 0.1


# ── data loading ──────────────────────────────────────────────────────────────

def load_training_data(db: Session, source: str) -> list[dict]:
    """
    Join signals × signal_labels for the given source and return every row that
    has a valid 5-day price label (the training target).
    """
    rows = (
        db.query(Signal, SignalLabel)
        .join(SignalLabel, SignalLabel.signal_id == Signal.id)
        .filter(Signal.source == source)
        .filter(SignalLabel.label_5d.isnot(None))
        .order_by(SignalLabel.filing_date)
        .all()
    )

    records: list[dict] = []
    for signal, label in rows:
        records.append({
            "raw_score": signal.score,
            "raw_direction": signal.direction,
            "label": label.label_5d,          # training target
            "filing_date": str(label.filing_date.date()),
            "return_5d": label.return_5d,
        })

    return records


# ── training ──────────────────────────────────────────────────────────────────

def train_xgboost_for_source(db: Session, source: str) -> dict:
    """
    Train and save an XGBoost classifier for a single signal source.

    Steps:
      1. Pull labelled rows (signals × signal_labels) for this source
      2. Stratified 80/20 train/val split
      3. Fit XGBClassifier on the raw score as the sole feature
      4. Print classification report vs. the raw heuristic baseline
      5. Persist model + label maps + metadata to SAVE_DIR/{source}.joblib
    """
    print(f"Step 1: Loading labelled training data for '{source}'...")
    records = load_training_data(db, source)
    n = len(records)
    print(f"  {n} labelled examples")
    if n < MIN_SAMPLES:
        raise ValueError(
            f"Only {n} labelled examples for source='{source}' — need >= {MIN_SAMPLES}. "
            "Backfill signal_labels for this source first."
        )

    X = np.array([[r["raw_score"]] for r in records], dtype=float)
    y = np.array([LABEL_MAP[r["label"]] for r in records], dtype=int)
    dist = {ID_TO_LABEL[i]: int((y == i).sum()) for i in range(3)}
    print(f"  Label distribution: {dist}")

    # ── split ────────────────────────────────────────────────────────────────
    print("Step 2: Train/val split...")
    try:
        train_idx, val_idx = train_test_split(
            range(n), test_size=VAL_SPLIT, random_state=42, stratify=y
        )
    except ValueError:
        # Stratification impossible when a class has only 1 sample
        train_idx, val_idx = train_test_split(range(n), test_size=VAL_SPLIT, random_state=42)

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    val_records = [records[i] for i in val_idx]
    print(f"  train={len(X_train)}  val={len(X_val)}")

    # ── train ─────────────────────────────────────────────────────────────────
    print(f"Step 3: Fitting XGBoost classifier for '{source}'...")
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        eval_metric="mlogloss",
        random_state=42,
    )
    model.fit(X_train, y_train)

    # ── evaluate ──────────────────────────────────────────────────────────────
    print("Step 4: Evaluating...")
    val_preds = model.predict(X_val)
    true_names = [ID_TO_LABEL[i] for i in y_val]
    pred_names = [ID_TO_LABEL[i] for i in val_preds]
    raw_pred_names = [r["raw_direction"] for r in val_records]

    print(f"\n  XGBoost ({source}) on val set:")
    print(classification_report(true_names, pred_names, labels=LABEL_NAMES, zero_division=0))

    print(f"  Raw heuristic baseline ({source}) on same val set:")
    print(classification_report(true_names, raw_pred_names, labels=LABEL_NAMES, zero_division=0))

    # ── save ──────────────────────────────────────────────────────────────────
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    save_path = SAVE_DIR / f"{source}.joblib"
    print(f"Step 5: Saving to {save_path}...")

    artefact = {
        "model": model,
        "label_map": LABEL_MAP,
        "id_to_label": ID_TO_LABEL,
        "source": source,
        "training_horizon": "label_5d",
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "label_distribution": dist,
        "n_estimators": N_ESTIMATORS,
        "max_depth": MAX_DEPTH,
        "learning_rate": LEARNING_RATE,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    joblib.dump(artefact, save_path)
    print("  Done.")

    return artefact


# ── inference ─────────────────────────────────────────────────────────────────

def load_xgboost_model(source: str) -> dict:
    """
    Load the saved artefact (model + label maps + metadata) for a signal source.
    Call once at FastAPI startup and pass the returned dict to predict().
    """
    save_path = SAVE_DIR / f"{source}.joblib"
    if not save_path.exists():
        raise FileNotFoundError(
            f"No trained model at {save_path}. "
            f"Run run_xgboost_pipeline() to train and save '{source}' first."
        )
    return joblib.load(save_path)


def _predict(source: str, score: float, artefact: dict | None = None) -> dict:
    """
    Score a single raw signal value with the trained XGBoost classifier for `source`.

    Returns
    -------
    score       float   p_bullish − p_bearish  →  [−1.0, +1.0], consistent with VADER/FinBERT scale
    direction   str     "bullish" | "bearish" | "neutral"
    confidence  float   max predicted-class probability
    label_probs dict    per-class probabilities
    """
    if artefact is None:
        artefact = load_xgboost_model(source)

    model = artefact["model"]
    id_to_label = artefact["id_to_label"]

    probs = model.predict_proba(np.array([[score]], dtype=float))[0].tolist()
    p_bullish, p_bearish, p_neutral = probs

    return {
        "score": round(p_bullish - p_bearish, 6),
        "direction": id_to_label[int(np.argmax(probs))],
        "confidence": round(max(probs), 4),
        "label_probs": {
            "bullish": round(p_bullish, 4),
            "bearish": round(p_bearish, 4),
            "neutral": round(p_neutral, 4),
        },
    }


def predict_google_trends(score: float, model: dict | None = None) -> dict:
    """Refine a raw google_trends signal score. Pass a pre-loaded artefact for repeated calls."""
    return _predict("google_trends", score, model)


def predict_weather(score: float, model: dict | None = None) -> dict:
    """Refine a raw weather signal score. Pass a pre-loaded artefact for repeated calls."""
    return _predict("weather", score, model)


def predict_options_flow(score: float, model: dict | None = None) -> dict:
    """Refine a raw options_flow signal score. Pass a pre-loaded artefact for repeated calls."""
    return _predict("options_flow", score, model)


# ── orchestrator ──────────────────────────────────────────────────────────────

def run_xgboost_pipeline() -> None:
    """
    End-to-end XGBoost training pipeline for every numerical signal source.

    Steps:
      1. For each of google_trends / weather / options_flow:
         a. Load labelled data from DB (requires that source's signal_labels to be backfilled)
         b. Train an XGBoost classifier on label_5d targets
         c. Evaluate against the raw heuristic score as baseline
         d. Save model + label maps + metadata to backend/models/saved/xgboost/{source}.joblib
      2. Sources with insufficient labelled data are skipped, not fatal.

    Saved models are loaded by FastAPI on startup via load_xgboost_model(source) so that
    the /signals endpoint can return refined XGBoost scores alongside the raw heuristic scores.
    """
    print("Starting XGBoost training pipeline...")
    db = SessionLocal()
    trained: list[str] = []
    skipped: list[str] = []

    try:
        for source in SOURCES:
            print(f"\n=== {source} ===")
            try:
                train_xgboost_for_source(db, source)
                trained.append(source)
            except ValueError as exc:
                print(f"  Skipping '{source}': {exc}")
                skipped.append(source)
    finally:
        db.close()

    print(f"\nTrained: {trained}")
    print(f"Skipped (insufficient data): {skipped}")

    if trained:
        print("\nSmoke-testing saved models...")
        for source in trained:
            result = _predict(source, score=0.5)
            print(
                f"  {source}: score={result['score']:+.4f}  "
                f"direction={result['direction']}  "
                f"confidence={result['confidence']:.2%}"
            )

    print("\nXGBoost training pipeline complete.")


if __name__ == "__main__":
    run_xgboost_pipeline()
