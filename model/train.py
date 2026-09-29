"""
train.py — End-to-end training pipeline for the HAR CNN + LSTM model.

Run from the project root:
    python model/train.py

Outputs written to saved_models/:
    har_model.keras          — Best model weights
    scaler_mean.npy          — Per-channel mean (for inference normalization)
    scaler_std.npy           — Per-channel std  (for inference normalization)
    activity_labels.json     — Ordered class name list
    model_config.json        — Hyperparameter configuration
    training_history.csv     — Epoch-by-epoch metrics
    training_curves.png      — Accuracy & loss plots
    confusion_matrix.png     — Confusion matrix on test set
    classification_report.txt — Precision / Recall / F1 per class
"""

import json
import logging
import sys
from pathlib import Path

# ── Allow importing from project root ────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import (
    EarlyStopping,  # type: ignore
    ModelCheckpoint,
    ReduceLROnPlateau,
)

from model.cnn_lstm import build_model, get_model_config
from model.dataset import load_dataset
from model.utils import (
    ACTIVITY_LABELS,
    ensure_dir,
    normalize_data,
    plot_confusion_matrix,
    plot_training_history,
    save_label_mapping,
)

# ── Configuration ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_ROOT = PROJECT_ROOT / "data" / "UCI HAR Dataset"
SAVE_DIR = PROJECT_ROOT / "saved_models"

BATCH_SIZE = 64
MAX_EPOCHS = 50
VAL_SPLIT = 0.15
PATIENCE = 7  # EarlyStopping patience
LR_PATIENCE = 4  # ReduceLROnPlateau patience
MIN_LR = 1e-6
RANDOM_SEED = 42


# ── Main Training Pipeline ───────────────────────────────────────────────────


def train() -> None:
    """Execute the complete training pipeline."""
    ensure_dir(str(SAVE_DIR))
    np.random.seed(RANDOM_SEED)

    # ── 1. Load Data ──────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 1 — Loading dataset")
    logger.info("=" * 60)
    X_train, y_train, X_test, y_test = load_dataset(data_root=DATA_ROOT)

    # ── 2. Normalize ──────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 2 — Normalizing data (Z-score per channel)")
    logger.info("=" * 60)
    X_train_norm, X_test_norm, mean, std = normalize_data(X_train, X_test)

    # Persist scaler statistics for inference
    np.save(SAVE_DIR / "scaler_mean.npy", mean)
    np.save(SAVE_DIR / "scaler_std.npy", std)
    logger.info("Scaler stats saved → scaler_mean.npy / scaler_std.npy")

    # ── 3. Build Model ────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 3 — Building CNN + LSTM model")
    logger.info("=" * 60)
    model = build_model()
    model.summary(print_fn=logger.info)

    config = get_model_config()
    config["training"] = {
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "val_split": VAL_SPLIT,
        "random_seed": RANDOM_SEED,
    }
    with open(SAVE_DIR / "model_config.json", "w") as f:
        json.dump(config, f, indent=2)
    logger.info("Model config saved → model_config.json")

    save_label_mapping(str(SAVE_DIR))

    # ── 4. Callbacks ──────────────────────────────────────────────────────
    model_path = str(SAVE_DIR / "har_model.keras")

    callbacks = [
        ModelCheckpoint(
            filepath=model_path,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_accuracy",
            patience=PATIENCE,
            mode="max",
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=LR_PATIENCE,
            min_lr=MIN_LR,
            verbose=1,
        ),
    ]

    # ── 5. Train ──────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 4 — Training (max %d epochs, batch %d)", MAX_EPOCHS, BATCH_SIZE)
    logger.info("=" * 60)
    history = model.fit(
        X_train_norm,
        y_train,
        validation_split=VAL_SPLIT,
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    # ── 6. Save History ───────────────────────────────────────────────────
    hist_df = pd.DataFrame(history.history)
    hist_csv = SAVE_DIR / "training_history.csv"
    hist_df.to_csv(hist_csv, index=False)
    logger.info("Training history saved → %s", hist_csv)

    plot_training_history(
        history_dict=history.history,
        save_path=str(SAVE_DIR / "training_curves.png"),
    )

    # ── 7. Evaluate on Test Set ───────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 5 — Evaluating on test set")
    logger.info("=" * 60)
    test_loss, test_acc = model.evaluate(X_test_norm, y_test, verbose=0)
    logger.info("Test Loss:     %.4f", test_loss)
    logger.info("Test Accuracy: %.4f  (%.2f%%)", test_acc, test_acc * 100)

    # ── 8. Classification Report & Confusion Matrix ───────────────────────
    y_pred_probs = model.predict(X_test_norm, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    report = classification_report(
        y_true,
        y_pred,
        target_names=ACTIVITY_LABELS,
        digits=4,
    )
    logger.info("\n%s", report)

    report_path = SAVE_DIR / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(f"Test Loss:     {test_loss:.4f}\n")
        f.write(f"Test Accuracy: {test_acc:.4f} ({test_acc * 100:.2f}%)\n\n")
        f.write(report)
    logger.info("Classification report saved → %s", report_path)

    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(
        cm=cm,
        labels=ACTIVITY_LABELS,
        save_path=str(SAVE_DIR / "confusion_matrix.png"),
    )

    # ── 9. Final Summary ──────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("  Model     → %s", model_path)
    logger.info("  Test Acc  → %.2f%%", test_acc * 100)
    logger.info("  Saved to  → %s", SAVE_DIR)
    logger.info("=" * 60)


if __name__ == "__main__":
    train()
