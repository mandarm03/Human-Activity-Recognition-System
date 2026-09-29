"""
utils.py — Shared utility functions for the HAR pipeline.

Provides normalization, input validation, label helpers, and
plotting utilities used across training and inference.
"""

import json
import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")  # Non-interactive backend for server environments

# ── Logging ─────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

ACTIVITY_LABELS: List[str] = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
]

SIGNAL_COLUMNS: List[str] = [
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
    "total_acc_x",
    "total_acc_y",
    "total_acc_z",
]

NUM_CLASSES: int = 6
TIMESTEPS: int = 128
NUM_CHANNELS: int = 9

# ── Label Helpers ────────────────────────────────────────────────────────────


def get_activity_labels() -> List[str]:
    """Return the ordered list of activity class names."""
    return ACTIVITY_LABELS.copy()


def get_activity_name(class_index: int) -> str:
    """Convert a class index (0-based) to a human-readable activity name."""
    if class_index < 0 or class_index >= NUM_CLASSES:
        raise ValueError(
            f"class_index must be in [0, {NUM_CLASSES - 1}], got {class_index}"
        )
    return ACTIVITY_LABELS[class_index]


def save_label_mapping(save_dir: str) -> None:
    """
    Persist the activity label list to a JSON file for inference use.

    Args:
        save_dir: Directory where ``activity_labels.json`` will be written.
    """
    path = Path(save_dir) / "activity_labels.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(ACTIVITY_LABELS, f, indent=2)
    logger.info("Activity labels saved → %s", path)


def load_label_mapping(save_dir: str) -> List[str]:
    """
    Load the activity label list from a JSON file.

    Args:
        save_dir: Directory containing ``activity_labels.json``.

    Returns:
        Ordered list of activity names.
    """
    path = Path(save_dir) / "activity_labels.json"
    if not path.exists():
        logger.warning("activity_labels.json not found, using defaults.")
        return ACTIVITY_LABELS.copy()
    with open(path, "r") as f:
        labels = json.load(f)
    return labels


# ── Normalization ────────────────────────────────────────────────────────────


def normalize_data(
    X_train: np.ndarray,
    X_test: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray]:
    """
    Apply per-channel Z-score normalization.

    Statistics are computed on ``X_train`` and applied to both splits so there
    is no data leakage from the test set.

    Args:
        X_train: Training data of shape ``(N_train, 128, 9)``.
        X_test:  Optional test data of shape ``(N_test, 128, 9)``.

    Returns:
        Tuple of ``(X_train_norm, X_test_norm, mean, std)``.
        ``X_test_norm`` is ``None`` when ``X_test`` is ``None``.
    """
    # Compute statistics along samples & timesteps, keep channel dim
    mean = X_train.mean(axis=(0, 1), keepdims=True)  # (1, 1, 9)
    std = X_train.std(axis=(0, 1), keepdims=True)  # (1, 1, 9)
    std = np.where(std == 0, 1.0, std)  # avoid division by zero

    X_train_norm = (X_train - mean) / std

    X_test_norm: Optional[np.ndarray] = None
    if X_test is not None:
        X_test_norm = (X_test - mean) / std

    logger.info("Normalization — mean shape %s, std shape %s", mean.shape, std.shape)
    return X_train_norm, X_test_norm, mean.squeeze(), std.squeeze()


def normalize_single(
    X: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """
    Normalize a single sample using pre-computed train statistics.

    Args:
        X:    Input array of shape ``(128, 9)`` or ``(1, 128, 9)``.
        mean: Per-channel mean of shape ``(9,)``.
        std:  Per-channel std  of shape ``(9,)``.

    Returns:
        Normalized array with the same shape as ``X``.
    """
    std_safe = np.where(std == 0, 1.0, std)
    return (X - mean) / std_safe


# ── Input Validation ────────────────────────────────────────────────────────


def validate_input(data: np.ndarray) -> None:
    """
    Validate that an inference input has the expected shape and is finite.

    Args:
        data: Array to validate.

    Raises:
        ValueError: When shape or data quality checks fail.
    """
    if data.ndim == 2:
        rows, cols = data.shape
    elif data.ndim == 3:
        _, rows, cols = data.shape
    else:
        raise ValueError(
            f"Expected 2-D (128, 9) or 3-D (1, 128, 9) array, got shape {data.shape}"
        )

    if rows != TIMESTEPS:
        raise ValueError(
            f"Expected {TIMESTEPS} time steps, got {rows}. "
            "Upload a CSV with exactly 128 rows."
        )
    if cols != NUM_CHANNELS:
        raise ValueError(
            f"Expected {NUM_CHANNELS} sensor channels, got {cols}. "
            f"Columns must be: {SIGNAL_COLUMNS}"
        )
    if not np.isfinite(data).all():
        raise ValueError("Input data contains NaN or Inf values.")


# ── Plotting Helpers ─────────────────────────────────────────────────────────


def plot_training_history(
    history_dict: dict,
    save_path: str,
) -> None:
    """
    Save accuracy and loss curves from a Keras training history.

    Args:
        history_dict: ``model.history.history`` dictionary.
        save_path:    Full file path for the output PNG.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#1a1a2e")

    for ax in axes:
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

    # Accuracy
    axes[0].plot(history_dict["accuracy"], color="#00d4ff", linewidth=2, label="Train")
    axes[0].plot(
        history_dict["val_accuracy"], color="#ff6b6b", linewidth=2, label="Val"
    )
    axes[0].set_title("Model Accuracy", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(facecolor="#16213e", labelcolor="white")
    axes[0].grid(color="#333", linestyle="--", linewidth=0.5)

    # Loss
    axes[1].plot(history_dict["loss"], color="#00d4ff", linewidth=2, label="Train")
    axes[1].plot(history_dict["val_loss"], color="#ff6b6b", linewidth=2, label="Val")
    axes[1].set_title("Model Loss", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend(facecolor="#16213e", labelcolor="white")
    axes[1].grid(color="#333", linestyle="--", linewidth=0.5)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info("Training history plot saved → %s", save_path)


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: List[str],
    save_path: str,
) -> None:
    """
    Save a styled confusion matrix heatmap.

    Args:
        cm:        Confusion matrix array ``(n_classes, n_classes)``.
        labels:    Class name strings.
        save_path: Full file path for the output PNG.
    """
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    tick_marks = np.arange(len(labels))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    short = [lbl.replace("_", "\n") for lbl in labels]
    ax.set_xticklabels(short, rotation=45, ha="right", color="white", fontsize=9)
    ax.set_yticklabels(short, color="white", fontsize=9)

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] < thresh else "black",
                fontsize=9,
            )

    ax.set_title(
        "Confusion Matrix", color="white", fontsize=13, fontweight="bold", pad=12
    )
    ax.set_xlabel("Predicted Label", color="white")
    ax.set_ylabel("True Label", color="white")
    ax.tick_params(colors="white")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info("Confusion matrix saved → %s", save_path)


# ── Misc ─────────────────────────────────────────────────────────────────────


def ensure_dir(path: str) -> None:
    """Create directory (and parents) if it does not already exist."""
    Path(path).mkdir(parents=True, exist_ok=True)
