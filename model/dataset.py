"""
dataset.py — UCI HAR Dataset loader using raw inertial signals.

Loads the 9 raw inertial signal files per split and stacks them into
tensors of shape (samples, 128, 9). Labels are converted from 1-6 to
0-5 and one-hot encoded.

Usage:
    from model.dataset import load_dataset
    X_train, y_train, X_test, y_test = load_dataset()
"""

import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical  # type: ignore

logger = logging.getLogger(__name__)

# ── Signal file name templates ───────────────────────────────────────────────

SIGNAL_NAMES: List[str] = [
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

# Default dataset root (relative to project root)
_DEFAULT_DATA_ROOT = Path("data") / "UCI HAR Dataset"


# ── Private helpers ──────────────────────────────────────────────────────────


def _load_signal_file(path: Path) -> np.ndarray:
    """
    Read a single space-separated signal file and return a 2-D array.

    Args:
        path: Absolute / relative path to the .txt signal file.

    Returns:
        NumPy array of shape ``(samples, 128)``.

    Raises:
        FileNotFoundError: When the file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Signal file not found: {path}")
    df = pd.read_csv(path, sep=r"\s+", header=None)
    return df.values  # shape: (samples, 128)


def _load_signals(split: str, data_root: Path) -> np.ndarray:
    """
    Load all 9 inertial signal files for one split and stack them.

    Args:
        split:     ``'train'`` or ``'test'``.
        data_root: Root directory of the UCI HAR Dataset.

    Returns:
        NumPy array of shape ``(samples, 128, 9)``.
    """
    signals_dir = data_root / split / "Inertial Signals"
    loaded = []

    for name in SIGNAL_NAMES:
        filename = f"{name}_{split}.txt"
        file_path = signals_dir / filename
        arr = _load_signal_file(file_path)  # (samples, 128)
        loaded.append(arr)
        logger.info("  Loaded %s → shape %s", filename, arr.shape)

    # Stack: list of (N, 128) → (9, N, 128) → (N, 128, 9)
    X = np.array(loaded)  # (9, N, 128)
    X = np.transpose(X, (1, 2, 0))  # (N, 128, 9)
    logger.info("  Stacked %s signals → %s", split, X.shape)
    return X


def _load_labels(split: str, data_root: Path) -> np.ndarray:
    """
    Load activity labels and convert from 1-based to 0-based indexing.

    Args:
        split:     ``'train'`` or ``'test'``.
        data_root: Root directory of the UCI HAR Dataset.

    Returns:
        1-D integer array of shape ``(samples,)`` with values in ``[0, 5]``.
    """
    label_file = data_root / split / f"y_{split}.txt"
    if not label_file.exists():
        raise FileNotFoundError(f"Label file not found: {label_file}")
    labels = pd.read_csv(label_file, header=None).values.flatten()
    labels = labels - 1  # 1-6 → 0-5
    logger.info(
        "  Labels %s → shape %s, unique %s", split, labels.shape, np.unique(labels)
    )
    return labels.astype(np.int32)


# ── Public API ───────────────────────────────────────────────────────────────


def load_dataset(
    data_root: str | Path = _DEFAULT_DATA_ROOT,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the complete UCI HAR Dataset from raw inertial signals.

    Args:
        data_root: Path to the ``UCI HAR Dataset`` directory.
                   Defaults to ``data/UCI HAR Dataset`` (project-relative).

    Returns:
        Tuple ``(X_train, y_train, X_test, y_test)`` where:

        * ``X_train`` — shape ``(7352, 128, 9)``, dtype float64
        * ``y_train`` — shape ``(7352, 6)``, one-hot encoded
        * ``X_test``  — shape ``(2947, 128, 9)``, dtype float64
        * ``y_test``  — shape ``(2947, 6)``, one-hot encoded
    """
    data_root = Path(data_root)
    logger.info("Loading UCI HAR Dataset from: %s", data_root.resolve())

    logger.info("Loading train signals …")
    X_train = _load_signals("train", data_root)
    y_train_raw = _load_labels("train", data_root)

    logger.info("Loading test signals …")
    X_test = _load_signals("test", data_root)
    y_test_raw = _load_labels("test", data_root)

    # One-hot encoding
    num_classes = 6
    y_train = to_categorical(y_train_raw, num_classes=num_classes)
    y_test = to_categorical(y_test_raw, num_classes=num_classes)

    logger.info(
        "Dataset loaded — X_train %s | y_train %s | X_test %s | y_test %s",
        X_train.shape,
        y_train.shape,
        X_test.shape,
        y_test.shape,
    )
    return X_train, y_train, X_test, y_test
