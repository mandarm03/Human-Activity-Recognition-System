"""
predict.py — Inference module for the HAR CNN + LSTM model.

Loads the saved model, scaler, and label mapping once at import time
(or on first call) and provides a clean predict_activity() API used by
the FastAPI backend.

Usage:
    from model.predict import predict_activity
    import numpy as np
    data = np.zeros((128, 9))   # replace with real sensor data
    result = predict_activity(data)
    print(result)
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import tensorflow as tf  # type: ignore

logger = logging.getLogger(__name__)

# ── Default paths (relative to project root) ─────────────────────────────────
_DEFAULT_SAVE_DIR = Path("saved_models")


# ── Lazy-loaded singletons ───────────────────────────────────────────────────

_model: Optional[tf.keras.Model] = None
_labels: Optional[List[str]] = None
_scaler_mean: Optional[np.ndarray] = None
_scaler_std: Optional[np.ndarray] = None
_save_dir: Path = _DEFAULT_SAVE_DIR


def configure_predictor(save_dir: str | Path) -> None:
    """
    Override the directory from which model artefacts are loaded.

    Call this before the first ``predict_activity()`` call, e.g. from the
    FastAPI lifespan handler.

    Args:
        save_dir: Directory containing ``har_model.keras``, ``scaler_mean.npy``,
                  ``scaler_std.npy``, and ``activity_labels.json``.
    """
    global _save_dir
    _save_dir = Path(save_dir)
    logger.info("Predictor configured — save_dir=%s", _save_dir.resolve())


def _load_artifacts() -> None:
    """Load model artefacts into module-level singletons (called lazily)."""
    global _model, _labels, _scaler_mean, _scaler_std

    model_path = _save_dir / "har_model.keras"
    labels_path = _save_dir / "activity_labels.json"
    mean_path = _save_dir / "scaler_mean.npy"
    std_path = _save_dir / "scaler_std.npy"

    for p in (model_path, labels_path, mean_path, std_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Required artefact not found: {p}\n"
                "Run 'python model/train.py' first to train and save the model."
            )

    logger.info("Loading model from %s …", model_path)
    _model = tf.keras.models.load_model(str(model_path))

    with open(labels_path, "r") as f:
        _labels = json.load(f)

    _scaler_mean = np.load(str(mean_path))
    _scaler_std = np.load(str(std_path))
    logger.info("Model and artefacts loaded successfully.")


def _ensure_loaded() -> None:
    """Initialise singletons on first call."""
    if _model is None:
        _load_artifacts()


def is_model_ready() -> bool:
    """
    Return True if all required artefact files are present on disk.

    Does not load the model — fast check for health-endpoint use.
    """
    required = [
        _save_dir / "har_model.keras",
        _save_dir / "activity_labels.json",
        _save_dir / "scaler_mean.npy",
        _save_dir / "scaler_std.npy",
    ]
    return all(p.exists() for p in required)


def predict_activity(data: np.ndarray) -> Dict:
    """
    Run inference on a single sensor window.

    Args:
        data: NumPy array of shape ``(128, 9)`` or ``(1, 128, 9)`` containing
              raw (unnormalized) sensor readings in the order:
              body_acc_x/y/z, body_gyro_x/y/z, total_acc_x/y/z.

    Returns:
        Dictionary with keys:
            - ``predicted_activity`` (str)  — class name
            - ``class_index``        (int)  — 0-based index
            - ``confidence``         (float) — probability of predicted class
            - ``probabilities``      (dict)  — per-class probability map
            - ``inference_time_ms``  (float) — wall-clock inference time

    Raises:
        ValueError:      When input shape is invalid.
        FileNotFoundError: When model artefacts are missing.
    """
    from model.utils import validate_input  # avoid circular import at module load

    _ensure_loaded()

    # ── Validate ─────────────────────────────────────────────────────────
    validate_input(data)

    # ── Preprocess ────────────────────────────────────────────────────────
    if data.ndim == 2:
        data = data[np.newaxis, ...]  # (1, 128, 9)

    # Z-score normalisation using training statistics
    std_safe = np.where(_scaler_std == 0, 1.0, _scaler_std)
    data_norm = (data - _scaler_mean) / std_safe

    # ── Inference ─────────────────────────────────────────────────────────
    t_start = time.perf_counter()
    probs = _model.predict(data_norm, verbose=0)  # (1, 6)
    t_end = time.perf_counter()

    inference_ms = (t_end - t_start) * 1000.0

    probs_flat = probs[0].tolist()
    class_index = int(np.argmax(probs_flat))
    confidence = float(probs_flat[class_index])
    activity = _labels[class_index]

    prob_map = {label: round(float(p), 6) for label, p in zip(_labels, probs_flat)}

    logger.info(
        "Prediction → %s (%.2f%%) in %.1f ms",
        activity,
        confidence * 100,
        inference_ms,
    )

    return {
        "predicted_activity": activity,
        "class_index": class_index,
        "confidence": round(confidence, 6),
        "probabilities": prob_map,
        "inference_time_ms": round(inference_ms, 3),
    }


def reload_model() -> None:
    """
    Force-reload all artefacts from disk.

    Useful after retraining without restarting the server.
    """
    global _model, _labels, _scaler_mean, _scaler_std
    _model = _labels = _scaler_mean = _scaler_std = None
    _load_artifacts()
    logger.info("Model reloaded from disk.")
