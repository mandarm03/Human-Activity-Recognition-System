"""
services.py — Business-logic layer for the HAR backend.

Separates all heavy lifting (CSV parsing, model calls, DB writes) from
the route handlers in routes.py, keeping controllers thin.

Variable-length input handling
-------------------------------
The /predict endpoint accepts CSVs with ANY number of rows, as long as
they have exactly 9 columns.  Three cases are handled automatically:

* rows == 128  → single inference, no adjustment needed
* rows  < 128  → zero-padded to 128 at the end (padding note returned)
* rows  > 128  → split into non-overlapping 128-sample windows;
                  majority-vote determines the final prediction.
                  All per-window results are included in the response.
"""

import json
import logging
import time
from collections import Counter
from io import StringIO
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from backend.models import Prediction
from backend.schemas import ActivityCount, WindowPrediction

logger = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────

EXPECTED_COLUMNS = 9
WINDOW_SIZE = 128

SIGNAL_ORDER = [
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


# ── CSV Parsing ───────────────────────────────────────────────────────────────


def parse_csv(content: bytes, filename: str) -> np.ndarray:
    """
    Parse uploaded CSV bytes into a raw NumPy array.

    Accepts files with or without a header row.  If named headers are found
    that match ``SIGNAL_ORDER`` (case-insensitive), columns are reordered to
    canonical order; otherwise positional order is assumed.

    Only the column count is validated here — row count is handled separately
    by ``preprocess_sensor_data()``.

    Args:
        content:  Raw file bytes from the uploaded CSV.
        filename: Original filename (used in error messages).

    Returns:
        NumPy float32 array of shape ``(N, 9)`` where N is the row count.

    Raises:
        ValueError: When the file cannot be parsed or has the wrong column count.
    """
    try:
        text = content.decode("utf-8", errors="replace")

        # ── Detect header ─────────────────────────────────────────────────
        try:
            df_with_header = pd.read_csv(StringIO(text))
            # If all column names look numeric → no real header
            all_numeric_cols = all(
                str(c)
                .replace(".", "")
                .replace("-", "")
                .replace("e", "")
                .replace("+", "")
                .isnumeric()
                for c in df_with_header.columns
            )
            if all_numeric_cols:
                df = pd.read_csv(StringIO(text), header=None)
            else:
                df = df_with_header
        except Exception:
            df = pd.read_csv(StringIO(text), header=None)

        # ── Column remapping if named headers match signal names ──────────
        if not all(isinstance(c, int) for c in df.columns):
            col_lower = {str(c).lower().strip(): c for c in df.columns}
            mapped: Optional[List] = []
            for sig in SIGNAL_ORDER:
                # Try exact match, then no-underscore variant
                key = col_lower.get(sig.lower()) or col_lower.get(
                    sig.replace("_", "").lower()
                )
                if key is not None:
                    mapped.append(df[key].values)
                else:
                    mapped = None
                    break
            data = (
                np.column_stack(mapped).astype(np.float32)
                if mapped
                else df.values.astype(np.float32)
            )
        else:
            data = df.values.astype(np.float32)

    except Exception as exc:
        raise ValueError(f"Could not parse CSV '{filename}': {exc}") from exc

    # ── Validate columns only ─────────────────────────────────────────────
    if data.ndim != 2 or data.shape[1] != EXPECTED_COLUMNS:
        actual_cols = data.shape[1] if data.ndim == 2 else "?"
        raise ValueError(
            f"'{filename}' has {actual_cols} column(s) but exactly {EXPECTED_COLUMNS} "
            f"are required. Expected: {SIGNAL_ORDER}"
        )

    if not np.isfinite(data).all():
        nan_count = np.sum(~np.isfinite(data))
        raise ValueError(
            f"'{filename}' contains {nan_count} NaN/Inf value(s). "
            "Please clean your data before uploading."
        )

    return data  # (N, 9)


# ── Windowing & Padding ───────────────────────────────────────────────────────


def preprocess_sensor_data(
    data: np.ndarray,
    window_size: int = WINDOW_SIZE,
) -> Tuple[np.ndarray, str]:
    """
    Adjust raw sensor data to one or more model-ready windows.

    Rules:
    - ``N == window_size`` → returned as-is (single window).
    - ``N < window_size``  → zero-padded at the **end** to ``window_size``.
    - ``N > window_size``  → sliced into non-overlapping windows of
                             ``window_size``; a partial trailing window is
                             zero-padded and included.

    Args:
        data:        Raw array of shape ``(N, 9)``.
        window_size: Target window length (default 128).

    Returns:
        Tuple of:
            - windows: array of shape ``(W, window_size, 9)``
            - note:    human-readable preprocessing description string
    """
    n_rows = data.shape[0]

    if n_rows == window_size:
        note = f"Input is exactly {window_size} rows — no adjustment needed."
        return data[np.newaxis, ...], note  # (1, 128, 9)

    if n_rows < window_size:
        pad_len = window_size - n_rows
        padded = np.zeros((window_size, EXPECTED_COLUMNS), dtype=np.float32)
        padded[:n_rows] = data
        note = (
            f"Input had {n_rows} rows (< {window_size}). "
            f"Automatically zero-padded {pad_len} row(s) to meet model requirements."
        )
        return padded[np.newaxis, ...], note  # (1, 128, 9)

    # n_rows > window_size → sliding non-overlapping windows
    windows = []
    start = 0
    win_idx = 1
    while start < n_rows:
        end = start + window_size
        chunk = data[start:end]
        if len(chunk) < window_size:
            # Pad final partial window
            pad = np.zeros(
                (window_size - len(chunk), EXPECTED_COLUMNS), dtype=np.float32
            )
            chunk = np.vstack([chunk, pad])
        windows.append(chunk)
        start = end
        win_idx += 1

    n_windows = len(windows)
    note = (
        f"Input had {n_rows} rows (> {window_size}). "
        f"Automatically split into {n_windows} non-overlapping window(s) of {window_size} rows. "
        f"Final activity determined by majority vote across all windows."
    )
    return np.array(windows), note  # (W, 128, 9)


# ── Prediction Service ────────────────────────────────────────────────────────


def run_prediction(
    raw_data: np.ndarray,
    filename: str,
    preprocessing_note: str,
    db: Session,
) -> Dict:
    """
    Run inference on one or more windows and persist the majority-vote result.

    Args:
        raw_data:           Pre-processed windows, shape ``(W, 128, 9)``.
        filename:           Original CSV filename.
        preprocessing_note: Human-readable note about how the data was adjusted.
        db:                 Active SQLAlchemy session.

    Returns:
        Dictionary matching ``PredictionResponse`` schema fields.
    """
    from model.predict import predict_activity  # lazy import avoids circular deps

    t_start = time.perf_counter()
    n_windows = raw_data.shape[0]

    # ── Per-window inference ──────────────────────────────────────────────
    window_results: List[Dict] = []
    for i in range(n_windows):
        res = predict_activity(raw_data[i])  # (128, 9)
        window_results.append(res)

    t_total_ms = (time.perf_counter() - t_start) * 1000.0

    # ── Majority vote ─────────────────────────────────────────────────────
    predicted_activities = [r["predicted_activity"] for r in window_results]
    vote_counter = Counter(predicted_activities)
    final_activity = vote_counter.most_common(1)[0][0]

    # Aggregate probabilities (mean across all windows)
    all_probs = [r["probabilities"] for r in window_results]
    avg_probs = {
        label: round(float(np.mean([p[label] for p in all_probs])), 6)
        for label in all_probs[0]
    }
    # Confidence = mean confidence of windows that voted for the winner
    winner_windows = [
        r for r in window_results if r["predicted_activity"] == final_activity
    ]
    final_confidence = float(np.mean([r["confidence"] for r in winner_windows]))

    # ── Persist to database ───────────────────────────────────────────────
    record = Prediction(
        filename=filename,
        predicted_activity=final_activity,
        confidence=round(final_confidence, 6),
        probabilities_json=json.dumps(avg_probs),
        processing_time_ms=round(t_total_ms, 3),
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    logger.info(
        "Saved prediction id=%d | %s (%.1f%%) | %d window(s) | %.1f ms",
        record.id,
        record.predicted_activity,
        record.confidence * 100,
        n_windows,
        record.processing_time_ms,
    )

    # ── Build per-window summary for response ─────────────────────────────
    window_summaries = [
        {
            "window_index": i + 1,
            "predicted_activity": r["predicted_activity"],
            "confidence": r["confidence"],
        }
        for i, r in enumerate(window_results)
    ]

    return {
        "id": record.id,
        "filename": record.filename,
        "predicted_activity": record.predicted_activity,
        "confidence": record.confidence,
        "probabilities": avg_probs,
        "processing_time_ms": record.processing_time_ms,
        "timestamp": record.timestamp,
        "preprocessing_note": preprocessing_note,
        "windows_count": n_windows,
        "window_predictions": window_summaries,
    }


# ── History & Analytics Services ─────────────────────────────────────────────


def get_all_predictions(
    db: Session,
    limit: int = 100,
    offset: int = 0,
) -> Dict:
    """
    Fetch paginated prediction history from the database.

    Args:
        db:     Active SQLAlchemy session.
        limit:  Maximum rows to return (default 100).
        offset: Row offset for pagination (default 0).

    Returns:
        Dict with ``total`` count and ``predictions`` list.
    """
    total = db.query(Prediction).count()
    rows = (
        db.query(Prediction)
        .order_by(Prediction.timestamp.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    return {"total": total, "predictions": rows}


def get_analytics(db: Session, latest_n: int = 5) -> Dict:
    """
    Compute aggregated analytics over all predictions.

    Args:
        db:       Active SQLAlchemy session.
        latest_n: Number of recent predictions to include.

    Returns:
        Dict matching ``AnalyticsResponse`` schema.
    """
    from sqlalchemy import func

    total = db.query(Prediction).count()

    # Activity distribution
    dist_rows = (
        db.query(
            Prediction.predicted_activity, func.count(Prediction.id).label("count")
        )
        .group_by(Prediction.predicted_activity)
        .order_by(func.count(Prediction.id).desc())
        .all()
    )
    distribution = [
        ActivityCount(activity=row.predicted_activity, count=row.count)
        for row in dist_rows
    ]

    # Average confidence
    avg_conf_row = db.query(func.avg(Prediction.confidence)).scalar()
    avg_conf = round(float(avg_conf_row or 0.0), 4)

    # Latest predictions
    latest = (
        db.query(Prediction).order_by(Prediction.timestamp.desc()).limit(latest_n).all()
    )

    return {
        "total_predictions": total,
        "activity_distribution": distribution,
        "avg_confidence": avg_conf,
        "latest_predictions": latest,
    }
