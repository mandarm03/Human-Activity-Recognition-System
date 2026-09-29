"""
routes.py — FastAPI route handlers for the HAR REST API.

Endpoints:
    POST /predict     — Upload CSV and predict activity
    GET  /history     — Paginated prediction history
    GET  /health      — API and model health check
    GET  /model-info  — Model architecture information
    GET  /analytics   — Aggregated prediction statistics
"""

import json
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.schemas import (
    AnalyticsResponse,
    HealthResponse,
    HistoryResponse,
    ModelInfoResponse,
    PredictionResponse,
)
from backend.services import (
    get_all_predictions,
    get_analytics,
    parse_csv,
    preprocess_sensor_data,
    run_prediction,
)

logger = logging.getLogger(__name__)
router = APIRouter()
SAVE_DIR = Path("saved_models")

# ── Health ────────────────────────────────────────────────────────────────────


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["System"],
)
def health_check(db: Session = Depends(get_db)) -> HealthResponse:
    """
    Returns the operational status of the API and model.

    - **status**: ``ok`` or ``degraded``
    - **model_loaded**: whether the trained model artefacts exist
    - **database**: ``connected`` or ``error``
    """
    from model.predict import is_model_ready

    try:
        db.execute(__import__("sqlalchemy").text("SELECT 1"))
        db_status = "connected"
    except Exception as exc:
        logger.error("DB health check failed: %s", exc)
        db_status = "error"

    model_ready = is_model_ready()
    overall = "ok" if (db_status == "connected" and model_ready) else "degraded"

    return HealthResponse(
        status=overall,
        model_loaded=model_ready,
        database=db_status,
        version="1.0.0",
    )


# ── Model Info ────────────────────────────────────────────────────────────────


@router.get(
    "/model-info",
    response_model=ModelInfoResponse,
    summary="Model architecture information",
    tags=["System"],
)
def model_info() -> ModelInfoResponse:
    """
    Returns metadata about the trained CNN + LSTM model.
    """
    from model.utils import ACTIVITY_LABELS

    config_path = SAVE_DIR / "model_config.json"
    model_cfg: Optional[dict] = None
    if config_path.exists():
        with open(config_path, "r") as f:
            model_cfg = json.load(f)

    return ModelInfoResponse(
        architecture="CNN + LSTM",
        input_shape=[128, 9],
        num_classes=6,
        activity_labels=ACTIVITY_LABELS,
        config=model_cfg,
    )


# ── Predict ───────────────────────────────────────────────────────────────────


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict human activity from sensor CSV",
    tags=["Prediction"],
)
async def predict(
    file: UploadFile = File(
        ..., description="CSV with any number of rows and exactly 9 sensor columns"
    ),
    db: Session = Depends(get_db),
) -> PredictionResponse:
    """
    Upload a CSV file and receive an activity prediction.

    **CSV format — only one hard requirement:**
    - Exactly **9 columns** (sensor channels in order:
      body_acc_x/y/z, body_gyro_x/y/z, total_acc_x/y/z)
    - Named headers are auto-mapped; numeric-only files are accepted as-is.

    **Variable row count — handled automatically:**
    - ``rows == 128`` → single inference, no adjustment.
    - ``rows < 128``  → zero-padded to 128 rows.
    - ``rows > 128``  → split into non-overlapping 128-sample windows;
      majority vote determines the final prediction.

    **Returns**
    - Predicted activity label (majority vote when multi-window)
    - Confidence score (0–1)
    - Per-class averaged probabilities
    - ``preprocessing_note`` describing any automatic adjustments
    - ``window_predictions`` listing each window's result
    """
    from model.predict import is_model_ready

    if not is_model_ready():
        raise HTTPException(
            status_code=503,
            detail=(
                "Model artefacts not found. "
                "Run 'python model/train.py' first to train the model."
            ),
        )

    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail=f"Only CSV files are accepted. Got: {file.filename}",
        )

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Step 1 — parse CSV (validates columns=9, reorders named headers)
    try:
        raw_data = parse_csv(content, file.filename)
    except ValueError as exc:
        logger.warning("CSV parse failed for '%s': %s", file.filename, exc)
        raise HTTPException(status_code=422, detail=str(exc))

    # Step 2 — auto-adjust row count (pad / window)
    windows, preprocessing_note = preprocess_sensor_data(raw_data)
    logger.info(
        "%s → %d row(s) → %d window(s) | %s",
        file.filename,
        raw_data.shape[0],
        windows.shape[0],
        preprocessing_note,
    )

    # Step 3 — inference + DB write
    try:
        result = run_prediction(windows, file.filename, preprocessing_note, db)
    except Exception as exc:
        logger.error("Prediction error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

    return PredictionResponse(**result)


# ── History ───────────────────────────────────────────────────────────────────


@router.get(
    "/history",
    response_model=HistoryResponse,
    summary="Paginated prediction history",
    tags=["History"],
)
def history(
    limit: int = Query(default=50, ge=1, le=500, description="Max results per page"),
    offset: int = Query(default=0, ge=0, description="Row offset"),
    db: Session = Depends(get_db),
) -> HistoryResponse:
    """
    Returns past predictions stored in the database, newest first.

    Supports pagination via ``limit`` and ``offset`` query parameters.
    """
    data = get_all_predictions(db, limit=limit, offset=offset)
    return HistoryResponse(total=data["total"], predictions=data["predictions"])


# ── Analytics ─────────────────────────────────────────────────────────────────


@router.get(
    "/analytics",
    response_model=AnalyticsResponse,
    summary="Aggregated prediction analytics",
    tags=["Analytics"],
)
def analytics(
    latest_n: int = Query(
        default=5, ge=1, le=50, description="Number of latest predictions"
    ),
    db: Session = Depends(get_db),
) -> AnalyticsResponse:
    """
    Returns summary statistics over all predictions including:
    - Total prediction count
    - Per-activity distribution
    - Average confidence score
    - Most recent predictions
    """
    data = get_analytics(db, latest_n=latest_n)
    return AnalyticsResponse(**data)
