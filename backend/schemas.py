"""
schemas.py — Pydantic request / response schemas for the HAR REST API.
"""

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

# ── Health ────────────────────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    """Response schema for GET /health."""

    status: str = Field(..., examples=["ok"])
    model_loaded: bool
    database: str = Field(..., examples=["connected"])
    version: str = Field(..., examples=["1.0.0"])


# ── Model Info ────────────────────────────────────────────────────────────────


class ModelInfoResponse(BaseModel):
    """Response schema for GET /model-info."""

    architecture: str
    input_shape: List[int]
    num_classes: int
    activity_labels: List[str]
    config: Optional[Dict] = None  # model_config is reserved in Pydantic v2


# ── Prediction ────────────────────────────────────────────────────────────────


class WindowPrediction(BaseModel):
    """Per-window prediction summary returned when input > 128 rows."""

    window_index: int
    predicted_activity: str
    confidence: float = Field(..., ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    """
    Response schema for POST /predict.

    Extra fields when the input was automatically pre-processed:

    - ``preprocessing_note`` — human-readable description of what was done
      (padding, windowing, or none).
    - ``windows_count`` — number of 128-sample windows the input was divided
      into (1 when the input was exactly 128 rows or was padded).
    - ``window_predictions`` — per-window results when ``windows_count > 1``
      (majority vote determines ``predicted_activity`` / ``confidence``).
    """

    id: int
    filename: str
    predicted_activity: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: Dict[str, float]
    processing_time_ms: float
    timestamp: datetime
    preprocessing_note: str = Field(
        default="Input is exactly 128 rows — no adjustment needed.",
        description="Describes any automatic pre-processing applied to the input.",
    )
    windows_count: int = Field(
        default=1,
        description="Number of 128-sample windows the input was split into.",
    )
    window_predictions: List[WindowPrediction] = Field(
        default_factory=list,
        description="Per-window predictions (non-empty only when windows_count > 1).",
    )

    model_config = {"from_attributes": True}


# ── History ───────────────────────────────────────────────────────────────────


class PredictionHistoryItem(BaseModel):
    """A single row in the prediction history."""

    id: int
    filename: str
    predicted_activity: str
    confidence: float
    processing_time_ms: Optional[float] = None
    timestamp: datetime

    model_config = {"from_attributes": True}


class HistoryResponse(BaseModel):
    """Response schema for GET /history."""

    total: int
    predictions: List[PredictionHistoryItem]


# ── Analytics ─────────────────────────────────────────────────────────────────


class ActivityCount(BaseModel):
    """Activity name and its occurrence count."""

    activity: str
    count: int


class AnalyticsResponse(BaseModel):
    """Aggregated analytics over all predictions."""

    total_predictions: int
    activity_distribution: List[ActivityCount]
    avg_confidence: float
    latest_predictions: List[PredictionHistoryItem]
