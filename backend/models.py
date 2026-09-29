"""
models.py — SQLAlchemy ORM models for the HAR backend.
"""

from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String, Text

from backend.database import Base


class Prediction(Base):
    """
    Stores every activity prediction made via the /predict endpoint.

    Columns
    -------
    id                  Auto-incrementing primary key.
    filename            Name of the uploaded CSV file.
    predicted_activity  Human-readable activity label (e.g. 'WALKING').
    confidence          Model confidence score [0.0, 1.0].
    probabilities_json  JSON string of per-class probability mapping.
    processing_time_ms  Total server-side processing time in milliseconds.
    timestamp           UTC datetime when the prediction was created.
    """

    __tablename__ = "predictions"

    id: int = Column(Integer, primary_key=True, index=True)
    filename: str = Column(String(255), nullable=False)
    predicted_activity: str = Column(String(100), nullable=False)
    confidence: float = Column(Float, nullable=False)
    probabilities_json: str = Column(Text, nullable=True)  # JSON string
    processing_time_ms: float = Column(Float, nullable=True)
    timestamp: datetime = Column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<Prediction id={self.id} activity={self.predicted_activity!r} "
            f"confidence={self.confidence:.3f} ts={self.timestamp}>"
        )
