"""
database.py — SQLAlchemy database engine and session factory.

Reads the database URL from the DATABASE_URL environment variable
(falls back to a local SQLite file in the project root).
"""

import logging
import os
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

logger = logging.getLogger(__name__)

# ── Database URL ──────────────────────────────────────────────────────────────

_DEFAULT_DB_PATH = (
    Path(__file__).resolve().parent.parent / "database" / "har_predictions.db"
)
_DEFAULT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    f"sqlite:///{_DEFAULT_DB_PATH}",
)

# ── Engine ────────────────────────────────────────────────────────────────────

# connect_args is only supported for SQLite
_connect_args = (
    {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
)

engine = create_engine(
    DATABASE_URL,
    connect_args=_connect_args,
    echo=False,
)

# Enable WAL mode for better concurrent read performance on SQLite
if DATABASE_URL.startswith("sqlite"):

    @event.listens_for(Engine, "connect")
    def _set_wal_mode(dbapi_conn, _connection_record):  # type: ignore[misc]
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.close()


# ── Session Factory ───────────────────────────────────────────────────────────

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

# ── Declarative Base ──────────────────────────────────────────────────────────


class Base(DeclarativeBase):
    """Base class for all ORM models."""

    pass


# ── Dependency ────────────────────────────────────────────────────────────────


def get_db():
    """
    FastAPI dependency that yields a database session and ensures cleanup.

    Usage:
        @router.get("/example")
        def example(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create all tables defined by ORM models (idempotent)."""
    # Import models here to ensure they are registered with Base
    from backend import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    logger.info("Database initialised at: %s", DATABASE_URL)
