"""
main.py — FastAPI application entry point for the HAR backend.

Run locally:
    uvicorn backend.main:app --reload --port 8000

Production (Render):
    uvicorn backend.main:app --host 0.0.0.0 --port $PORT
"""

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# ── Project root on sys.path so model.* imports resolve ──────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.database import init_db
from backend.routes import router

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Lifespan ──────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.

    Startup:
      - Initialise the SQLite database (creates tables if absent)
      - Configure the model predictor with the correct save-dir path
    Shutdown:
      - Graceful cleanup (no-op for SQLite)
    """
    logger.info("=== HAR Backend starting up ===")

    # Initialise database
    init_db()

    # Configure model predictor path
    save_dir = os.getenv("MODEL_PATH", str(PROJECT_ROOT / "saved_models"))
    from model.predict import configure_predictor, is_model_ready

    configure_predictor(save_dir)

    if is_model_ready():
        logger.info("Model artefacts found — ready for inference.")
    else:
        logger.warning(
            "Model artefacts NOT found at '%s'. "
            "Run 'python model/train.py' to train the model first.",
            save_dir,
        )

    yield  # ── application is running ──

    logger.info("=== HAR Backend shutting down ===")


# ── FastAPI App ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Human Activity Recognition API",
    description=(
        "Production-ready REST API for predicting human activities "
        "from smartphone inertial sensor data using a CNN + LSTM model "
        "trained on the UCI HAR Dataset.\n\n"
        "**Model input**: 128 time steps × 9 sensor channels\n\n"
        "**Activities**: WALKING · WALKING_UPSTAIRS · WALKING_DOWNSTAIRS · "
        "SITTING · STANDING · LAYING"
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────

_allowed_origins = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:8501,http://localhost:3000,*",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ────────────────────────────────────────────────────────────────────

app.include_router(router, prefix="")


# ── Root Redirect ─────────────────────────────────────────────────────────────


@app.get("/", include_in_schema=False)
def root():
    return {
        "message": "HAR API is running",
        "docs": "/docs",
        "health": "/health",
    }


# ── Dev entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True,
    )
