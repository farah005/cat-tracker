"""
CatTracker API – FastAPI entry point.
"""
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.database import engine, Base
from app.api import cats, positions, upload

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s – %(message)s",
)

# ── DB bootstrap (create tables if they don't exist) ─────────────────────────
Base.metadata.create_all(bind=engine)

# ── Application ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="🐱 CatTracker API",
    description="GPS tracking & ML prediction for domestic cats",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(cats.router)
app.include_router(positions.router)
app.include_router(upload.router)


@app.get("/health", tags=["meta"])
def health():
    return {"status": "ok", "version": "1.0.0"}
