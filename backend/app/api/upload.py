"""
Upload & prediction endpoints.

POST /upload/{chat_id}  – ingest a GPS CSV, then retrain LSTM
GET  /predict/{chat_id} – predict next position with the trained LSTM
"""
import logging
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.orm import Chat, Position
from app.models.schemas import UploadResult, PredictionOut
from app.services.ingestion import ingest_csv
from app.ml.lstm_predictor import LSTMPredictor
from app.config import get_settings

log      = logging.getLogger(__name__)
settings = get_settings()
router   = APIRouter(tags=["upload & prediction"])


# ── Upload ────────────────────────────────────────────────────────────────────

@router.post("/upload/{chat_id}", response_model=UploadResult)
async def upload_csv(
    chat_id:    int,
    background: BackgroundTasks,
    file:       UploadFile = File(...),
    db:         Session    = Depends(get_db),
):
    """
    Upload a GPS CSV (timestamp, latitude, longitude).
    After insertion the LSTM is retrained in the background.
    """
    cat = db.query(Chat).filter(Chat.id == chat_id).first()
    if not cat:
        raise HTTPException(404, detail=f"Cat {chat_id} not found")

    if not file.filename.endswith(".csv"):
        raise HTTPException(400, detail="Only .csv files are accepted")

    content = await file.read()
    try:
        inserted, skipped = ingest_csv(
            content, chat_id, db, cat.lat_home, cat.lon_home
        )
    except ValueError as e:
        raise HTTPException(400, detail=str(e))

    # Trigger async model retraining
    background.add_task(_retrain, chat_id, db)

    return UploadResult(
        chat_id=chat_id,
        inserted=inserted,
        skipped=skipped,
        model_retrained=False,   # retraining is async
    )


# ── Prediction ────────────────────────────────────────────────────────────────

@router.get("/predict/{chat_id}", response_model=PredictionOut)
def predict_next(chat_id: int, db: Session = Depends(get_db)):
    """Return the predicted next GPS position using the LSTM model."""
    cat = db.query(Chat).filter(Chat.id == chat_id).first()
    if not cat:
        raise HTTPException(404, detail=f"Cat {chat_id} not found")

    predictor = LSTMPredictor(chat_id)
    if not predictor.is_trained():
        raise HTTPException(
            503,
            detail="Model not trained yet. Upload a CSV first (need ≥ 56 data points).",
        )

    rows = (
        db.query(Position)
        .filter(Position.chat_id == chat_id)
        .order_by(Position.ts.desc())
        .limit(settings.sequence_len)
        .all()
    )

    df = pd.DataFrame([
        {
            "ts":              r.ts,
            "latitude":        r.latitude,
            "longitude":       r.longitude,
            "distance_home_m": r.distance_home_m or 0.0,
            "vitesse_ms":      r.vitesse_ms or 0.0,
        }
        for r in reversed(rows)
    ])

    result = predictor.predict_next(df)
    if result is None:
        raise HTTPException(503, detail="Prediction failed – insufficient recent data.")

    pred_lat, pred_lon = result
    return PredictionOut(
        chat_id=chat_id,
        predicted_latitude=pred_lat,
        predicted_longitude=pred_lon,
    )


# ── Background retraining ─────────────────────────────────────────────────────

def _retrain(chat_id: int, db: Session):
    """Load all positions for chat_id and retrain the LSTM."""
    try:
        rows = (
            db.query(Position)
            .filter(Position.chat_id == chat_id)
            .order_by(Position.ts)
            .all()
        )
        df = pd.DataFrame([
            {
                "ts":              r.ts,
                "latitude":        r.latitude,
                "longitude":       r.longitude,
                "distance_home_m": r.distance_home_m or 0.0,
                "vitesse_ms":      r.vitesse_ms or 0.0,
            }
            for r in rows
        ])
        predictor = LSTMPredictor(chat_id)
        predictor.fit(df)
    except Exception as exc:
        log.error("LSTM retraining failed for cat %d: %s", chat_id, exc)
