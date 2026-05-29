from pydantic import BaseModel, ConfigDict
from datetime import datetime
from typing import Optional, List


# ─── Chat ────────────────────────────────────────────────────────────────────

class ChatCreate(BaseModel):
    nom: str
    race: Optional[str] = None
    couleur: Optional[str] = None
    poids_kg: Optional[float] = None
    lat_home: float = 48.8566
    lon_home: float = 2.3522


class ChatOut(ChatCreate):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime


# ─── Position ────────────────────────────────────────────────────────────────

class PositionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    chat_id: int
    ts: datetime
    latitude: float
    longitude: float
    vitesse_ms: Optional[float]
    distance_home_m: Optional[float]


# ─── Home Range ──────────────────────────────────────────────────────────────

class HomeRangeOut(BaseModel):
    chat_id: int
    area_km2: float
    polygon_geojson: dict          # GeoJSON Polygon
    centroid: dict                 # {"lat": float, "lon": float}
    n_points: int


# ─── Prediction ──────────────────────────────────────────────────────────────

class PredictionOut(BaseModel):
    chat_id: int
    predicted_latitude: float
    predicted_longitude: float
    confidence: Optional[float] = None
    model_version: str = "lstm_v1"


# ─── Upload result ───────────────────────────────────────────────────────────

class UploadResult(BaseModel):
    chat_id: int
    inserted: int
    skipped: int
    model_retrained: bool
