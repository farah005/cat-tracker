from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime

from app.database import get_db
from app.models.orm import Chat, Position
from app.models.schemas import PositionOut, HomeRangeOut
from app.services.geo import compute_convex_hull

router = APIRouter(prefix="/positions", tags=["positions"])


@router.get("/{chat_id}", response_model=List[PositionOut])
def get_positions(
    chat_id: int,
    limit:   int      = Query(1000, le=10_000),
    since:   Optional[datetime] = None,
    db:      Session  = Depends(get_db),
):
    """Return GPS positions for a cat, optionally filtered by date."""
    _ensure_cat(chat_id, db)
    q = db.query(Position).filter(Position.chat_id == chat_id)
    if since:
        q = q.filter(Position.ts >= since)
    q = q.order_by(Position.ts.desc()).limit(limit)
    return q.all()


@router.get("/{chat_id}/home-range", response_model=HomeRangeOut)
def home_range(chat_id: int, db: Session = Depends(get_db)):
    """
    Compute the Minimum Convex Polygon (MCP / convex hull) home range.
    Returns a GeoJSON polygon + area in km².
    """
    _ensure_cat(chat_id, db)
    rows = (
        db.query(Position.latitude, Position.longitude)
        .filter(Position.chat_id == chat_id)
        .all()
    )
    if len(rows) < 3:
        raise HTTPException(400, detail="Need at least 3 positions to compute home range")

    lats = [r.latitude  for r in rows]
    lons = [r.longitude for r in rows]

    geojson, area_km2, (clat, clon) = compute_convex_hull(lats, lons)

    return HomeRangeOut(
        chat_id=chat_id,
        area_km2=round(area_km2, 4),
        polygon_geojson=geojson,
        centroid={"lat": clat, "lon": clon},
        n_points=len(rows),
    )


def _ensure_cat(chat_id: int, db: Session):
    if not db.query(Chat).filter(Chat.id == chat_id).first():
        raise HTTPException(404, detail=f"Cat {chat_id} not found")
