"""
CSV ingestion service.
Reads a CSV file, validates columns, applies median filter,
computes per-point metrics, and bulk-inserts into the DB.
"""
import io
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Tuple

from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.models.orm import Position
from app.services.geo import haversine_m, speed_ms, median_filter_positions
from app.config import get_settings

settings = get_settings()

REQUIRED_COLS = {"timestamp", "latitude", "longitude"}
MAX_SPEED_MS  = 10.0   # 36 km/h – cat top speed; used to flag outliers


def ingest_csv(
    file_bytes: bytes,
    chat_id: int,
    db: Session,
    lat_home: float,
    lon_home: float,
) -> Tuple[int, int]:
    """
    Parse and insert GPS positions from a CSV file.

    Parameters
    ----------
    file_bytes : raw bytes of the uploaded CSV
    chat_id    : target cat id
    db         : SQLAlchemy session
    lat_home, lon_home : home coordinates for distance calculation

    Returns
    -------
    (inserted, skipped) counts
    """
    df = _parse_csv(file_bytes)
    df = _clean(df, lat_home, lon_home)
    inserted, skipped = _bulk_insert(df, chat_id, db)
    return inserted, skipped


# ─── Private helpers ──────────────────────────────────────────────────────────

def _parse_csv(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(file_bytes))
    df.columns = df.columns.str.strip().str.lower()

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing CSV columns: {missing}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["latitude"]  = pd.to_numeric(df["latitude"],  errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    n_before = len(df)
    df.dropna(subset=["timestamp", "latitude", "longitude"], inplace=True)
    df.drop_duplicates(subset=["timestamp"], inplace=True)
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def _clean(df: pd.DataFrame, lat_home: float, lon_home: float) -> pd.DataFrame:
    # Median spatial filter to remove GPS noise
    lats_f, lons_f = median_filter_positions(
        df["latitude"].tolist(), df["longitude"].tolist(), window=5
    )
    df["latitude"]  = lats_f
    df["longitude"] = lons_f

    # Distance to home
    df["distance_home_m"] = df.apply(
        lambda r: haversine_m(r["latitude"], r["longitude"], lat_home, lon_home),
        axis=1,
    )

    # Speed between consecutive points
    vitesses = [None]
    for i in range(1, len(df)):
        v = speed_ms(
            df.at[i - 1, "latitude"], df.at[i - 1, "longitude"], df.at[i - 1, "timestamp"],
            df.at[i,     "latitude"], df.at[i,     "longitude"],  df.at[i,     "timestamp"],
        )
        vitesses.append(v)
    df["vitesse_ms"] = vitesses

    return df


def _bulk_insert(df: pd.DataFrame, chat_id: int, db: Session) -> Tuple[int, int]:
    records = [
        {
            "chat_id":         chat_id,
            "ts":              row["timestamp"].to_pydatetime(),
            "latitude":        row["latitude"],
            "longitude":       row["longitude"],
            "vitesse_ms":      row["vitesse_ms"],
            "distance_home_m": row["distance_home_m"],
        }
        for _, row in df.iterrows()
    ]

    if not records:
        return 0, 0

    stmt = pg_insert(Position).values(records)
    stmt = stmt.on_conflict_do_nothing()   # skip exact duplicates by PK

    result   = db.execute(stmt)
    db.commit()
    inserted = result.rowcount
    skipped  = len(records) - inserted
    return inserted, skipped
