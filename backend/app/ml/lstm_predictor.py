"""
LSTM-based next-position predictor.

Architecture:
  Input  : sequence of SEQ_LEN steps × 6 features
           [lat_norm, lon_norm, hour_sin, hour_cos, dist_home_norm, speed_norm]
  Output : [lat_norm, lon_norm]  →  denormalized to WGS-84 degrees

Training is triggered automatically after each CSV upload (if enough data).
The trained model + scalers are persisted to disk (ml_models/<chat_id>/).
"""
import os
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple

import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import joblib

from app.config import get_settings

log = logging.getLogger(__name__)
settings = get_settings()

SEQ_LEN   = settings.sequence_len      # 6 input steps
EPOCHS    = settings.lstm_epochs
BATCH     = settings.lstm_batch_size
MIN_ROWS  = SEQ_LEN + 50              # need at least this many records to train


class LSTMPredictor:
    """Wraps a Keras LSTM model with fit/predict and persistence."""

    def __init__(self, chat_id: int):
        self.chat_id   = chat_id
        self.model_dir = Path(settings.ml_model_dir) / str(chat_id)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.model:      Optional[keras.Model] = None
        self.scaler_in:  Optional[MinMaxScaler] = None
        self.scaler_out: Optional[MinMaxScaler] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> None:
        """
        Train the LSTM on a DataFrame with columns:
          [ts, latitude, longitude, distance_home_m, vitesse_ms]
        """
        if len(df) < MIN_ROWS:
            log.warning("Not enough rows (%d < %d) to train LSTM for cat %d",
                        len(df), MIN_ROWS, self.chat_id)
            return

        X, y = self._build_sequences(df)
        if X is None:
            return

        self.model = self._build_model(X.shape[1:])
        self.model.fit(
            X, y,
            epochs=EPOCHS,
            batch_size=BATCH,
            validation_split=0.1,
            verbose=0,
            callbacks=[keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)],
        )
        self._save()
        log.info("LSTM trained for cat %d (%d sequences)", self.chat_id, len(X))

    def predict_next(self, df: pd.DataFrame) -> Optional[Tuple[float, float]]:
        """
        Given the most recent rows, predict the next (lat, lon).
        Returns None if model is not available or too few data.
        """
        if self.model is None:
            self._load()
        if self.model is None:
            return None

        if len(df) < SEQ_LEN:
            return None

        features = self._extract_features(df.tail(SEQ_LEN))
        X = self.scaler_in.transform(features).reshape(1, SEQ_LEN, -1)
        y_norm = self.model.predict(X, verbose=0)[0]
        lat, lon = self.scaler_out.inverse_transform([y_norm])[0]
        return float(lat), float(lon)

    def is_trained(self) -> bool:
        return (self.model_dir / "model.keras").exists()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """Build feature matrix from a slice of the DataFrame."""
        hours = pd.to_datetime(df["ts"]).dt.hour + pd.to_datetime(df["ts"]).dt.minute / 60
        hour_sin = np.sin(2 * np.pi * hours / 24).values
        hour_cos = np.cos(2 * np.pi * hours / 24).values

        return np.column_stack([
            df["latitude"].values,
            df["longitude"].values,
            hour_sin,
            hour_cos,
            df["distance_home_m"].fillna(0).values,
            df["vitesse_ms"].fillna(0).values,
        ])

    def _build_sequences(self, df: pd.DataFrame):
        feats = self._extract_features(df)
        targets = df[["latitude", "longitude"]].values

        # Fit scalers
        self.scaler_in  = MinMaxScaler()
        self.scaler_out = MinMaxScaler()
        feats_s   = self.scaler_in.fit_transform(feats)
        targets_s = self.scaler_out.fit_transform(targets)

        X, y = [], []
        for i in range(len(feats_s) - SEQ_LEN):
            X.append(feats_s[i : i + SEQ_LEN])
            y.append(targets_s[i + SEQ_LEN])

        if not X:
            return None, None
        return np.array(X), np.array(y)

    @staticmethod
    def _build_model(input_shape: Tuple) -> keras.Model:
        model = keras.Sequential([
            keras.layers.Input(shape=input_shape),
            keras.layers.LSTM(64, return_sequences=True),
            keras.layers.Dropout(0.2),
            keras.layers.LSTM(32),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(2),   # lat, lon
        ])
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        return model

    def _save(self) -> None:
        self.model.save(self.model_dir / "model.keras")
        joblib.dump(self.scaler_in,  self.model_dir / "scaler_in.pkl")
        joblib.dump(self.scaler_out, self.model_dir / "scaler_out.pkl")

    def _load(self) -> None:
        model_path = self.model_dir / "model.keras"
        if not model_path.exists():
            return
        self.model      = keras.models.load_model(str(model_path))
        self.scaler_in  = joblib.load(self.model_dir / "scaler_in.pkl")
        self.scaler_out = joblib.load(self.model_dir / "scaler_out.pkl")
        log.info("Loaded LSTM model for cat %d", self.chat_id)
