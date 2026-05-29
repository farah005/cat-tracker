from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    database_url: str = "postgresql://cattracker:cattracker_secret@localhost:5432/cattracker_db"
    lat_home: float = 48.8566
    lon_home: float = 2.3522
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    ml_model_dir: str = "/app/ml_models"
    sequence_len: int = 6   # LSTM input window
    lstm_epochs: int = 50
    lstm_batch_size: int = 32

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    return Settings()
