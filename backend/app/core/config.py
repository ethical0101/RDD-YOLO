"""Application settings, loaded from environment variables / the project .env file."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", env_prefix="RDD_", extra="ignore")

    app_name: str = "RDD-YOLO API"
    environment: str = "development"
    log_level: str = "INFO"

    # SQLite by default; any SQLAlchemy URL works (e.g. postgresql+psycopg://user:pass@host/db)
    database_url: str = f"sqlite:///{(PROJECT_ROOT / 'outputs' / 'rdd_yolo.db').as_posix()}"

    # Checkpoint used for inference. Empty -> auto-select the best available in models/weights.
    model_weights: str = ""
    model_device: str = ""  # "" = auto (CUDA if available), "cpu", "0"
    model_imgsz: int = 640
    default_confidence: float = 0.25

    outputs_dir: Path = PROJECT_ROOT / "outputs"
    experiments_dir: Path = PROJECT_ROOT / "experiments"
    weights_dir: Path = PROJECT_ROOT / "models" / "weights"
    dataset_dir: Path = PROJECT_ROOT / "dataset" / "processed" / "rdd2022_yolo"

    max_upload_mb: int = 200
    video_default_stride: int = 3
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    # When true, the API does not load a model at startup (used by unit tests).
    lazy_model: bool = False

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
