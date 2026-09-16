from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "LCT2026"
    environment: str = "production"
    database_url: str = "postgresql+asyncpg://lct:lct@localhost:5432/lct2026"
    media_dir: Path = Path("media")
    import_staging_dir: Path = Path("imports/staging")
    max_import_items: int = Field(100_000, ge=1, le=1_000_000)
    dino_model_path: Path = Path("models/dinov2_label_finetuned")
    dino_base_model_path: Path = Path("models/dinov2-small")
    yolo_model_path: Path = Path("models/yolo_label.pt")
    embedding_model_name: str = "dinov2-label-v1"
    embedding_dimension: int = 384
    canonical_size: int = 256
    yolo_confidence: float = Field(0.25, ge=0, le=1)
    max_upload_bytes: int = Field(10_485_760, ge=1024)
    max_image_pixels: int = Field(25_000_000, ge=65_536)
    max_top_k: int = Field(20, ge=1, le=100)
    candidate_pool_size: int = Field(20, ge=1, le=100)
    gpu_concurrency: int = Field(1, ge=1, le=8)
    sift_concurrency: int = Field(2, ge=1, le=16)
    cors_origins: str = ""
    log_level: str = "INFO"

    @property
    def allowed_origins(self) -> list[str]:
        return [value.strip() for value in self.cors_origins.split(",") if value.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
