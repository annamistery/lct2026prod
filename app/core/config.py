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
    yolo_seg_model_path: Path = Path("models/yolo_seg.pt")
    embedding_model_name: str = "dinov2-label-v1"
    embedding_model_v2_name: str = "dinov2-label-v2-rectified"
    dino_v3_model_path: Path = Path("models/dinov2_label_finetuned_v3")
    dino_v3_base_model_path: Path = Path("models/dinov2-base")
    embedding_model_v3_name: str = "dinov2-base-label-v3-lora"
    embedding_dimension: int = 384
    embedding_dimension_v3: int = 768
    canonical_size: int = 256
    canonical_size_v3: int = 518
    enable_sift_rerank_v3: bool = True
    sift_rerank_v3_threshold: float = 0.85
    v3_vote_pool_size: int = 50
    yolo_confidence: float = Field(0.25, ge=0, le=1)
    yolo_seg_confidence: float = Field(0.25, ge=0, le=1)
    max_upload_bytes: int = Field(10_485_760, ge=1024)
    max_image_pixels: int = Field(25_000_000, ge=65_536)
    max_top_k: int = Field(20, ge=1, le=100)
    candidate_pool_size: int = Field(20, ge=1, le=100)
    gpu_concurrency: int = Field(1, ge=1, le=8)
    sift_concurrency: int = Field(2, ge=1, le=16)
    enable_sift_rerank: bool = Field(False)
    cors_origins: str = ""
    log_level: str = "INFO"
    # v4 SigLIP 2 pipeline
    siglip_v4_model_path: Path = Path("models/siglip2_v4_finetuned")
    siglip_v4_base_model_path: Path = Path("models/siglip2-base-patch16-512")
    embedding_model_v4_name: str = "siglip2-v4"
    embedding_dimension_v4: int = 768
    canonical_size_v4: int = 518          # server dataset is 518×518 (same letterbox as v3)
    enable_ocr_rerank_v4: bool = True
    ocr_rerank_weight_sim: float = 0.5
    ocr_rerank_weight_vintage: float = 0.3
    ocr_rerank_weight_text: float = 0.2
    v4_vote_pool_size: int = Field(50, ge=1, le=500)
    v4_candidate_pool_size: int = Field(20, ge=1, le=100)
    sommelier_csv_path: Path = Path("app/sommelier/data/wines_integrated.csv")
    sommelier_max_sessions: int = Field(500, ge=1, le=100_000)
    sommelier_session_ttl_seconds: int = Field(3600, ge=60, le=86_400)

    @property
    def allowed_origins(self) -> list[str]:
        return [value.strip() for value in self.cors_origins.split(",") if value.strip()]

    @property
    def resolved_yolo_model_path(self) -> Path:
        if self.yolo_model_path.is_file():
            return self.yolo_model_path
        if Path("/models/yolo_label.pt").is_file():
            return Path("/models/yolo_label.pt")
        base = Path(__file__).resolve().parent.parent.parent
        cand = base / "models" / "yolo_label.pt"
        if cand.is_file():
            return cand
        return self.yolo_model_path

    @property
    def resolved_yolo_seg_model_path(self) -> Path:
        if self.yolo_seg_model_path.is_file():
            return self.yolo_seg_model_path
        if Path("/models/yolo_seg.pt").is_file():
            return Path("/models/yolo_seg.pt")
        base = Path(__file__).resolve().parent.parent.parent
        cand = base / "models" / "yolo_seg.pt"
        if cand.is_file():
            return cand
        return self.yolo_seg_model_path

    @property
    def resolved_sommelier_csv_path(self) -> Path:
        if self.sommelier_csv_path.is_file():
            return self.sommelier_csv_path
        cand = Path(__file__).resolve().parent.parent / "sommelier" / "data" / "wines_integrated.csv"
        if cand.is_file():
            return cand
        return self.sommelier_csv_path

    @property
    def resolved_siglip_v4_model_path(self) -> Path:
        base = Path(__file__).resolve().parent.parent.parent
        candidates = [
            self.siglip_v4_model_path,
            Path("/models/siglip2_v4_finetuned"),
            self.media_dir / "models" / "siglip2_v4_finetuned",
            base / "models" / "siglip2_v4_finetuned",
        ]
        for cand in candidates:
            if cand.is_dir() and ((cand / "adapter_model.bin").is_file() or (cand / "adapter_model.safetensors").is_file()):
                return cand
        for cand in candidates:
            if cand.is_dir():
                return cand
        return self.siglip_v4_model_path

    @property
    def resolved_siglip_v4_base_model_path(self) -> Path:
        base = Path(__file__).resolve().parent.parent.parent
        candidates = [
            self.siglip_v4_base_model_path,
            Path("/models/siglip2-base-patch16-512"),
            self.media_dir / "models" / "siglip2-base-patch16-512",
            base / "models" / "siglip2-base-patch16-512",
        ]
        for cand in candidates:
            if cand.is_dir() and ((cand / "model.safetensors").is_file() or (cand / "pytorch_model.bin").is_file()):
                return cand
        for cand in candidates:
            if cand.is_dir():
                return cand
        return self.siglip_v4_base_model_path

    @property
    def resolved_dino_model_path(self) -> Path:
        base = Path(__file__).resolve().parent.parent.parent
        candidates = [
            self.dino_model_path,
            Path("/models/dinov2_label_finetuned"),
            self.media_dir / "models" / "dinov2_label_finetuned",
            base / "models" / "dinov2_label_finetuned",
        ]
        for cand in candidates:
            if cand.is_dir() and ((cand / "adapter_model.bin").is_file() or (cand / "adapter_model.safetensors").is_file()):
                return cand
        for cand in candidates:
            if cand.is_dir():
                return cand
        return self.dino_model_path

    @property
    def resolved_dino_base_model_path(self) -> Path:
        base = Path(__file__).resolve().parent.parent.parent
        candidates = [
            self.dino_base_model_path,
            Path("/models/dinov2-small"),
            self.media_dir / "models" / "dinov2-small",
            base / "models" / "dinov2-small",
        ]
        for cand in candidates:
            if cand.is_dir() and ((cand / "model.safetensors").is_file() or (cand / "pytorch_model.bin").is_file()):
                return cand
        for cand in candidates:
            if cand.is_dir():
                return cand
        return self.dino_base_model_path

    @property
    def resolved_sommelier_feedback_path(self) -> Path:
        """Пишем фидбек сомелье в media_dir: app/ в проде смонтирован read-only, media — единственный writable путь."""
        return self.media_dir / "sommelier" / "feedback.jsonl"


@lru_cache
def get_settings() -> Settings:
    return Settings()
