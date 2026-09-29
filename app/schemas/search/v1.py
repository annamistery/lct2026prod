import uuid

from pydantic import BaseModel


class SearchResult(BaseModel):
    product_id: uuid.UUID
    slug: str | None = None
    title: str
    manufacturer: str
    description: str
    image_url: str
    dino_similarity: float
    sift_score: float | None = None
    inliers: int | None = None
    inlier_ratio: float | None = None
    pgvector_rank: int
    final_rank: int


class SearchTimings(BaseModel):
    embedding_ms: float
    pgvector_ms: float
    sift_ms: float | None = None
    total_ms: float


class SearchResponse(BaseModel):
    winner: SearchResult | None
    results: list[SearchResult]
    timings: SearchTimings
    query_crop: str | None = None
    augmentation_applied: list[str] | None = None


class PredictResponse(BaseModel):
    slug: str | None = None
    status: str | None = None  # "found" | "not_in_catalog"
    stage_reached: str | None = None  # "fusion" (DINOv2 + SigLIP 2) | "v1_only"
    confidence: float | None = None  # SigLIP 2 similarity of the cascade answer
    latency_ms: float | None = None  # server-side recognition time
