import uuid

from pydantic import BaseModel


class SearchResultV3(BaseModel):
    product_id: uuid.UUID
    slug: str | None = None
    title: str
    manufacturer: str
    description: str
    image_url: str
    matched_aug_image: str | None = None
    dino_similarity: float
    vote_count: int
    vote_ratio: float
    matched_aug_name: str
    sift_score: float | None = None
    sift_inliers: int | None = None
    rank: int


class SearchTimingsV3(BaseModel):
    query_prep_ms: float
    embedding_ms: float
    pgvector_ms: float
    sift_ms: float | None = None
    total_ms: float


class SearchResponseV3(BaseModel):
    winner: SearchResultV3 | None
    results: list[SearchResultV3]
    timings: SearchTimingsV3
    query_crop: str | None = None
    bbox_crop: str | None = None
    seg_polygon: list[list[float]] | None = None
    is_fallback: bool = False


class PredictResponseV3(BaseModel):
    slug: str | None = None
