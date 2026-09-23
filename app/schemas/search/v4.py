import uuid

from pydantic import BaseModel


class SearchResultV4(BaseModel):
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
    ocr_vintage_match: bool | None = None
    ocr_score: float | None = None
    final_score: float
    rank: int


class SearchTimingsV4(BaseModel):
    query_prep_ms: float
    embedding_ms: float
    pgvector_ms: float
    ocr_ms: float | None = None
    total_ms: float


class SearchResponseV4(BaseModel):
    winner: SearchResultV4 | None
    results: list[SearchResultV4]
    timings: SearchTimingsV4
    query_crop: str | None = None
    bbox_crop: str | None = None
    seg_polygon: list[list[float]] | None = None
    is_fallback: bool = False
    vintage_detected: str | None = None


class PredictResponseV4(BaseModel):
    slug: str | None = None
