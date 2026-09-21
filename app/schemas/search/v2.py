import uuid

from pydantic import BaseModel


class SearchResultV2(BaseModel):
    product_id: uuid.UUID
    slug: str | None = None
    title: str
    manufacturer: str
    description: str
    image_url: str
    dino_similarity: float
    rank: int


class SearchTimingsV2(BaseModel):
    rectification_ms: float
    embedding_ms: float
    pgvector_ms: float
    total_ms: float


class SearchResponseV2(BaseModel):
    winner: SearchResultV2 | None
    results: list[SearchResultV2]
    timings: SearchTimingsV2
    query_crop: str | None = None
    bbox_crop: str | None = None
    quad_corners: list[list[float]] | None = None
    is_fallback: bool = False


class PredictResponseV2(BaseModel):
    slug: str | None = None
