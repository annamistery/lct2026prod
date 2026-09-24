import uuid

from pydantic import BaseModel

from app.schemas.search.v1 import SearchResult
from app.schemas.search.v4 import SearchResultV4


class V1DecisionDetails(BaseModel):
    is_confident: bool
    top1_similarity: float
    top2_similarity: float | None = None
    margin: float | None = None
    reason: str


class CascadeNeighborItem(BaseModel):
    product_id: uuid.UUID
    slug: str | None = None
    title: str
    manufacturer: str
    dino_similarity: float
    rank_v1: int


class CascadeTimings(BaseModel):
    bbox_detect_ms: float | None = None
    v1_total_ms: float
    decision_ms: float
    v4_total_ms: float | None = None
    total_ms: float


class CascadeSearchResponse(BaseModel):
    stage_reached: str  # "v1_confident" | "v4_refined"
    winner: SearchResultV4 | SearchResult | None = None
    final_results: list[SearchResultV4 | SearchResult]
    decision: V1DecisionDetails
    v1_results: list[SearchResult]
    v1_neighbors: list[CascadeNeighborItem]
    v4_results: list[SearchResultV4] | None = None
    timings: CascadeTimings
    bbox_crop: str | None = None
    v4_query_crop: str | None = None
    vintage_detected: str | None = None


class CascadePredictResponse(BaseModel):
    slug: str | None = None
    stage_reached: str
    confidence: float | None = None
