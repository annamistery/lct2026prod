import uuid

from pydantic import BaseModel

from app.schemas.search.v1 import SearchResult


class CascadeCandidate(BaseModel):
    product_id: uuid.UUID
    slug: str | None = None
    title: str
    manufacturer: str
    description: str
    image_url: str
    v1_similarity: float  # DINOv2: best of the letterbox and segmented views
    v4_similarity: float | None = None  # SigLIP 2: best of the two views (None when v4 is unavailable)
    fusion_score: float  # ranking score: v4(letterbox) + v4(segmented) + w * (v1(letterbox) + v1(segmented))
    rank: int
    # Kept for clients written against the previous cascade response
    dino_similarity: float
    final_score: float


class CascadeDecision(BaseModel):
    status: str  # "found" | "probable" | "not_in_catalog"
    similarity: float | None = None  # SigLIP 2 similarity of the best candidate
    margin: float | None = None  # fusion score gap between the first and the second candidate
    reason: str


class CascadeTimings(BaseModel):
    bbox_detect_ms: float | None = None
    query_prep_ms: float
    v1_total_ms: float
    v4_total_ms: float | None = None
    total_ms: float


class CascadeSearchResponse(BaseModel):
    status: str  # "found" | "probable" | "not_in_catalog"
    message: str
    confidence: float | None = None
    stage_reached: str  # "fusion" (DINOv2 + SigLIP 2) | "v1_only" (SigLIP 2 unavailable)
    winner: CascadeCandidate | None = None  # None when the wine is not in the catalog
    final_results: list[CascadeCandidate]  # best candidates; for not_in_catalog — the most similar catalog labels
    decision: CascadeDecision
    v1_results: list[SearchResult]
    timings: CascadeTimings
    bbox_crop: str | None = None
    v4_query_crop: str | None = None


class CascadePredictResponse(BaseModel):
    slug: str | None = None
    status: str  # "found" | "not_in_catalog" (the scanner's «probable» zone is reported as "found")
    stage_reached: str
    confidence: float | None = None


class CascadeThresholdsResponse(BaseModel):
    """Answer-zone thresholds the running cascade uses (CASCADE_* settings in .env)."""

    found_min_similarity: float  # «найдено»: SigLIP 2 similarity ≥ this ...
    found_min_margin: float  # ... and fusion gap to the second candidate ≥ this
    reject_below_similarity: float  # «нет в каталоге»: SigLIP 2 similarity < this
    predict_threshold: float | None = None  # optional extra rejection threshold
    candidate_pool: int
    v1_weight: float
    stage: str  # "fusion" | "v1_only"
