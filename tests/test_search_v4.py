"""Tests for Search Pipeline v4 (SigLIP 2 + OCR Reranker)."""

from __future__ import annotations

import re
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# 1. Schema serialisation
# ---------------------------------------------------------------------------

def test_v4_schema_serialisation():
    from app.schemas.search.v4 import SearchResponseV4, SearchResultV4, SearchTimingsV4

    result = SearchResultV4(
        product_id=uuid.uuid4(),
        slug="test-wine-2020",
        title="Test Wine",
        manufacturer="Test Winery",
        description="A fine test wine 2020",
        image_url="/api/media/labels_v4/abc.webp",
        dino_similarity=0.91,
        vote_count=12,
        vote_ratio=0.24,
        matched_aug_name="perspective",
        ocr_vintage_match=True,
        ocr_score=0.87,
        final_score=0.89,
        rank=1,
    )
    timings = SearchTimingsV4(
        query_prep_ms=5.1,
        embedding_ms=22.3,
        pgvector_ms=8.4,
        ocr_ms=45.7,
        total_ms=81.5,
    )
    response = SearchResponseV4(
        winner=result,
        results=[result],
        timings=timings,
        vintage_detected="2020",
    )
    data = response.model_dump()
    assert data["winner"]["slug"] == "test-wine-2020"
    assert data["timings"]["ocr_ms"] == 45.7
    assert data["vintage_detected"] == "2020"
    assert data["is_fallback"] is False


def test_v4_schema_optional_fields():
    from app.schemas.search.v4 import SearchResponseV4, SearchTimingsV4

    timings = SearchTimingsV4(
        query_prep_ms=3.0, embedding_ms=20.0, pgvector_ms=7.0, total_ms=30.0
    )
    response = SearchResponseV4(winner=None, results=[], timings=timings)
    data = response.model_dump()
    assert data["winner"] is None
    assert data["ocr_ms"] is None
    assert data["vintage_detected"] is None


# ---------------------------------------------------------------------------
# 2. SigLIP2EmbeddingService — raises FileNotFoundError on missing model
# ---------------------------------------------------------------------------

def test_siglip2_embedding_service_missing_model(tmp_path: Path):
    from app.services.siglip_embeddings import SigLIP2EmbeddingService

    with pytest.raises(FileNotFoundError, match="SigLIP2 model not found"):
        SigLIP2EmbeddingService(
            model_path=tmp_path / "nonexistent_model",
            base_model_path=tmp_path / "nonexistent_base",
        )


# ---------------------------------------------------------------------------
# 3. OCR Reranker — vintage extraction regex
# ---------------------------------------------------------------------------

_YEAR_RE = re.compile(r"\b(1[89]\d{2}|20[012]\d)\b")


def _extract_vintage(text: str):
    m = _YEAR_RE.search(text)
    return m.group(0) if m else None


def test_ocr_vintage_extraction_found():
    assert _extract_vintage("Chateau X 2019 Grand Cru Bordeaux") == "2019"


def test_ocr_vintage_extraction_early_year():
    assert _extract_vintage("Vintage 1985 reserve") == "1985"


def test_ocr_vintage_extraction_not_found():
    assert _extract_vintage("No year here at all") is None


def test_ocr_vintage_extraction_ignores_non_wine_years():
    # Years outside 1800–2029 should not match
    assert _extract_vintage("Year 1700 old text") is None
    assert _extract_vintage("Future 2100") is None


# ---------------------------------------------------------------------------
# 4. OcrReranker — graceful behaviour without OCR installed
# ---------------------------------------------------------------------------

def test_ocr_reranker_raises_import_error_without_ocr():
    """OcrReranker.__post_init__ raises ImportError if neither paddleocr nor easyocr is installed."""
    with patch.dict("sys.modules", {"paddleocr": None, "easyocr": None}):
        with pytest.raises(ImportError, match="paddleocr or easyocr"):
            from app.services.ocr_reranker import OcrReranker
            # Force re-import to hit the fresh sys.modules state
            import importlib
            import app.services.ocr_reranker as _m
            importlib.reload(_m)
            _m.OcrReranker()


# ---------------------------------------------------------------------------
# 5. ProductCandidateV4 dataclass
# ---------------------------------------------------------------------------

def test_product_candidate_v4_dataclass():
    from app.db.repositories.products import ProductCandidateV4

    product_mock = MagicMock()
    product_mock.id = uuid.uuid4()

    cand = ProductCandidateV4(
        product=product_mock,
        embedding_id=uuid.uuid4(),
        image_path="labels_v4/abc.webp",
        sample_type="catalog",
        distance=0.12,
        aug_name="catalog",
        aug_seed=0,
        vote_count=5,
    )
    assert cand.distance == 0.12
    assert cand.aug_name == "catalog"
    assert cand.vote_count == 5

    # Frozen — must not allow mutation
    with pytest.raises((AttributeError, TypeError)):
        cand.distance = 0.5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 6. Router v4 — 503 when pipeline not loaded
# ---------------------------------------------------------------------------

def test_router_v4_returns_503_when_pipeline_not_ready():
    from app.main import create_app

    app = create_app()

    # Override lifespan: set pipeline_v4 to None without loading models
    app.state.pipeline_v4 = None

    with TestClient(app, raise_server_exceptions=False) as client:
        import io
        from PIL import Image as PILImage

        buf = io.BytesIO()
        PILImage.new("RGB", (100, 100), color=(128, 0, 0)).save(buf, format="JPEG")
        buf.seek(0)

        response = client.post(
            "/api/v4/search",
            files={"image": ("test.jpg", buf, "image/jpeg")},
            data={"k": "1"},
        )
    assert response.status_code == 503


# ---------------------------------------------------------------------------
# 7. Architecture: v4 routes registered
# ---------------------------------------------------------------------------

def test_v4_routes_registered():
    from app.api.router import router

    paths = {route.path for route in router.routes}
    assert "/api/v4/search" in paths
    assert "/api/v4/search-from-crop" in paths
    assert "/api/v4/eval/predict" in paths
