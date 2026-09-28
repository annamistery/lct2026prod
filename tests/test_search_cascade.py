"""Tests for the cascade: DINOv2 candidates + SigLIP 2 / DINOv2 fusion + answer zones."""

from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from PIL import Image

from app.db.models.product import ProductEmbedding, ProductEmbeddingV4
from app.pipelines.search.cascade.decision import FOUND, NOT_IN_CATALOG, PROBABLE, RecognitionThresholds, classify
from app.pipelines.search.cascade.pipeline import CascadeSearchPipeline

THRESHOLDS = RecognitionThresholds(found_min_similarity=0.83, found_min_margin=0.15, reject_below_similarity=0.485)

# ---------------------------------------------------------------------------
# 1. Answer zones
# ---------------------------------------------------------------------------


def test_zone_not_in_catalog_below_reject_threshold():
    status, reason = classify(0.40, 0.5, THRESHOLDS)
    assert status == NOT_IN_CATALOG
    assert "0.400" in reason


def test_zone_found_needs_similarity_and_margin():
    assert classify(0.90, 0.30, THRESHOLDS)[0] == FOUND
    assert classify(0.90, 0.05, THRESHOLDS)[0] == PROBABLE  # close twin
    assert classify(0.70, 0.90, THRESHOLDS)[0] == PROBABLE  # not similar enough
    assert classify(0.90, None, THRESHOLDS)[0] == FOUND  # single candidate


def test_zone_without_siglip_is_probable():
    assert classify(None, 0.5, THRESHOLDS)[0] == PROBABLE


# ---------------------------------------------------------------------------
# 2. Pipeline with fake models and repository
# ---------------------------------------------------------------------------


def product(slug: str, manufacturer: str = "Винодельня") -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid.uuid4(), slug=slug, title=slug.title(), manufacturer=manufacturer, description="",
        label_image_path=f"products/{slug}/label.webp",
    )


class FakeEmbeddings:
    """Returns one vector per view; the vector only tags the view for the fake repository."""

    def __init__(self, prefix: str):
        self.prefix = prefix

    def embed_batch(self, images, batch_size: int = 32):
        return [[float(i), 1.0] for i, _ in enumerate(images)]


class FakeRepository:
    """similarities[model][view][slug] — per-view best similarity of every product."""

    def __init__(self, products: list[SimpleNamespace], v1_order: list[str], similarities: dict):
        self.products = {p.slug: p for p in products}
        self.v1_order = v1_order
        self.similarities = similarities

    async def nearest(self, embedding, limit):
        return [SimpleNamespace(product=self.products[slug], distance=0.1 * rank) for rank, slug in enumerate(self.v1_order[:limit])]

    async def max_similarity_by_product(self, model, embedding, product_ids):
        name = "v1" if model is ProductEmbedding else "v4"
        assert model in (ProductEmbedding, ProductEmbeddingV4)
        view = int(embedding[0])
        table = self.similarities[name][view]
        return {p.id: table[p.slug] for p in self.products.values() if p.id in product_ids and p.slug in table}


def make_pipeline(with_v4: bool = True, predict_threshold: float | None = None) -> CascadeSearchPipeline:
    detector = MagicMock()
    detector.best_box.return_value = (10, 10, 150, 190)
    query_prep = MagicMock()
    query_prep.prepare_crop.side_effect = lambda crop: SimpleNamespace(image=crop)
    return CascadeSearchPipeline(
        detector=detector,
        v1_embeddings=FakeEmbeddings("v1"),
        v4_embeddings=FakeEmbeddings("v4") if with_v4 else None,
        query_prep=query_prep,
        gpu_semaphore=asyncio.Semaphore(1),
        thresholds=THRESHOLDS,
        candidate_pool=10,
        v1_weight=0.3,
        predict_threshold=predict_threshold,
    )


def twins_repository(v4_rose_sira: float, v4_rose: float) -> FakeRepository:
    """DINOv2 prefers «rose-sira», SigLIP 2 decides; «merlot» is a distant third candidate."""
    items = [product("rose-sira"), product("rose"), product("merlot", "Другая")]
    return FakeRepository(
        items,
        ["rose-sira", "rose", "merlot"],
        {
            "v1": [{"rose-sira": 0.95, "rose": 0.93, "merlot": 0.60}, {"rose-sira": 0.94, "rose": 0.93, "merlot": 0.55}],
            "v4": [{"rose-sira": v4_rose_sira, "rose": v4_rose, "merlot": 0.30}, {"rose-sira": v4_rose_sira - 0.02, "rose": v4_rose - 0.02, "merlot": 0.28}],
        },
    )


IMAGE = Image.new("RGB", (200, 200), color="white")


@pytest.mark.asyncio
async def test_fusion_lets_siglip_overrule_dinov2_and_answers_found():
    response = await make_pipeline().run(IMAGE, twins_repository(v4_rose_sira=0.70, v4_rose=0.95), k=3)
    assert response.stage_reached == "fusion"
    assert response.status == FOUND
    assert response.winner.slug == "rose"
    assert [c.slug for c in response.final_results] == ["rose", "rose-sira", "merlot"]
    assert response.confidence == pytest.approx(0.95)
    assert response.decision.margin > THRESHOLDS.found_min_margin
    assert response.bbox_crop.startswith("data:image/webp;base64,")


@pytest.mark.asyncio
async def test_close_twin_is_probable_not_found():
    response = await make_pipeline().run(IMAGE, twins_repository(v4_rose_sira=0.92, v4_rose=0.91), k=3)
    assert response.status == PROBABLE
    assert response.winner.slug == "rose-sira"
    assert "отрыв" in response.decision.reason


@pytest.mark.asyncio
async def test_low_similarity_means_not_in_catalog():
    pipeline = make_pipeline()
    repository = twins_repository(v4_rose_sira=0.40, v4_rose=0.35)
    response = await pipeline.run(IMAGE, repository, k=3)
    assert response.status == NOT_IN_CATALOG
    assert response.message == "Данного вина нет в каталоге"
    assert response.winner is None
    assert response.final_results  # the most similar catalog labels are still returned
    prediction = await pipeline.predict_top1(IMAGE, repository)
    assert prediction.slug is None
    assert prediction.status == NOT_IN_CATALOG


@pytest.mark.asyncio
async def test_predict_threshold_rejects_weak_answers():
    repository = twins_repository(v4_rose_sira=0.70, v4_rose=0.60)
    assert (await make_pipeline().predict_top1(IMAGE, repository)).slug == "rose-sira"
    rejected = await make_pipeline(predict_threshold=0.8).predict_top1(IMAGE, repository)
    assert rejected.slug is None
    assert rejected.status == NOT_IN_CATALOG


@pytest.mark.asyncio
async def test_without_siglip_dinov2_answers_as_probable():
    response = await make_pipeline(with_v4=False).run(IMAGE, twins_repository(v4_rose_sira=0.9, v4_rose=0.9), k=2)
    assert response.stage_reached == "v1_only"
    assert response.status == PROBABLE
    assert response.winner.slug == "rose-sira"
    assert response.final_results[0].v4_similarity is None


@pytest.mark.asyncio
async def test_empty_catalog_is_not_in_catalog():
    response = await make_pipeline().run(IMAGE, FakeRepository([], [], {"v1": [{}, {}], "v4": [{}, {}]}), k=3)
    assert response.status == NOT_IN_CATALOG
    assert response.winner is None
    assert response.final_results == []


def test_warm_up_runs_every_model_once():
    pipeline = make_pipeline()
    pipeline.warm_up()
    pipeline.detector.best_box.assert_called_once()
    pipeline.query_prep.prepare_crop.assert_called_once()


# ---------------------------------------------------------------------------
# 3. Thresholds: API and configuration
# ---------------------------------------------------------------------------


def test_thresholds_endpoint_reports_running_zones():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.api.router import router

    app = FastAPI()
    app.include_router(router)
    app.state.pipeline_cascade = make_pipeline(predict_threshold=0.5)
    with TestClient(app) as client:
        data = client.get("/api/cascade/thresholds").json()
    assert data == {
        "found_min_similarity": 0.83, "found_min_margin": 0.15, "reject_below_similarity": 0.485,
        "predict_threshold": 0.5, "candidate_pool": 10, "v1_weight": 0.3, "stage": "fusion",
    }


def test_settings_reject_inverted_zones():
    from pydantic import ValidationError

    from app.core.config import Settings

    with pytest.raises(ValidationError, match="must not exceed"):
        Settings(_env_file=None, cascade_found_min_similarity=0.4, cascade_reject_below_similarity=0.5)
