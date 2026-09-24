"""Tests for Cascade Search Pipeline (v1 Coarse + v4 Refinement)."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from PIL import Image

from app.pipelines.search.cascade.decision import CascadeDecisionEngine
from app.pipelines.search.cascade.pipeline import CascadeSearchPipeline
from app.schemas.search.cascade import (
    CascadePredictResponse,
    CascadeSearchResponse,
    CascadeTimings,
    V1DecisionDetails,
)
from app.schemas.search.v1 import SearchResponse, SearchResult, SearchTimings
from app.schemas.search.v4 import SearchResponseV4, SearchResultV4, SearchTimingsV4

# ---------------------------------------------------------------------------
# 1. Schema serialisation tests
# ---------------------------------------------------------------------------

def test_cascade_schemas():
    pid1 = uuid.uuid4()
    v1_item = SearchResult(
        product_id=pid1,
        slug="wine-alpha-2018",
        title="Wine Alpha 2018",
        manufacturer="Alpha Winery",
        description="Fine wine",
        image_url="/api/media/alpha.webp",
        dino_similarity=0.88,
        pgvector_rank=1,
        final_rank=1,
    )
    decision = V1DecisionDetails(
        is_confident=True,
        top1_similarity=0.88,
        top2_similarity=0.75,
        margin=0.13,
        reason="Ярко выраженный ТОП-1",
    )
    timings = CascadeTimings(
        bbox_detect_ms=12.5,
        v1_total_ms=18.0,
        decision_ms=0.5,
        v4_total_ms=None,
        total_ms=31.0,
    )
    resp = CascadeSearchResponse(
        stage_reached="v1_confident",
        winner=v1_item,
        final_results=[v1_item],
        decision=decision,
        v1_results=[v1_item],
        v1_neighbors=[],
        v4_results=None,
        timings=timings,
    )
    data = resp.model_dump()
    assert data["stage_reached"] == "v1_confident"
    assert data["winner"]["slug"] == "wine-alpha-2018"
    assert data["decision"]["is_confident"] is True
    assert data["timings"]["v4_total_ms"] is None


def test_cascade_predict_schema():
    pred = CascadePredictResponse(
        slug="wine-beta-2021",
        stage_reached="v4_refined",
        confidence=0.9234,
    )
    data = pred.model_dump()
    assert data["slug"] == "wine-beta-2021"
    assert data["stage_reached"] == "v4_refined"
    assert data["confidence"] == 0.9234


# ---------------------------------------------------------------------------
# 2. Decision engine unit tests
# ---------------------------------------------------------------------------

def test_decision_engine_unambiguous_winner():
    engine = CascadeDecisionEngine(confidence_margin=0.05, min_confidence_score=0.65)
    items = [
        SearchResult(
            product_id=uuid.uuid4(),
            slug="wine-a",
            title="Wine A",
            manufacturer="Winery X",
            description="",
            image_url="",
            dino_similarity=0.85,
            pgvector_rank=1,
            final_rank=1,
        ),
        SearchResult(
            product_id=uuid.uuid4(),
            slug="wine-b",
            title="Wine B",
            manufacturer="Winery Y",
            description="",
            image_url="",
            dino_similarity=0.74,
            pgvector_rank=2,
            final_rank=2,
        ),
    ]
    decision, neighbors = engine.evaluate(items)
    assert decision.is_confident is True
    assert decision.margin == 0.11
    assert len(neighbors) == 0
    assert "Ярко выраженный ТОП-1" in decision.reason


def test_decision_engine_close_neighbors():
    engine = CascadeDecisionEngine(confidence_margin=0.05)
    items = [
        SearchResult(
            product_id=uuid.uuid4(),
            slug="wine-a",
            title="Wine A",
            manufacturer="Winery X",
            description="",
            image_url="",
            dino_similarity=0.83,
            pgvector_rank=1,
            final_rank=1,
        ),
        SearchResult(
            product_id=uuid.uuid4(),
            slug="wine-b",
            title="Wine B",
            manufacturer="Winery Y",
            description="",
            image_url="",
            dino_similarity=0.81,
            pgvector_rank=2,
            final_rank=2,
        ),
    ]
    decision, neighbors = engine.evaluate(items)
    assert decision.is_confident is False
    assert decision.margin == 0.02
    assert len(neighbors) >= 2
    assert neighbors[0].product_id == items[0].product_id
    assert neighbors[1].product_id == items[1].product_id


def test_decision_engine_same_winery_conflict():
    engine = CascadeDecisionEngine(confidence_margin=0.05)
    items = [
        SearchResult(
            product_id=uuid.uuid4(),
            slug="chateau-rouge-2019",
            title="Chateau Rouge 2019",
            manufacturer="Chateau Tamagne",
            description="",
            image_url="",
            dino_similarity=0.89,
            pgvector_rank=1,
            final_rank=1,
        ),
        SearchResult(
            product_id=uuid.uuid4(),
            slug="chateau-blanc-2020",
            title="Chateau Blanc 2020",
            manufacturer="Chateau Tamagne",
            description="",
            image_url="",
            dino_similarity=0.82,
            pgvector_rank=2,
            final_rank=2,
        ),
    ]
    decision, neighbors = engine.evaluate(items)
    # Even though margin is 0.07 (which is >= 0.05), same manufacturer triggers neighbor refinement
    assert decision.is_confident is False
    assert len(neighbors) >= 2
    assert "одного производителя" in decision.reason


def test_decision_engine_single_or_empty():
    engine = CascadeDecisionEngine()
    empty_dec, empty_n = engine.evaluate([])
    assert empty_dec.is_confident is False
    assert len(empty_n) == 0

    single_item = [
        SearchResult(
            product_id=uuid.uuid4(),
            slug="solo",
            title="Solo Wine",
            manufacturer="Solo",
            description="",
            image_url="",
            dino_similarity=0.90,
            pgvector_rank=1,
            final_rank=1,
        )
    ]
    single_dec, single_n = engine.evaluate(single_item)
    assert single_dec.is_confident is True
    assert len(single_n) == 0


# ---------------------------------------------------------------------------
# 3. Pipeline integration with mock stages
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cascade_pipeline_branch_v1_confident():
    detector = MagicMock()
    detector.best_box.return_value = (10, 10, 100, 100)

    images = MagicMock()
    mock_crop = Image.new("RGB", (90, 90), color="red")
    images.crop.return_value = mock_crop

    p1 = MagicMock()
    p1_item = SearchResult(
        product_id=uuid.uuid4(),
        slug="confident-wine",
        title="Confident Wine",
        manufacturer="Alpha",
        description="",
        image_url="/api/media/a.webp",
        dino_similarity=0.95,
        pgvector_rank=1,
        final_rank=1,
    )
    p1_second = SearchResult(
        product_id=uuid.uuid4(),
        slug="other-wine",
        title="Other Wine",
        manufacturer="Beta",
        description="",
        image_url="/api/media/b.webp",
        dino_similarity=0.70,
        pgvector_rank=2,
        final_rank=2,
    )
    p1.run = AsyncMock(
        return_value=SearchResponse(
            winner=p1_item,
            results=[p1_item, p1_second],
            timings=SearchTimings(embedding_ms=10.0, pgvector_ms=5.0, total_ms=15.0),
        )
    )

    p4 = MagicMock()
    p4.run = AsyncMock()

    cascade = CascadeSearchPipeline(
        detector=detector,
        pipeline_v1=p1,
        pipeline_v4=p4,
        decision_engine=CascadeDecisionEngine(confidence_margin=0.05),
        images=images,
    )

    full_image = Image.new("RGB", (200, 200), color="white")
    repo = MagicMock()

    resp = await cascade.run(full_image, repo, k=5, is_already_crop=False)

    assert resp.stage_reached == "v1_confident"
    assert resp.winner.slug == "confident-wine"
    assert resp.v4_results is None
    p4.run.assert_not_called()  # v4 was correctly skipped


@pytest.mark.asyncio
async def test_cascade_pipeline_branch_v4_refined():
    detector = MagicMock()
    detector.best_box.return_value = (10, 10, 100, 100)

    images = MagicMock()
    mock_crop = Image.new("RGB", (90, 90), color="red")
    images.crop.return_value = mock_crop

    p1 = MagicMock()
    pid1, pid2 = uuid.uuid4(), uuid.uuid4()
    p1_first = SearchResult(
        product_id=pid1,
        slug="twin-one-2018",
        title="Twin One 2018",
        manufacturer="Twin Winery",
        description="",
        image_url="/api/media/1.webp",
        dino_similarity=0.85,
        pgvector_rank=1,
        final_rank=1,
    )
    p1_second = SearchResult(
        product_id=pid2,
        slug="twin-two-2019",
        title="Twin Two 2019",
        manufacturer="Twin Winery",
        description="",
        image_url="/api/media/2.webp",
        dino_similarity=0.84,
        pgvector_rank=2,
        final_rank=2,
    )
    p1.run = AsyncMock(
        return_value=SearchResponse(
            winner=p1_first,
            results=[p1_first, p1_second],
            timings=SearchTimings(embedding_ms=10.0, pgvector_ms=5.0, total_ms=15.0),
        )
    )

    p4_winner = SearchResultV4(
        product_id=pid2,  # v4 selected the second candidate as winner via OCR/SigLIP
        slug="twin-two-2019",
        title="Twin Two 2019",
        manufacturer="Twin Winery",
        description="",
        image_url="/api/media/2.webp",
        dino_similarity=0.84,
        vote_count=10,
        vote_ratio=0.5,
        matched_aug_name="original",
        ocr_vintage_match=True,
        ocr_score=0.92,
        final_score=0.95,
        rank=1,
    )
    p4 = MagicMock()
    p4.run = AsyncMock(
        return_value=SearchResponseV4(
            winner=p4_winner,
            results=[p4_winner],
            timings=SearchTimingsV4(query_prep_ms=5.0, embedding_ms=15.0, pgvector_ms=4.0, total_ms=24.0),
            vintage_detected="2019",
        )
    )

    cascade = CascadeSearchPipeline(
        detector=detector,
        pipeline_v1=p1,
        pipeline_v4=p4,
        decision_engine=CascadeDecisionEngine(confidence_margin=0.05),
        images=images,
    )

    full_image = Image.new("RGB", (200, 200), color="white")
    repo = MagicMock()

    resp = await cascade.run(full_image, repo, k=5, is_already_crop=False)

    assert resp.stage_reached == "v4_refined"
    assert resp.winner.slug == "twin-two-2019"
    assert resp.vintage_detected == "2019"
    p4.run.assert_called_once()
    # Check that product_ids scope was passed strictly matching the neighbors
    call_kwargs = p4.run.call_args.kwargs
    assert call_kwargs["product_ids"] == [pid1, pid2]
