"""Adding a wine (form or batch import) must build the same references as scripts/ingest_cascade_catalog.py."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

from app.db.models import ProductEmbedding, ProductEmbeddingV2, ProductEmbeddingV4
from app.services.product_ingestion import ProductIngestionService


class FakeEmbeddings:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.sizes: list[tuple[int, int]] = []

    def embed_batch(self, images, batch_size: int = 32):
        self.sizes.extend(image.size for image in images)
        return [[0.0] * self.dimension for _ in images]


def make_service(with_v4: bool = True):
    detector = MagicMock()
    detector.best_box.return_value = (100, 50, 300, 450)
    images = MagicMock()
    images.save_product.return_value = SimpleNamespace(source_path="products/x/source.webp", label_path="products/x/label.webp")
    v1 = FakeEmbeddings(384)
    v4 = FakeEmbeddings(768) if with_v4 else None
    pipeline = SimpleNamespace(embeddings=v1, gpu_semaphore=asyncio.Semaphore(1))
    return ProductIngestionService(images, detector, pipeline, "dinov2-test", v4_embeddings=v4), images, v1, v4


def make_session():
    session = MagicMock()
    session.commit = AsyncMock()
    session.refresh = AsyncMock()
    session.rollback = AsyncMock()
    return session


def stored_rows(session, repository) -> dict[type, list]:
    rows: dict[type, list] = {ProductEmbedding: [], ProductEmbeddingV2: [], ProductEmbeddingV4: []}
    for call in repository.add.call_args_list:
        rows[ProductEmbedding].append(call.args[1])
    for call in repository.add_embedding_v2.call_args_list:
        rows[ProductEmbeddingV2].append(call.args[0])
    for call in session.add_all.call_args_list:
        for row in call.args[0]:
            rows[type(row)].append(row)
    return rows


@pytest.mark.asyncio
async def test_new_wine_gets_cascade_references_for_both_models():
    service, images, v1, v4 = make_service()
    session = make_session()
    repository = MagicMock()
    repository.get_by_slug = AsyncMock(return_value=None)
    with patch("app.services.product_ingestion.ProductRepository", return_value=repository):
        product = await service.create(session, "Вино", "Винодельня", "", Image.new("RGB", (600, 800), "white"), slug="vino-1")

    assert product.slug == "vino-1"
    service.detector.best_box.assert_called_once()
    label = images.save_product.call_args.args[2]
    assert label.size == (518, 518)  # letterbox 518 of the YOLO crop, as in ingest_cascade_catalog.py
    assert set(v1.sizes) == {(518, 518)} and set(v4.sizes) == {(518, 518)}
    rows = stored_rows(session, repository)
    assert len(rows[ProductEmbedding]) == 116 and len(rows[ProductEmbeddingV4]) == 116
    assert sum(row.sample_type == "catalog" for row in rows[ProductEmbeddingV4]) == 1
    session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_without_siglip_only_dinov2_references_are_stored():
    service, _, _, _ = make_service(with_v4=False)
    session = make_session()
    repository = MagicMock()
    repository.get_by_slug = AsyncMock(return_value=None)
    with patch("app.services.product_ingestion.ProductRepository", return_value=repository):
        await service.create(session, "Вино", "Винодельня", "", Image.new("RGB", (600, 800), "white"))
    rows = stored_rows(session, repository)
    assert len(rows[ProductEmbedding]) == 116 and rows[ProductEmbeddingV4] == []


@pytest.mark.asyncio
async def test_duplicate_slug_is_rejected_before_embedding():
    service, images, v1, _ = make_service()
    repository = MagicMock()
    repository.get_by_slug = AsyncMock(return_value=SimpleNamespace(slug="vino-1"))
    with patch("app.services.product_ingestion.ProductRepository", return_value=repository), pytest.raises(ValueError, match="already exists"):
        await service.create(make_session(), "Вино", "Винодельня", "", Image.new("RGB", (600, 800)), slug="vino-1")
    assert v1.sizes == [] and not images.save_product.called
