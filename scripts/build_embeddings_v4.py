#!/usr/bin/env python3
"""Build product_embeddings_v4 from the pre-built augmented dataset in media/images/.

Each product directory media/images/{product_uuid}/ already contains 116 WebP files:
  catalog.webp
  {aug_name}_{aug_seed}.webp  (× 115)

This script:
  1. Reads all products from the DB.
  2. For each product finds its directory under media/images/{product.id}/.
  3. Reads every file, extracts aug_name/aug_seed from the filename.
  4. Embeds all 116 images with SigLIP2EmbeddingService (768d).
  5. Writes ProductEmbeddingV4 rows with image_path = 'images/{product_id}/{filename}'.

image_path stores the path to the EXACT augmented file, so the pipeline can
read it directly without needing to reproduce the augmentation from a seed.

Usage:
    python scripts/build_embeddings_v4.py [--limit N] [--batch-size 32] [--replace]
    python scripts/build_embeddings_v4.py --replace  # rebuild all
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import re
import sys
import time
from pathlib import Path

# Ensure writable cache directories for container execution (same as scripts/import_from_csv.py)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("HF_HOME", "/tmp/huggingface")
os.environ.setdefault("TRANSFORMERS_CACHE", "/tmp/huggingface")
os.environ.setdefault("TORCH_HOME", "/tmp/torch")

from PIL import Image
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.core.config import get_settings
from app.db.models.product import Product, ProductEmbeddingV4
from app.services.siglip_embeddings import SigLIP2EmbeddingService

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_embeddings_v4")

_FNAME_RE = re.compile(r"^(.+)_(\d+)\.webp$")


def _parse_aug(filename: str) -> tuple[str, int]:
    """Return (aug_name, aug_seed) from filename like 'perspective_123456.webp' or 'catalog.webp'."""
    if filename == "catalog.webp":
        return "catalog", 0
    m = _FNAME_RE.match(filename)
    if m:
        return m.group(1), int(m.group(2))
    # Fallback: treat whole stem as aug_name, seed=0
    stem = Path(filename).stem
    return stem, 0


async def build_v4_catalog(limit: int | None = None, batch_size: int = 32, replace: bool = True):
    settings = get_settings()
    engine = create_async_engine(settings.database_url, echo=False)
    session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    images_root = settings.media_dir / "images"
    if not images_root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {images_root}")

    logger.info("Initialising SigLIP 2 v4 embedding service...")
    embeddings_service = SigLIP2EmbeddingService(
        settings.resolved_siglip_v4_model_path,
        settings.resolved_siglip_v4_base_model_path,
        settings.embedding_dimension_v4,
        skip_resize=True,
    )

    async with session_factory() as session:
        query = select(Product).order_by(Product.created_at)
        if limit:
            query = query.limit(limit)
        products = list((await session.scalars(query)).all())
        total_products = len(products)
        logger.info("Found %d products in DB", total_products)

        if replace:
            logger.info("Clearing existing product_embeddings_v4...")
            await session.execute(delete(ProductEmbeddingV4))
            await session.commit()

        start_all = time.perf_counter()
        processed_count = 0
        skipped_count = 0
        total_vectors = 0

        for idx, product in enumerate(products, 1):
            prod_dir = images_root / str(product.id)
            if not prod_dir.is_dir() and product.slug:
                prod_dir = images_root / product.slug
            if not prod_dir.is_dir():
                logger.warning("[%d/%d] Skip %s: no directory %s",
                               idx, total_products, product.title[:40], prod_dir)
                skipped_count += 1
                continue

            # Collect all WebP files
            webp_files = sorted(prod_dir.glob("*.webp"))
            if not webp_files:
                logger.warning("[%d/%d] Skip %s: empty directory", idx, total_products, product.title[:40])
                skipped_count += 1
                continue

            prod_start = time.perf_counter()

            # Parse filenames → aug_name, aug_seed, relative path
            aug_meta: list[tuple[str, int, str]] = []  # (aug_name, aug_seed, rel_path)
            images: list[Image.Image] = []
            for fpath in webp_files:
                aug_name, aug_seed = _parse_aug(fpath.name)
                rel_path = f"images/{prod_dir.name}/{fpath.name}"
                aug_meta.append((aug_name, aug_seed, rel_path))
                images.append(Image.open(fpath).convert("RGB"))

            # Batch embed (768d)
            vectors = embeddings_service.embed_batch(images, batch_size=batch_size)

            # Write embeddings to DB
            embeddings_v4 = [
                ProductEmbeddingV4(
                    product_id=product.id,
                    image_path=rel_path,
                    sample_type="catalog" if aug_name == "catalog" else "augmented",
                    aug_name=aug_name,
                    aug_seed=aug_seed,
                    embedding=vector,
                    embedding_model=settings.embedding_model_v4_name,
                )
                for (aug_name, aug_seed, rel_path), vector in zip(aug_meta, vectors)
            ]

            session.add_all(embeddings_v4)
            await session.commit()

            processed_count += 1
            total_vectors += len(embeddings_v4)
            elapsed = time.perf_counter() - prod_start
            logger.info(
                "[%d/%d] %s (%s) — %d vectors in %.2fs",
                idx, total_products, product.title[:40], product.slug or "no-slug",
                len(embeddings_v4), elapsed,
            )

        total_time = time.perf_counter() - start_all
        logger.info(
            "DONE! Processed: %d, Skipped: %d, Total vectors: %d. Time: %.1f min (%.2fs/product)",
            processed_count, skipped_count, total_vectors,
            total_time / 60.0, total_time / max(1, processed_count),
        )

    await engine.dispose()


def main():
    parser = argparse.ArgumentParser(
        description="Build product_embeddings_v4 from media/images/ (SigLIP 2 768d)"
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of products")
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size")
    parser.add_argument(
        "--replace", action=argparse.BooleanOptionalAction, default=True,
        help="Clear existing product_embeddings_v4 first (default: True)",
    )
    args = parser.parse_args()

    asyncio.run(build_v4_catalog(limit=args.limit, batch_size=args.batch_size, replace=args.replace))


if __name__ == "__main__":
    main()
