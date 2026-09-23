#!/usr/bin/env python3
"""Build and populate product_embeddings_v3 from catalog products using DINOv2-base 768d.

Generates 116 augmentations per product at 518×518 letterbox with zero-fill,
computes 768d embeddings via the fine-tuned LoRA model, and writes them to
product_embeddings_v3 with aug_name and aug_seed for SIFT rerank regeneration.

Usage:
    python scripts/build_embeddings_v3.py [--limit N] [--batch-size 32] [--replace]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

from PIL import Image
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.core.config import get_settings
from app.db.models.product import Product, ProductEmbeddingV3
from app.services.augment import generate_augmented_cloud_v3
from app.services.detector import DetectorService
from app.services.embeddings import EmbeddingService
from app.services.images import ImageService
from app.services.query_prep_v3 import letterbox_pil

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_embeddings_v3")

CANONICAL_SIZE = 518


async def build_v3_catalog(limit: int | None = None, batch_size: int = 32, replace: bool = True):
    settings = get_settings()
    engine = create_async_engine(settings.database_url, echo=False)
    session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    logger.info("Initializing DINOv2-base v3 embedding service...")
    embeddings_service = EmbeddingService(
        settings.dino_v3_model_path, settings.dino_v3_base_model_path, settings.embedding_dimension_v3,
        skip_resize=True,
    )
    images_service = ImageService(
        settings.media_dir, settings.canonical_size, settings.max_upload_bytes, settings.max_image_pixels,
    )
    detector = DetectorService(settings.yolo_model_path, settings.yolo_confidence)
    logger.info("YOLO detector loaded for v3 crop extraction")

    async with session_factory() as session:
        query = select(Product).order_by(Product.created_at)
        if limit:
            query = query.limit(limit)
        products = list((await session.scalars(query)).all())
        total_products = len(products)
        logger.info("Found %d products for v3 embedding generation", total_products)

        if replace:
            logger.info("Clearing existing product_embeddings_v3...")
            await session.execute(delete(ProductEmbeddingV3))
            await session.commit()
            logger.info("product_embeddings_v3 cleared.")

        start_all = time.perf_counter()
        processed_count = 0
        total_vectors = 0

        for idx, product in enumerate(products, 1):
            prod_start = time.perf_counter()

            # 1. Load source image and re-detect label with YOLO
            source_img = None
            if product.source_image_path:
                src_path = images_service.resolve(product.source_image_path)
                if src_path.is_file():
                    try:
                        with Image.open(src_path) as raw:
                            source_img = raw.convert("RGB")
                    except Exception:
                        source_img = None

            if source_img is None:
                logger.warning("[%d/%d] Skip %s: no source image", idx, total_products, product.title[:40])
                continue

            # 2. YOLO detect → crop with natural proportions (no squash to square)
            box = detector.best_box(source_img)
            if box is not None:
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(source_img.width, x2 + 1)
                y2 = min(source_img.height, y2 + 1)
                label_crop = source_img.crop((x1, y1, x2, y2))
            else:
                label_crop = source_img

            # 3. Letterbox 518×518 (preserve proportions, zero-fill)
            canonical = letterbox_pil(label_crop, CANONICAL_SIZE)

            # 3. Generate v3 augmented cloud (116 images with reproducible seeds)
            aug_items = generate_augmented_cloud_v3(canonical, variants_per_aug=5, seed=42)
            aug_images = [img for _, _, img in aug_items]

            # 4. Batch embed (768d)
            vectors = embeddings_service.embed_batch(aug_images, batch_size=batch_size)

            # 5. Save label master for v3
            label_rel_path = f"labels_v3/{product.id}.webp"
            label_abs_path = settings.media_dir / label_rel_path
            label_abs_path.parent.mkdir(parents=True, exist_ok=True)
            canonical.save(label_abs_path, format="WEBP", quality=95)

            # 6. Write embeddings to DB
            embeddings_v3 = [
                ProductEmbeddingV3(
                    product_id=product.id,
                    image_path=label_rel_path,
                    sample_type="catalog" if aug_name == "catalog" else "augmented",
                    aug_name=aug_name,
                    aug_seed=aug_seed,
                    embedding=vector,
                    embedding_model=settings.embedding_model_v3_name,
                )
                for (aug_name, aug_seed, _), vector in zip(aug_items, vectors, strict=True)
            ]

            session.add_all(embeddings_v3)
            await session.commit()

            processed_count += 1
            total_vectors += len(embeddings_v3)
            elapsed = time.perf_counter() - prod_start
            logger.info(
                "[%d/%d] %s (%s) — %d vectors in %.2fs",
                idx, total_products, product.title[:40], product.slug or "no-slug",
                len(embeddings_v3), elapsed,
            )

        total_time = time.perf_counter() - start_all
        logger.info(
            "DONE! Processed: %d products, %d vectors. Time: %.1f min (%.2fs/product)",
            processed_count, total_vectors, total_time / 60.0, total_time / max(1, processed_count),
        )

    await engine.dispose()


def main():
    parser = argparse.ArgumentParser(description="Build product_embeddings_v3 (DINOv2-base 768d)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of products")
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size")
    parser.add_argument("--replace", action=argparse.BooleanOptionalAction, default=True,
                        help="Clear existing product_embeddings_v3 first (default: True)")
    args = parser.parse_args()

    asyncio.run(build_v3_catalog(limit=args.limit, batch_size=args.batch_size, replace=args.replace))


if __name__ == "__main__":
    main()
