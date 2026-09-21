#!/usr/bin/env python3
"""Build and populate product_embeddings_v2 from catalog products using Rectification cascade."""

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

from PIL import Image
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.core.config import get_settings
from app.db.models.product import Product, ProductEmbeddingV2
from app.services.augment import generate_augmented_cloud
from app.services.detector import DetectorService
from app.services.embeddings import EmbeddingService
from app.services.images import ImageService
from app.services.rectification import RectificationService
from app.services.segmenter import SegmenterService

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_embeddings_v2")


async def build_v2_catalog(limit: int | None = None, batch_size: int = 32, replace: bool = False):
    settings = get_settings()

    engine = create_async_engine(settings.database_url, echo=False)
    session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    # 1. Guarantee table product_embeddings_v2 exists in database
    logger.info("Проверка структуры базы данных...")
    async with session_factory() as session:
        await session.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS product_embeddings_v2 (
                    id UUID PRIMARY KEY,
                    product_id UUID NOT NULL REFERENCES products(id) ON DELETE CASCADE,
                    image_path VARCHAR(500) NOT NULL,
                    sample_type VARCHAR(20) NOT NULL CHECK (sample_type IN ('catalog', 'augmented', 'real', 'customer')),
                    embedding vector(384) NOT NULL,
                    embedding_model VARCHAR(200) NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                CREATE INDEX IF NOT EXISTS ix_product_embeddings_v2_product_id ON product_embeddings_v2(product_id);
                CREATE INDEX IF NOT EXISTS ix_product_embeddings_v2_sample_type ON product_embeddings_v2(sample_type);
                INSERT INTO alembic_version (version_num) VALUES ('20260921_0005') ON CONFLICT (version_num) DO NOTHING;
                UPDATE alembic_version SET version_num = '20260921_0005';
                """
            )
        )
        await session.commit()
        logger.info("✓ Таблица product_embeddings_v2 и индексы готовы в PostgreSQL.")

    logger.info("Инициализация моделей для пайплайна v2...")
    detector = DetectorService(settings.resolved_yolo_model_path, settings.yolo_confidence)
    segmenter = None
    seg_path = settings.resolved_yolo_seg_model_path
    if seg_path.is_file():
        segmenter = SegmenterService(seg_path, settings.yolo_seg_confidence)
        logger.info("✓ YOLO сегментатор загружен: %s", seg_path)
    else:
        logger.warning("⚠️ YOLO сегментатор не найден: %s (будет использован BBox fallback)", seg_path)

    rectifier = RectificationService(
        detector=detector,
        segmenter=segmenter,
        target_size=settings.canonical_size,
    )
    embeddings_service = EmbeddingService(settings.dino_model_path, settings.dino_base_model_path, settings.embedding_dimension)
    images_service = ImageService(settings.media_dir, settings.canonical_size, settings.max_upload_bytes, settings.max_image_pixels)

    async with session_factory() as session:
        query = select(Product).order_by(Product.created_at)
        if limit:
            query = query.limit(limit)
        products = list((await session.scalars(query)).all())
        total_products = len(products)
        logger.info("Найдено товаров для генерации v2: %d", total_products)

        if replace:
            logger.info("Очистка старых векторов из product_embeddings_v2...")
            from sqlalchemy import delete
            await session.execute(delete(ProductEmbeddingV2))
            await session.commit()

        start_all = time.perf_counter()
        processed_count = 0
        total_vectors = 0

        for idx, product in enumerate(products, 1):
            prod_start = time.perf_counter()

            # Resolve image path
            img_path = None
            for p in (product.source_image_path, product.label_image_path):
                if p:
                    cand = images_service.resolve(p)
                    if cand.is_file():
                        img_path = cand
                        break

            if img_path is None:
                logger.warning("[%d/%d] Пропуск %s: исходное изображение не найдено", idx, total_products, product.title)
                continue

            try:
                with Image.open(img_path) as raw:
                    full_img = raw.convert("RGB")
            except Exception as e:
                logger.warning("[%d/%d] Ошибка чтения изображения %s: %s", idx, total_products, img_path, e)
                continue

            # 1. Cascade Rectification
            rect_res = rectifier.rectify(full_img)

            # 2. Save rectified master matrix to media
            rect_rel_path = f"rectified/{product.id}.webp"
            rect_abs_path = settings.media_dir / rect_rel_path
            rect_abs_path.parent.mkdir(parents=True, exist_ok=True)
            rect_res.rectified_image.save(rect_abs_path, format="WEBP", quality=95)

            # 3. Generate 116 augmentation cloud from rectified matrix
            augmented_items = generate_augmented_cloud(rect_res.rectified_image, variants_per_aug=5)
            aug_images = [img for _, img in augmented_items]

            # 4. Extract embeddings
            vectors = embeddings_service.embed_batch(aug_images, batch_size=batch_size)

            # 5. Insert rows into product_embeddings_v2
            embeddings_v2 = [
                ProductEmbeddingV2(
                    product_id=product.id,
                    image_path=rect_rel_path,
                    sample_type="catalog" if aug_name == "catalog" else "augmented",
                    embedding=vector,
                    embedding_model=settings.embedding_model_v2_name,
                )
                for (aug_name, _), vector in zip(augmented_items, vectors, strict=True)
            ]

            session.add_all(embeddings_v2)
            await session.commit()

            processed_count += 1
            total_vectors += len(embeddings_v2)
            elapsed = time.perf_counter() - prod_start
            logger.info(
                "[%d/%d] %s (%s) — 116 векторов за %.2fc (fallback=%s)",
                idx,
                total_products,
                product.title[:40],
                product.slug or "no-slug",
                elapsed,
                rect_res.is_fallback,
            )

        total_time = time.perf_counter() - start_all
        logger.info(
            "ГОТОВО! Обработано товаров: %d, записано векторов v2: %d. Общее время: %.1f мин (в среднем %.2fc/товар)",
            processed_count,
            total_vectors,
            total_time / 60.0,
            total_time / max(1, processed_count),
        )

    await engine.dispose()


def main():
    parser = argparse.ArgumentParser(description="Build product_embeddings_v2 catalog embeddings")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of products to process")
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size")
    parser.add_argument("--replace", action="store_true", help="Delete existing product_embeddings_v2 first")
    args = parser.parse_args()

    asyncio.run(build_v2_catalog(limit=args.limit, batch_size=args.batch_size, replace=args.replace))


if __name__ == "__main__":
    main()
