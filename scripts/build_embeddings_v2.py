#!/usr/bin/env python3
"""Build and populate product_embeddings_v2 from catalog products using Rectification cascade."""

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

from PIL import Image
from sqlalchemy import select
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


async def build_v2_catalog(limit: int | None = None, batch_size: int = 32, replace: bool = True):
    settings = get_settings()
    engine = create_async_engine(settings.database_url, echo=False)
    session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    logger.info("Инициализация моделей для построения базы векторов v2...")
    detector = DetectorService(settings.resolved_yolo_model_path, settings.yolo_confidence)
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
            logger.info("Очистка старых векторов из таблицы product_embeddings_v2...")
            from sqlalchemy import delete
            await session.execute(delete(ProductEmbeddingV2))
            await session.commit()
            logger.info("✓ Таблица product_embeddings_v2 полностью очищена.")

        start_all = time.perf_counter()
        processed_count = 0
        total_vectors = 0

        for idx, product in enumerate(products, 1):
            prod_start = time.perf_counter()

            # 1. Приоритет: берем чистый эталонный кроп этикетки из каталога
            # Никакой сегментации и гомографии на эталонах — они уже чистые и прямоугольные!
            label_img = None
            if product.label_image_path:
                cand = images_service.resolve(product.label_image_path)
                if cand.is_file():
                    try:
                        with Image.open(cand) as raw:
                            label_img = raw.convert("RGB")
                    except Exception:
                        label_img = None

            # Если отдельного кропа нет — детектируем BBox на полном фото каталога
            if label_img is None and product.source_image_path:
                cand = images_service.resolve(product.source_image_path)
                if cand.is_file():
                    try:
                        with Image.open(cand) as raw:
                            full_source = raw.convert("RGB")
                            box = detector.best_box(full_source)
                            if box is not None:
                                label_img = images_service.crop(full_source, box)
                            else:
                                label_img = images_service.canonical(full_source)
                    except Exception:
                        label_img = None

            if label_img is None:
                logger.warning("[%d/%d] Пропуск %s: эталонное изображение не найдено", idx, total_products, product.title)
                continue

            # 2. Канонический чистый прямоугольный мастер 256×256
            clean_master = images_service.canonical(label_img)

            # Сохраняем эталонный мастер для v2
            rect_rel_path = f"rectified/{product.id}.webp"
            rect_abs_path = settings.media_dir / rect_rel_path
            rect_abs_path.parent.mkdir(parents=True, exist_ok=True)
            clean_master.save(rect_abs_path, format="WEBP", quality=95)

            # 3. Облако из 116 аугментаций от ЧИСТОГО эталона
            augmented_items = generate_augmented_cloud(clean_master, variants_per_aug=5)
            aug_images = [img for _, img in augmented_items]

            # 4. Расчет эмбеддингов DINOv2 батчами
            vectors = embeddings_service.embed_batch(aug_images, batch_size=batch_size)

            # 5. Запись в базу данных v2
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
                "[%d/%d] %s (%s) — 116 векторов за %.2fc",
                idx,
                total_products,
                product.title[:40],
                product.slug or "no-slug",
                elapsed,
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
    parser.add_argument("--replace", action=argparse.BooleanOptionalAction, default=True, help="Clear existing product_embeddings_v2 first (default: True)")
    args = parser.parse_args()

    asyncio.run(build_v2_catalog(limit=args.limit, batch_size=args.batch_size, replace=args.replace))


if __name__ == "__main__":
    main()
