#!/usr/bin/env python3
"""Fast unified catalog import and embedding builder for Cascade (v1 + v4).

For each wine in the CSV catalog:
1. Verifies if product already exists (by slug or title/winery).
2. Detects label bounding box using YOLO (DetectorService.best_box).
3. Saves original and crop WebP files into media/products/{id}/.
4. Generates DINOv2 embedding (v1, 384d).
5. Generates SigLIP 2 embedding (v4, 768d, Letterbox 518).
6. Persists Product, ProductEmbedding, and ProductEmbeddingV4 atomically.

Speed: ~20-30 items/sec on GPU (~1.5-2 minutes for the entire catalog).
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import os
import sys
import time
import uuid
from pathlib import Path

# Ensure writable cache directories for container execution
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/ultralytics")
os.environ.setdefault("HF_HOME", "/tmp/huggingface")
os.environ.setdefault("TORCH_HOME", "/tmp/torch")

from PIL import Image  # noqa: E402
from sqlalchemy import select  # noqa: E402
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine  # noqa: E402

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.core.config import get_settings  # noqa: E402
from app.db.models.product import Product, ProductEmbedding, ProductEmbeddingV4  # noqa: E402
from app.services.detector import DetectorService  # noqa: E402
from app.services.embeddings import EmbeddingService  # noqa: E402
from app.services.query_prep_v3 import QueryPrepV3  # noqa: E402
from app.services.siglip_embeddings import SigLIP2EmbeddingService  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("fast_import_cascade")


def resolve_image_path(row: dict[str, str], images_dir: Path) -> Path | None:
    slug = (row.get("Slug") or "").strip()
    photo_name = (row.get("Название фото") or row.get("local_image_path") or "").strip()

    candidates: list[str] = []
    if photo_name:
        candidates.append(Path(photo_name).name)
    if slug:
        candidates.append(f"{slug}.webp")
        candidates.append(f"{slug}.jpg")
        candidates.append(f"{slug}.png")

    for cand in candidates:
        target = images_dir / cand
        if target.is_file() and target.stat().st_size > 0:
            return target

    # Case-insensitive or stem match fallback
    if slug:
        matched = list(images_dir.glob(f"{slug}.*"))
        if matched and matched[0].is_file() and matched[0].stat().st_size > 0:
            return matched[0]

    return None


async def run_fast_import(
    csv_path: Path,
    images_dir: Path,
    limit: int = 0,
) -> int:
    settings = get_settings()

    if not csv_path.is_file():
        logger.error("CSV file not found: %s", csv_path)
        return 1

    if not images_dir.is_dir():
        logger.error("Images directory not found: %s", images_dir)
        return 1

    rows: list[dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    if limit > 0:
        rows = rows[:limit]

    total = len(rows)
    logger.info("Connecting to database and reading existing products...")
    engine = create_async_engine(settings.database_url, echo=False)
    session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    async with session_factory() as session:
        res_slugs = await session.scalars(select(Product.slug).where(Product.slug.is_not(None)))
        existing_slugs = set(res_slugs)
        res_tm = await session.execute(select(Product.title, Product.manufacturer))
        existing_tm = {(r[0].strip().lower(), r[1].strip().lower()) for r in res_tm}

    logger.info("Found %d existing products in DB.", len(existing_slugs))

    logger.info("Loading YOLO detector (%s)...", settings.yolo_model_path)
    detector = await asyncio.to_thread(DetectorService, settings.yolo_model_path, settings.yolo_confidence)

    logger.info("Loading DINOv2 v1 (384d)...")
    v1_embeddings = await asyncio.to_thread(
        EmbeddingService,
        settings.dino_model_path,
        settings.dino_base_model_path,
        settings.embedding_dimension,
    )

    logger.info("Loading SigLIP 2 v4 (768d)...")
    v4_embeddings = await asyncio.to_thread(
        SigLIP2EmbeddingService,
        settings.resolved_siglip_v4_model_path,
        settings.resolved_siglip_v4_base_model_path,
        settings.embedding_dimension_v4,
        skip_resize=True,
    )

    query_prep_v4 = QueryPrepV3()

    logger.info("Starting fast ingestion of %d catalog items...", total)
    ingested = 0
    skipped = 0
    missing_img = 0
    failed = 0
    t_start = time.perf_counter()

    for idx, row in enumerate(rows, 1):
        slug = (row.get("Slug") or "").strip()
        title = (row.get("title") or row.get("Название вина") or "").strip()
        manufacturer = (row.get("manufacturer") or row.get("Винодельня") or "").strip()
        description = (row.get("description") or row.get("Описание") or "").strip()

        # Duplicate check
        if slug and slug in existing_slugs:
            skipped += 1
            continue
        if not slug and (title.lower(), manufacturer.lower()) in existing_tm:
            skipped += 1
            continue

        img_file = resolve_image_path(row, images_dir)
        if img_file is None:
            missing_img += 1
            continue

        try:
            with Image.open(img_file) as f_img:
                source_image = f_img.convert("RGB")

            # 1. Label detection via YOLO (best_box is synchronous, run in thread)
            box = await asyncio.to_thread(detector.best_box, source_image)
            if box is not None:
                x1, y1, x2, y2 = box
                w, h = source_image.size
                ix1 = max(0, min(w - 1, int(x1)))
                iy1 = max(0, min(h - 1, int(y1)))
                ix2 = max(0, min(w, int(x2)))
                iy2 = max(0, min(h, int(y2)))
                if ix2 > ix1 and iy2 > iy1:
                    crop_img = source_image.crop((ix1, iy1, ix2, iy2))
                else:
                    crop_img = source_image
            else:
                crop_img = source_image

            # 2. Embedding v1 (DINOv2, 384d)
            v1_vec = await asyncio.to_thread(v1_embeddings.embed_image, crop_img)

            # 3. Embedding v4 (SigLIP 2, 768d, Letterbox 518)
            v4_prep = query_prep_v4.prepare_from_crop(crop_img)
            v4_vec = await asyncio.to_thread(v4_embeddings.embed_image, v4_prep)

            # 4. Save media images to media/products/{product_id}/
            prod_id = uuid.uuid4()
            prod_dir = settings.media_dir / "products" / str(prod_id)
            prod_dir.mkdir(parents=True, exist_ok=True)

            source_rel = f"products/{prod_id}/source.webp"
            label_rel = f"products/{prod_id}/label.webp"

            source_image.save(settings.media_dir / source_rel, format="WEBP", quality=90)
            crop_img.save(settings.media_dir / label_rel, format="WEBP", quality=90)

            # 5. Persist Product + v1 + v4 in one transaction
            async with session_factory() as session:
                product = Product(
                    id=prod_id,
                    slug=slug or None,
                    title=title,
                    manufacturer=manufacturer,
                    description=description,
                    source_image_path=source_rel,
                    label_image_path=label_rel,
                )
                session.add(product)

                emb_v1 = ProductEmbedding(
                    product_id=prod_id,
                    image_path=label_rel,
                    sample_type="catalog",
                    embedding=v1_vec,
                    embedding_model=settings.embedding_model_name,
                )
                session.add(emb_v1)

                emb_v4 = ProductEmbeddingV4(
                    product_id=prod_id,
                    image_path=label_rel,
                    sample_type="catalog",
                    aug_name="catalog",
                    aug_seed=0,
                    embedding=v4_vec,
                    embedding_model="siglip2_v4",
                )
                session.add(emb_v4)

                await session.commit()

            if slug:
                existing_slugs.add(slug)
            existing_tm.add((title.lower(), manufacturer.lower()))
            ingested += 1

            if ingested % 50 == 0 or idx == total:
                elapsed = time.perf_counter() - t_start
                rate = ingested / elapsed if elapsed > 0 else 0
                logger.info(
                    "Progress: [%d/%d] Ingested: %d, Skipped: %d, Missing: %d, Failed: %d (%.1f items/sec)",
                    idx,
                    total,
                    ingested,
                    skipped,
                    missing_img,
                    failed,
                    rate,
                )

        except Exception as e:
            logger.error("[%d/%d] Error processing '%s': %s", idx, total, title, e)
            failed += 1

    await engine.dispose()
    total_time = time.perf_counter() - t_start
    logger.info(
        "ALL DONE! Ingested: %d, Skipped: %d, Missing: %d, Failed: %d. Total time: %.1f seconds (%.2f items/sec)",
        ingested,
        skipped,
        missing_img,
        failed,
        total_time,
        ingested / max(1, total_time),
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Fast Catalog Importer with v1 + v4 Embeddings")
    parser.add_argument("--csv", type=Path, required=True, help="Path to catalog CSV file")
    parser.add_argument("--images-dir", type=Path, required=True, help="Path to images directory")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of items (0 = all)")
    args = parser.parse_args()

    return asyncio.run(run_fast_import(args.csv, args.images_dir, args.limit))


if __name__ == "__main__":
    raise SystemExit(main())
