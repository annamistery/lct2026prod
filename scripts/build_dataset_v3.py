#!/usr/bin/env python3
"""Build the v3 training dataset: letterbox 518×518 + 116 augmented images per product.

Reads all products from the database, applies letterbox 518 (preserving proportions,
zero-fill), generates 116 augmentations via v3 augmentation cloud, and writes
a v3_manifest.json for training.

Usage:
    python scripts/build_dataset_v3.py [--limit N] [--workers 8] [--clean]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import multiprocessing
import os
import shutil
import sys
import time
from pathlib import Path

from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.core.config import get_settings
from app.services.augment import generate_augmented_cloud_v3
from app.services.query_prep_v3 import letterbox_pil

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_dataset_v3")

CANONICAL_SIZE = 518

# Resolved at runtime from settings.media_dir (writable mount inside Docker)
_DATASET_DIR: Path | None = None
_MANIFEST_FILE: Path | None = None


def _init_paths() -> tuple[Path, Path]:
    global _DATASET_DIR, _MANIFEST_FILE
    if _DATASET_DIR is None:
        settings = get_settings()
        _DATASET_DIR = settings.media_dir / "dataset_v3"
        _MANIFEST_FILE = _DATASET_DIR / "v3_manifest.json"
    return _DATASET_DIR, _MANIFEST_FILE


def _process_product(args: tuple) -> list[dict]:
    """Worker: generate 116 augmented images for one product.

    args: (idx, product_id, source_path_str, bbox_or_none, seed, dataset_dir_str)
    bbox_or_none: (x1, y1, x2, y2) ints or None (use full image)
    """
    idx, product_id, source_path_str, bbox, seed, dataset_dir_str = args
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    import cv2
    cv2.setNumThreads(1)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    source_path = Path(source_path_str)
    if not source_path.is_file():
        return []

    try:
        source_img = Image.open(source_path).convert("RGB")
    except Exception as e:
        print(f"[skip] {product_id}: {e}")
        return []

    # Crop with natural proportions (no squash to square)
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(source_img.width, x2)
        y2 = min(source_img.height, y2)
        label_crop = source_img.crop((x1, y1, x2, y2))
    else:
        label_crop = source_img

    # Letterbox 518×518 preserving proportions with zero-fill
    canonical = letterbox_pil(label_crop, CANONICAL_SIZE)

    # Product directory (inside the writable media mount)
    dataset_dir = Path(dataset_dir_str)
    prod_dir = dataset_dir / "images" / product_id
    prod_dir.mkdir(parents=True, exist_ok=True)

    # Generate v3 augmented cloud
    aug_items = generate_augmented_cloud_v3(canonical, variants_per_aug=5, seed=seed)

    records: list[dict] = []
    for aug_name, variant_seed, aug_img in aug_items:
        if aug_name == "catalog":
            fname = "catalog.webp"
        else:
            fname = f"{aug_name}_{variant_seed}.webp"
        out_path = prod_dir / fname
        aug_img.save(out_path, format="WEBP", quality=95)

        # Store path relative to dataset_dir for the manifest
        rel_path = str(out_path.relative_to(dataset_dir)).replace("\\", "/")
        records.append({
            "image_id": f"{product_id}_{aug_name}_{variant_seed}",
            "group_id": product_id,
            "aug_name": aug_name,
            "aug_seed": variant_seed,
            "image_path": f"dataset_v3/{rel_path}",
        })

    if idx % 50 == 0:
        print(f"[{idx}] {product_id}: {len(records)} images")

    return records


async def build_dataset(limit: int | None = None, workers: int | None = None, clean: bool = False):
    settings = get_settings()
    dataset_dir = settings.media_dir / "dataset_v3"
    manifest_file = dataset_dir / "v3_manifest.json"

    if clean and dataset_dir.exists():
        logger.info("Cleaning old dataset_v3...")
        shutil.rmtree(dataset_dir / "images", ignore_errors=True)
        manifest_file.unlink(missing_ok=True)

    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "images").mkdir(exist_ok=True)

    # Load products from DB
    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
    from app.db.models.product import Product

    engine = create_async_engine(settings.database_url, echo=False)
    session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    async with session_factory() as session:
        query = select(Product).order_by(Product.created_at)
        if limit:
            query = query.limit(limit)
        products = list((await session.scalars(query)).all())
    await engine.dispose()

    logger.info("Found %d products for v3 dataset", len(products))

    # Prepare tasks: detect label bboxes from source images
    from app.services.images import ImageService
    from app.services.detector import DetectorService

    img_svc = ImageService(settings.media_dir, settings.canonical_size, settings.max_upload_bytes, settings.max_image_pixels)
    detector = DetectorService(settings.yolo_model_path, settings.yolo_confidence)
    logger.info("YOLO detector loaded — detecting bboxes for %d products...", len(products))

    tasks: list[tuple] = []
    detect_t0 = time.perf_counter()
    for idx, product in enumerate(products):
        source_path = img_svc.resolve(product.source_image_path) if product.source_image_path else None
        if source_path is None or not source_path.is_file():
            logger.warning("Skip %s: no source image", product.title[:40])
            continue

        # YOLO detect on source image to get bbox with natural proportions
        try:
            with Image.open(source_path) as src:
                src_rgb = src.convert("RGB")
                box = detector.best_box(src_rgb)
        except Exception as e:
            logger.warning("Skip %s: %s", product.title[:40], e)
            continue

        bbox = None
        if box is not None:
            bbox = (int(box[0]), int(box[1]), int(box[2]) + 1, int(box[3]) + 1)

        product_id = str(product.id)
        tasks.append((idx, product_id, str(source_path), bbox, 42 + idx, str(dataset_dir)))

    detect_elapsed = time.perf_counter() - detect_t0
    logger.info("YOLO detection done in %.1fs — %d products ready", detect_elapsed, len(tasks))

    if workers is None:
        workers = max(1, os.cpu_count() or 1)

    logger.info("Generating v3 dataset with %d workers for %d products...", workers, len(tasks))
    t0 = time.perf_counter()

    manifest: list[dict] = []
    if workers == 1:
        for task in tasks:
            manifest.extend(_process_product(task))
    else:
        with multiprocessing.Pool(processes=workers, maxtasksperchild=5) as pool:
            for records in pool.imap(_process_product, tasks, chunksize=1):
                manifest.extend(records)

    with manifest_file.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    elapsed = time.perf_counter() - t0
    logger.info(
        "DONE! %d images for %d products in %.1f min (%.2f s/product). Manifest: %s",
        len(manifest), len(tasks), elapsed / 60, elapsed / max(1, len(tasks)), manifest_file,
    )


def main():
    parser = argparse.ArgumentParser(description="Build v3 training dataset (letterbox 518 + 116 augmentations)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of products")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers (default: CPU count)")
    parser.add_argument("--clean", action="store_true", help="Remove old dataset_v3 before building")
    args = parser.parse_args()

    asyncio.run(build_dataset(limit=args.limit, workers=args.workers, clean=args.clean))


if __name__ == "__main__":
    main()
