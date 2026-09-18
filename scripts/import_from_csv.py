#!/usr/bin/env python3
"""Direct CSV dataset import runner for LCT2026.

Imports products from a CSV catalog file and a local directory of downloaded images.
Can be executed inside the Docker container or locally against the database.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import os
import sys
from pathlib import Path

# Ensure writable cache directories for container execution
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/ultralytics")
os.environ.setdefault("HF_HOME", "/tmp/huggingface")
os.environ.setdefault("TRANSFORMERS_CACHE", "/tmp/huggingface")

from PIL import Image


def resolve_image_filename(row: dict[str, str]) -> str:
    local_path = (row.get("local_image_path") or "").strip()
    if local_path:
        return Path(local_path).name
    slug = (row.get("Slug") or "").strip()
    if slug:
        return f"{slug}.webp"
    photo_name = (row.get("Название фото") or "").strip()
    return photo_name if photo_name else "unnamed.webp"


async def run_import(csv_path: Path, images_dir: Path, limit: int = 0) -> int:
    from app.core.config import get_settings
    from app.db.models import Product
    from app.db.session import create_engine_and_session_factory
    from app.pipelines.search.v1.pipeline import SearchPipelineV1
    from app.pipelines.search.v1.reranking import SiftReranker
    from app.services.detector import DetectorService
    from app.services.embeddings import EmbeddingService
    from app.services.images import ImageService
    from app.services.product_ingestion import ProductIngestionService

    settings = get_settings()

    if not csv_path.is_file():
        print(f"Error: CSV file not found: {csv_path}", file=sys.stderr)
        return 1

    if not images_dir.is_dir():
        print(f"Error: Images directory not found: {images_dir}", file=sys.stderr)
        return 1

    rows: list[dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if limit > 0:
        rows = rows[:limit]

    total = len(rows)
    print(f"Initializing ML pipeline and DB connection...")
    engine, session_factory = create_engine_and_session_factory(settings.database_url)
    images_service = ImageService(settings.media_dir, settings.canonical_size, settings.max_upload_bytes, settings.max_image_pixels)

    detector = await asyncio.to_thread(DetectorService, settings.yolo_model_path, settings.yolo_confidence)
    embeddings = await asyncio.to_thread(EmbeddingService, settings.dino_model_path, settings.dino_base_model_path, settings.embedding_dimension)
    pipeline = SearchPipelineV1(embeddings, images_service, SiftReranker(), asyncio.Semaphore(settings.gpu_concurrency), asyncio.Semaphore(settings.sift_concurrency), settings.candidate_pool_size)
    ingestion_service = ProductIngestionService(images_service, detector, pipeline, settings.embedding_model_name)

    print(f"Starting direct CSV import of {total} items from {csv_path} (images: {images_dir})...")

    success = 0
    failed = 0

    try:
        for idx, row in enumerate(rows, 1):
            title = (row.get("title") or row.get("Название вина") or "").strip()
            manufacturer = (row.get("manufacturer") or row.get("Винодельня") or "").strip()
            description = (row.get("description") or row.get("Описание") or "").strip()
            filename = resolve_image_filename(row)
            image_path = images_dir / filename

            if not image_path.is_file():
                print(f"[{idx}/{total}] SKIP/FAIL {title}: image not found at {image_path}")
                failed += 1
                continue

            try:
                with image_path.open("rb") as img_file:
                    raw_bytes = img_file.read(settings.max_upload_bytes + 1)
                source_img: Image.Image = images_service.decode(raw_bytes)

                async with session_factory() as session:
                    product: Product = await ingestion_service.create(
                        session=session,
                        title=title,
                        manufacturer=manufacturer,
                        description=description,
                        source=source_img,
                    )
                    print(f"[{idx}/{total}] OK: id={product.id} | {product.title} ({product.manufacturer})")
                    success += 1
            except Exception as exc:
                print(f"[{idx}/{total}] ERROR {title}: {exc}")
                failed += 1
    finally:
        await engine.dispose()

    print(f"\nImport finished! Success: {success}, Failed: {failed}")
    return 0 if failed == 0 else 1


def get_default_paths() -> tuple[Path, Path]:
    if Path("/imports").is_dir():
        return (
            Path("/imports/merged_strapi_vines - merged_strapi_wines.csv"),
            Path("/imports/images"),
        )
    return (
        Path("imports/merged_strapi_vines - merged_strapi_wines.csv"),
        Path("imports/images"),
    )


def main() -> int:
    default_csv, default_images = get_default_paths()
    parser = argparse.ArgumentParser(description="Import products directly from CSV and downloaded images.")
    parser.add_argument("--csv", type=Path, default=default_csv, help="Path to catalog CSV")
    parser.add_argument("--images-dir", type=Path, default=default_images, help="Path to downloaded images directory")
    parser.add_argument("--limit", type=int, default=0, help="Number of records to import (0 = all)")
    args = parser.parse_args()

    return asyncio.run(run_import(args.csv, args.images_dir, args.limit))


if __name__ == "__main__":
    raise SystemExit(main())
