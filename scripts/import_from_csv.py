#!/usr/bin/env python3
"""Direct CSV dataset import runner for LCT2026.

Imports products from a CSV catalog file and a local directory of downloaded images.
Can be executed inside the Docker container or locally against the database.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import sys
import uuid
from pathlib import Path

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
    from app.core.dependencies import get_images, get_ingestion, get_session_factory
    from app.db.models import Product

    settings = get_settings()
    session_factory = get_session_factory(settings)
    images_service = get_images(settings)
    ingestion_service = get_ingestion(settings)

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
    print(f"Starting direct CSV import of {total} items from {csv_path} (images: {images_dir})...")

    success = 0
    failed = 0

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
