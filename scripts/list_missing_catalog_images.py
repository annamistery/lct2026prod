#!/usr/bin/env python3
"""List catalog rows whose image files are missing from the import folder.

Produces a CSV with the expected filename so missing images can be downloaded
or renamed before running the cascade ingestion.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def _expected_filename(row: dict[str, str]) -> str:
    """Mirror the filename resolution used by download_images.py / ingest."""
    local_path = (row.get("local_image_path") or "").strip()
    if local_path:
        return Path(local_path).name
    slug = (row.get("Slug") or "").strip()
    if slug:
        return f"{slug}.webp"
    photo_name = (row.get("Название фото") or "").strip()
    return photo_name or ""


def _resolve_image_path(row: dict[str, str], images_dir: Path) -> Path | None:
    """Return the first existing image file for the row, or None."""
    slug = (row.get("Slug") or "").strip()
    photo_name = (row.get("Название фото") or row.get("local_image_path") or "").strip()

    candidates: list[str] = []
    if photo_name:
        candidates.append(Path(photo_name).name)
    if slug:
        candidates.extend([f"{slug}.webp", f"{slug}.jpg", f"{slug}.png"])

    for cand in candidates:
        target = images_dir / cand
        if target.is_file() and target.stat().st_size > 0:
            return target

    if slug:
        matched = list(images_dir.glob(f"{slug}.*"))
        if matched and matched[0].is_file() and matched[0].stat().st_size > 0:
            return matched[0]

    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="List catalog rows missing image files.")
    parser.add_argument("--csv", type=Path, default=Path("imports/wines_integrated_cleared.csv"), help="Path to catalog CSV")
    parser.add_argument("--images-dir", type=Path, default=Path("media/catalog_sources"), help="Directory with downloaded images")
    parser.add_argument("--output", type=Path, default=Path("imports/missing_images.csv"), help="Output CSV path")
    args = parser.parse_args()

    if not args.csv.is_file():
        print(f"CSV file not found: {args.csv}", file=sys.stderr)
        return 1

    args.images_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    with args.csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    missing: list[dict[str, str]] = []
    for row in rows:
        title = (row.get("title") or row.get("Название вина") or "").strip()
        manufacturer = (row.get("manufacturer") or row.get("Винодельня") or "").strip()
        if not title or not manufacturer:
            continue
        if _resolve_image_path(row, args.images_dir) is not None:
            continue
        missing.append(
            {
                "title": title,
                "manufacturer": manufacturer,
                "slug": (row.get("Slug") or "").strip(),
                "photo_name": (row.get("Название фото") or "").strip(),
                "local_image_path": (row.get("local_image_path") or "").strip(),
                "image_url": (row.get("Ссылка на изображение") or row.get("image_url") or "").strip(),
                "expected_filename": _expected_filename(row),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "title",
                "manufacturer",
                "slug",
                "photo_name",
                "local_image_path",
                "image_url",
                "expected_filename",
            ],
        )
        writer.writeheader()
        writer.writerows(missing)

    print(f"Missing images: {len(missing)} / {len(rows)}")
    print(f"Report saved to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
