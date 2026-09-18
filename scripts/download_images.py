#!/usr/bin/env python3
"""Download wine bottle images from the Strapi catalog CSV export.

Reads image URLs from the CSV file and saves them locally with concurrent workers.
Supports idempotency (skips already downloaded files) and limit for testing.
"""

from __future__ import annotations

import argparse
import csv
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DEFAULT_CSV = Path("imports/merged_strapi_vines - merged_strapi_wines.csv")
DEFAULT_OUTPUT_DIR = Path("imports/images")
USER_AGENT = "Mozilla/5.0 (compatible; LCT2026-ImageDownloader/1.0)"


def resolve_image_target(row: dict[str, str], output_dir: Path) -> Path:
    """Determine the destination file path from CSV columns."""
    local_path = (row.get("local_image_path") or "").strip()
    if local_path:
        filename = Path(local_path).name
    else:
        slug = (row.get("Slug") or "").strip()
        if slug:
            filename = f"{slug}.webp"
        else:
            photo_name = (row.get("Название фото") or "").strip()
            filename = photo_name if photo_name else "unnamed.webp"
    return output_dir / filename


def download_single(url: str, target: Path, timeout: int = 15) -> tuple[bool, str]:
    """Download a single image file if it does not already exist."""
    if not url:
        return False, "Empty URL"

    if target.exists() and target.stat().st_size > 0:
        return True, "Already exists"

    target.parent.mkdir(parents=True, exist_ok=True)
    temp_target = target.with_suffix(f"{target.suffix}.tmp")

    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            content = response.read()
            if not content:
                return False, "Downloaded 0 bytes"
            with temp_target.open("wb") as f:
                f.write(content)
            temp_target.replace(target)
            return True, "Downloaded"
    except urllib.error.HTTPError as e:
        if temp_target.exists():
            temp_target.unlink()
        return False, f"HTTP {e.code}"
    except Exception as e:
        if temp_target.exists():
            temp_target.unlink()
        return False, str(e)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download images from wine dataset CSV.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Path to input CSV file")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Destination directory for images")
    parser.add_argument("--limit", type=int, default=0, help="Max images to download (0 = all)")
    parser.add_argument("--workers", type=int, default=8, help="Number of concurrent download threads")
    parser.add_argument("--timeout", type=int, default=15, help="Request timeout in seconds")
    args = parser.parse_args()

    csv_path = args.csv
    if not csv_path.is_file():
        print(f"Error: CSV file not found: {csv_path}", file=sys.stderr)
        return 1

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    total_rows = len(rows)
    print(f"Loaded {total_rows} records from {csv_path}")

    if args.limit > 0:
        rows = rows[: args.limit]
        print(f"Limiting to first {len(rows)} records")

    tasks = []
    for row in rows:
        url = (row.get("image_url") or "").strip()
        target = resolve_image_target(row, out_dir)
        title = (row.get("title") or row.get("Название вина") or "Unknown").strip()
        tasks.append((url, target, title))

    success_count = 0
    fail_count = 0
    skipped_count = 0

    print(f"Starting download with {args.workers} workers into {out_dir}...")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(download_single, url, target, args.timeout): (title, target) for url, target, title in tasks}
        for idx, future in enumerate(as_completed(futures), 1):
            title, target = futures[future]
            try:
                ok, msg = future.result()
                if ok:
                    if msg == "Already exists":
                        skipped_count += 1
                    else:
                        success_count += 1
                else:
                    fail_count += 1
                    print(f"[{idx}/{len(tasks)}] FAILED {title}: {msg} ({target.name})")
            except Exception as e:
                fail_count += 1
                print(f"[{idx}/{len(tasks)}] ERROR {title}: {e}")

            if idx % 100 == 0 or idx == len(tasks):
                print(f"Progress: [{idx}/{len(tasks)}] (new: {success_count}, cached: {skipped_count}, failed: {fail_count})")

    print(f"\nDone! Downloaded: {success_count}, Cached: {skipped_count}, Failed: {fail_count}")
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
