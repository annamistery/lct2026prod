#!/usr/bin/env python3
"""Unified catalog management script for Cascade (v1 + v4).

Provides complete lifecycle management:
1. clear: Safely truncates products and all embeddings (v1 and v4), clears media.
2. download: Downloads source images from CSV URLs with duplicate skipping and concurrency.
3. import: Reads catalog CSV, checks for existing products/slugs, creates Product rows,
   extracts label crops via YOLO, generates DINOv2 (v1, 384d) and SigLIP 2 (v4, 768d) embeddings,
   and saves them in a unified pipeline transaction.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import os
import shutil
import sys
import time
from pathlib import Path

# Ensure writable cache directories for container execution
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/ultralytics")
os.environ.setdefault("HF_HOME", "/tmp/huggingface")
os.environ.setdefault("TRANSFORMERS_CACHE", "/tmp/huggingface")
os.environ.setdefault("TORCH_HOME", "/tmp/torch")

from PIL import Image  # noqa: E402
from sqlalchemy import select, text  # noqa: E402
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
logger = logging.getLogger("manage_catalog_cascade")


def resolve_image_filename(row: dict[str, str]) -> str:
    local_path = (row.get("local_image_path") or "").strip()
    if local_path:
        return Path(local_path).name
    slug = (row.get("Slug") or "").strip()
    if slug:
        return f"{slug}.webp"
    photo_name = (row.get("Название фото") or "").strip()
    if photo_name:
        return photo_name
    return "unnamed.webp"


async def clear_database(confirm: bool = False) -> None:
    settings = get_settings()
    engine = create_async_engine(settings.database_url, echo=False)
    session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    if not confirm:
        val = input("Confirm clearing all products, v1/v4 embeddings, and media? [y/N]: ")
        if val.strip().lower() not in ("y", "yes"):
            logger.info("Operation aborted by user.")
            return

    logger.info("Truncating database tables...")
    async with session_factory() as session:
        await session.execute(text("TRUNCATE TABLE import_items CASCADE;"))
        await session.execute(text("TRUNCATE TABLE import_jobs CASCADE;"))
        await session.execute(text("TRUNCATE TABLE product_embeddings CASCADE;"))
        if hasattr(Product, "embeddings_v2"):
            await session.execute(text("TRUNCATE TABLE product_embeddings_v2 CASCADE;"))
        if hasattr(Product, "embeddings_v3"):
            await session.execute(text("TRUNCATE TABLE product_embeddings_v3 CASCADE;"))
        await session.execute(text("TRUNCATE TABLE product_embeddings_v4 CASCADE;"))
        await session.execute(text("TRUNCATE TABLE products CASCADE;"))
        await session.commit()
    logger.info("Database tables truncated.")

    products_media = settings.media_dir / "products"
    if products_media.is_dir():
        logger.info("Removing media files from %s...", products_media)
        shutil.rmtree(products_media, ignore_errors=True)
        products_media.mkdir(parents=True, exist_ok=True)

    await engine.dispose()
    logger.info("Clear operation completed successfully.")


def transform_to_high_res(url: str) -> str:
    """Transform low-res preview URL to high-res full bottle URL (1600/1600)."""
    if not url:
        return ""
    import re
    return re.sub(r"/\d+/\d+/resize/", "/1600/1600/resize/", url)


def _download_single_image(url: str, dest: Path, timeout: int = 20) -> tuple[bool, str]:
    import urllib.error
    import urllib.request

    url = transform_to_high_res(url)
    if not url:
        return False, "Empty URL"

    if dest.exists() and dest.stat().st_size > 0:
        return True, "Already exists"

    dest.parent.mkdir(parents=True, exist_ok=True)
    temp_target = dest.with_suffix(f"{dest.suffix}.tmp")
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; LCT2026-ImageDownloader/1.0)"},
    )

    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                content = response.read()
                if not content:
                    return False, "Downloaded 0 bytes"
                with temp_target.open("wb") as f:
                    f.write(content)
                temp_target.replace(dest)
                return True, "Downloaded"
        except urllib.error.HTTPError as e:
            if temp_target.exists():
                temp_target.unlink(missing_ok=True)
            if e.code in (404, 410):
                return False, f"HTTP {e.code}"
        except Exception:
            if temp_target.exists():
                temp_target.unlink(missing_ok=True)
            if attempt == 2:
                return False, "Network timeout"
            time.sleep(1.0)
    return False, "Failed"


def download_images_from_csv(csv_path: Path, output_dir: Path, concurrency: int = 10) -> None:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if not csv_path.is_file():
        logger.error("CSV file not found: %s", csv_path)
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    logger.info("Found %d rows in CSV. Checking images in %s...", len(rows), output_dir)
    tasks: list[tuple[str, Path]] = []
    for r in rows:
        url = (
            r.get("Ссылка на изображение")
            or r.get("Ссылка на фото")
            or r.get("image_url")
            or r.get("url")
            or ""
        ).strip()
        fname = resolve_image_filename(r)
        dest = output_dir / fname
        if not (dest.is_file() and dest.stat().st_size > 0) and url.startswith("http"):
            tasks.append((url, dest))

    if not tasks:
        logger.info("All images already exist in %s (%d files).", output_dir, len(rows))
        return

    logger.info("%d images need to be downloaded with %d worker threads...", len(tasks), concurrency)
    downloaded = 0
    failed = 0
    t0 = time.perf_counter()

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_map = {executor.submit(_download_single_image, u, d): (u, d) for u, d in tasks}
        for idx, fut in enumerate(as_completed(future_map), 1):
            success, reason = fut.result()
            if success:
                downloaded += 1
            else:
                failed += 1
            if idx % 50 == 0 or idx == len(tasks):
                elapsed = time.perf_counter() - t0
                rate = idx / elapsed if elapsed > 0 else 0
                logger.info(
                    "Download progress: %d/%d (Success: %d, Failed: %d, %.1f img/s)",
                    idx,
                    len(tasks),
                    downloaded,
                    failed,
                    rate,
                )

    logger.info("Download completed: %d downloaded, %d failed.", downloaded, failed)


async def import_cascade_catalog(
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
        for row in reader:
            rows.append(row)

    if limit > 0:
        rows = rows[:limit]

    total_rows = len(rows)
    logger.info("Loading DB and checking existing records...")
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

    logger.info("Loading DINOv2 v1 embedding service (384d)...")
    v1_embeddings = await asyncio.to_thread(
        EmbeddingService,
        settings.dino_model_path,
        settings.dino_base_model_path,
        settings.embedding_dimension,
    )

    logger.info("Loading SigLIP 2 v4 embedding service (768d)...")
    v4_embeddings = await asyncio.to_thread(
        SigLIP2EmbeddingService,
        settings.resolved_siglip_v4_model_path,
        settings.resolved_siglip_v4_base_model_path,
        settings.embedding_dimension_v4,
        skip_resize=True,
    )

    query_prep_v4 = QueryPrepV3()

    logger.info("Processing %d catalog items through unified Cascade pipeline...", total_rows)
    success = 0
    skipped = 0
    failed = 0
    t_start = time.perf_counter()

    for idx, row in enumerate(rows, 1):
        slug = (row.get("Slug") or "").strip()
        title = (row.get("title") or row.get("Название вина") or "").strip()
        manufacturer = (row.get("manufacturer") or row.get("Винодельня") or "").strip()
        description = (row.get("description") or row.get("Описание") or "").strip()

        if slug and slug in existing_slugs:
            skipped += 1
            continue
        if not slug and (title.lower(), manufacturer.lower()) in existing_tm:
            skipped += 1
            continue

        filename = resolve_image_filename(row)
        local_img_path = images_dir / filename
        if not local_img_path.is_file():
            found = list(images_dir.glob(f"{Path(filename).stem}.*"))
            if found:
                local_img_path = found[0]
            else:
                logger.warning("[%d/%d] Image missing for '%s' (%s), skipping.", idx, total_rows, title, filename)
                failed += 1
                continue

        try:
            with Image.open(local_img_path) as img:
                source_image = img.convert("RGB")

            # 1. Detect label crop using YOLO
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

            # 2. Compute v1 embedding (DINOv2, 384d)
            v1_vec = await asyncio.to_thread(v1_embeddings.embed_image, crop_img)

            # 3. Compute v4 embedding (SigLIP 2, 768d with Letterbox 518)
            v4_prep = query_prep_v4.prepare_from_crop(crop_img)
            v4_vec = await asyncio.to_thread(v4_embeddings.embed_image, v4_prep)

            # 4. Save media files
            import uuid

            product_id = uuid.uuid4()
            prod_dir = settings.media_dir / "products" / str(product_id)
            prod_dir.mkdir(parents=True, exist_ok=True)

            source_rel = f"products/{product_id}/source.webp"
            label_rel = f"products/{product_id}/label.webp"

            source_image.save(settings.media_dir / source_rel, format="WEBP", quality=90)
            crop_img.save(settings.media_dir / label_rel, format="WEBP", quality=90)

            # 5. Persist Product + v1 vector + v4 vector in single transaction
            async with session_factory() as session:
                product = Product(
                    id=product_id,
                    slug=slug or None,
                    title=title,
                    manufacturer=manufacturer,
                    description=description,
                    source_image_path=source_rel,
                    label_image_path=label_rel,
                )
                session.add(product)

                emb_v1 = ProductEmbedding(
                    product_id=product_id,
                    image_path=label_rel,
                    sample_type="catalog",
                    embedding=v1_vec,
                    embedding_model=settings.embedding_model_name,
                )
                session.add(emb_v1)

                emb_v4 = ProductEmbeddingV4(
                    product_id=product_id,
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
            success += 1

            if idx % 10 == 0 or idx == total_rows:
                elapsed = time.perf_counter() - t_start
                rate = idx / elapsed if elapsed > 0 else 0
                logger.info(
                    "[%d/%d] Ingested: %d, Skipped: %d, Failed: %d (%.1f items/sec)",
                    idx,
                    total_rows,
                    success,
                    skipped,
                    failed,
                    rate,
                )

        except Exception as e:
            logger.error("[%d/%d] Error processing '%s': %s", idx, total_rows, title, e, exc_info=True)
            failed += 1

    await engine.dispose()
    logger.info("Import finished: %d ingested, %d skipped, %d failed.", success, skipped, failed)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Unified Catalog Manager for Cascade (v1+v4)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: clear
    clear_p = subparsers.add_parser("clear", help="Clear all products, embeddings, and media")
    clear_p.add_argument("--yes", action="store_true", help="Skip confirmation prompt")

    # Subcommand: download
    dl_p = subparsers.add_parser("download", help="Download source images from CSV")
    dl_p.add_argument("--csv", type=Path, default=Path("data/wines_integrated.csv"), help="CSV catalog path")
    dl_p.add_argument("--output", type=Path, default=Path("media/catalog_sources"), help="Destination images directory")
    dl_p.add_argument("--concurrency", type=int, default=10, help="Parallel download limit")

    # Subcommand: import
    imp_p = subparsers.add_parser("import", help="Import products and build both v1 & v4 embeddings")
    imp_p.add_argument("--csv", type=Path, default=Path("data/wines_integrated.csv"), help="CSV catalog path")
    imp_p.add_argument("--images-dir", type=Path, default=Path("media/catalog_sources"), help="Images directory")
    imp_p.add_argument("--limit", type=int, default=0, help="Limit number of processed rows (0 = all)")

    args = parser.parse_args()

    if args.command == "clear":
        asyncio.run(clear_database(confirm=args.yes))
        return 0
    elif args.command == "download":
        download_images_from_csv(args.csv, args.output, args.concurrency)
        return 0
    elif args.command == "import":
        return asyncio.run(import_cascade_catalog(args.csv, args.images_dir, args.limit))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
