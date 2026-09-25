#!/usr/bin/env python3
"""Unified high-performance catalog ingestion and embedding pipeline for Cascade (v1 + v4).

Key architectural features:
1. Single YOLO detection + single Letterbox 518x518 crop per product.
2. Generates 116-augmentation cloud once in memory (no intermediate disk writes).
3. Computes DINOv2 (384d, v1) and SigLIP 2 (768d, v4) embeddings in GPU batches.
4. Full idempotency and integrity check:
   - Skips products that already have full 116-vector clouds in both tables.
   - Automatically repairs broken/incomplete products (cleans broken vectors and re-embeds).
   - Ingests new products atomically with both v1 and v4 embeddings.

Intended to run inside the API container:
    docker compose exec api python scripts/ingest_cascade_catalog.py \
        --csv /imports/wines_integrated_cleared.csv \
        --images-dir /media/catalog_sources
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
import uuid
from pathlib import Path

# Ensure container-safe writable directories
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/ultralytics")
os.environ.setdefault("HF_HOME", "/tmp/huggingface")
os.environ.setdefault("TORCH_HOME", "/tmp/torch")

from PIL import Image  # noqa: E402
from sqlalchemy import delete, func, select  # noqa: E402
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine  # noqa: E402

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.core.config import get_settings  # noqa: E402
from app.db.models.product import Product, ProductEmbedding, ProductEmbeddingV4  # noqa: E402
from app.services.augment import generate_augmented_cloud_v3  # noqa: E402
from app.services.detector import DetectorService  # noqa: E402
from app.services.embeddings import EmbeddingService  # noqa: E402
from app.services.query_prep_v3 import letterbox_pil  # noqa: E402
from app.services.siglip_embeddings import SigLIP2EmbeddingService  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ingest_cascade_catalog")


def get_default_paths() -> tuple[Path, Path]:
    """Return default (csv, images_dir) paths suitable for Docker container first, local fallback."""
    if Path("/imports").is_dir():
        csv_path = Path("/imports/wines_integrated_cleared.csv")
    else:
        csv_path = Path("data/wines_integrated_cleared.csv")

    if Path("/media/catalog_sources").is_dir():
        images_dir = Path("/media/catalog_sources")
    else:
        images_dir = Path("media/catalog_sources")

    return csv_path, images_dir


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


def _ensure_media_dir(settings) -> None:
    settings.media_dir.mkdir(parents=True, exist_ok=True)


def _save_product_media(product_id: uuid.UUID, source: Image.Image, label: Image.Image, settings) -> tuple[str, str]:
    """Persist source.webp and label.webp under media/products/<product_id>/.

    Uses atomic writes: save to temp, then os.replace.
    If directory already exists (repair/force-rebuild), it is removed first.
    """
    prod_dir = settings.media_dir / "products" / str(product_id)
    if prod_dir.is_dir():
        shutil.rmtree(prod_dir)
    prod_dir.mkdir(parents=True, exist_ok=False)

    source_rel = f"products/{product_id}/source.webp"
    label_rel = f"products/{product_id}/label.webp"
    source_path = settings.media_dir / source_rel
    label_path = settings.media_dir / label_rel

    _atomic_save(source, source_path)
    _atomic_save(label, label_path)

    return source_rel, label_rel


def _atomic_save(image: Image.Image, destination: Path) -> None:
    temporary = destination.with_suffix(f".{uuid.uuid4().hex}.tmp")
    try:
        image.save(temporary, format="WEBP", quality=95)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _remove_product_media(product_id: uuid.UUID, settings) -> None:
    prod_dir = settings.media_dir / "products" / str(product_id)
    if prod_dir.is_dir():
        shutil.rmtree(prod_dir)


async def _load_existing_products(session: AsyncSession) -> tuple[dict[str, uuid.UUID], dict[tuple[str, str], uuid.UUID]]:
    res_prods = await session.execute(
        select(Product.id, Product.slug, Product.title, Product.manufacturer)
    )
    slug_to_id: dict[str, uuid.UUID] = {}
    tm_to_id: dict[tuple[str, str], uuid.UUID] = {}
    for p_id, p_slug, p_title, p_manuf in res_prods.all():
        if p_slug:
            slug_to_id[p_slug.strip()] = p_id
        key = (p_title.strip().lower(), p_manuf.strip().lower())
        tm_to_id[key] = p_id
    return slug_to_id, tm_to_id


async def _load_embedding_counts(session: AsyncSession) -> tuple[dict[uuid.UUID, int], dict[uuid.UUID, int]]:
    res_v1 = await session.execute(
        select(ProductEmbedding.product_id, func.count(ProductEmbedding.id)).group_by(ProductEmbedding.product_id)
    )
    v1_counts: dict[uuid.UUID, int] = dict(res_v1.all())

    res_v4 = await session.execute(
        select(ProductEmbeddingV4.product_id, func.count(ProductEmbeddingV4.id)).group_by(ProductEmbeddingV4.product_id)
    )
    v4_counts: dict[uuid.UUID, int] = dict(res_v4.all())

    return v1_counts, v4_counts


def _evaluate_product_status(
    slug: str,
    title: str,
    manufacturer: str,
    slug_to_id: dict[str, uuid.UUID],
    tm_to_id: dict[tuple[str, str], uuid.UUID],
    v1_counts: dict[uuid.UUID, int],
    v4_counts: dict[uuid.UUID, int],
) -> tuple[uuid.UUID | None, bool, bool]:
    """Return (product_id, is_repair, is_complete)."""
    prod_id: uuid.UUID | None = None
    if slug and slug in slug_to_id:
        prod_id = slug_to_id[slug]
    else:
        key = (title.strip().lower(), manufacturer.strip().lower())
        if key in tm_to_id:
            prod_id = tm_to_id[key]

    if prod_id is None:
        return None, False, False

    c_v1 = v1_counts.get(prod_id, 0)
    c_v4 = v4_counts.get(prod_id, 0)
    return prod_id, False, c_v1 == 116 and c_v4 == 116


def _dry_run_summary(rows: list[dict[str, str]], images_dir: Path) -> int:
    found = missing = missing_meta = 0
    samples: list[tuple[str, str, str | None]] = []
    for row in rows:
        title = (row.get("title") or row.get("Название вина") or "").strip()
        manufacturer = (row.get("manufacturer") or row.get("Винодельня") or "").strip()
        if not title or not manufacturer:
            missing_meta += 1
            logger.warning("Row missing title/manufacturer, skipping: %s", row)
            continue

        img = resolve_image_path(row, images_dir)
        if img is None:
            missing += 1
        else:
            found += 1
            if len(samples) < 5:
                samples.append((title, manufacturer, img.name))

    logger.info(
        "DRY RUN SUMMARY: total=%d, found_image=%d, missing_image=%d, missing_meta=%d",
        len(rows), found, missing, missing_meta,
    )
    for title, manufacturer, img_name in samples:
        logger.info("  sample: '%s' / '%s' -> %s", title[:40], manufacturer[:40], img_name)
    return 0


async def ingest_cascade_catalog(
    csv_path: Path,
    images_dir: Path,
    batch_size: int = 32,
    limit: int = 0,
    force_rebuild: bool = False,
    dry_run: bool = False,
) -> int:
    settings = get_settings()

    if not csv_path.is_file():
        logger.error("CSV catalog file not found: %s", csv_path)
        return 1

    images_dir.mkdir(parents=True, exist_ok=True)
    if not images_dir.is_dir():
        logger.error("Images directory not found and could not be created: %s", images_dir)
        return 1

    rows: list[dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    if limit > 0:
        rows = rows[:limit]

    total = len(rows)

    if dry_run:
        return _dry_run_summary(rows, images_dir)

    logger.info("Initializing database connection...")
    engine = create_async_engine(settings.database_url, echo=False)
    session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    async with session_factory() as session:
        slug_to_id, tm_to_id = await _load_existing_products(session)
        v1_counts, v4_counts = await _load_embedding_counts(session)

    logger.info(
        "Database ready: %d existing products (%d with full v1, %d with full v4).",
        len(slug_to_id) + len(set(tm_to_id.values()) - set(slug_to_id.values())),
        sum(1 for c in v1_counts.values() if c == 116),
        sum(1 for c in v4_counts.values() if c == 116),
    )

    _ensure_media_dir(settings)

    # GPU concurrency semaphore shared by detector and embedding services
    gpu_semaphore = asyncio.Semaphore(settings.gpu_concurrency)

    logger.info("Loading YOLO detector (%s)...", settings.yolo_model_path)
    detector = await asyncio.to_thread(DetectorService, settings.yolo_model_path, settings.yolo_confidence)

    logger.info("Loading DINOv2 v1 (384d)...")
    v1_embeddings = await asyncio.to_thread(
        EmbeddingService,
        settings.resolved_dino_model_path,
        settings.resolved_dino_base_model_path,
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

    logger.info("Starting unified Cascade ingestion for %d items...", total)
    ingested = 0
    repaired = 0
    skipped = 0
    missing_img = 0
    failed = 0
    t_start = time.perf_counter()

    for idx, row in enumerate(rows, 1):
        slug = (row.get("Slug") or "").strip()
        title = (row.get("title") or row.get("Название вина") or "").strip()
        manufacturer = (row.get("manufacturer") or row.get("Винодельня") or "").strip()
        description = (row.get("description") or row.get("Описание") or "").strip()

        if not title or not manufacturer:
            logger.warning("[%d/%d] Skipping row without title/manufacturer", idx, total)
            failed += 1
            continue

        prod_id, _, is_complete = _evaluate_product_status(
            slug, title, manufacturer, slug_to_id, tm_to_id, v1_counts, v4_counts
        )

        needs_processing = True
        is_repair = False
        if prod_id is not None and not force_rebuild:
            if is_complete:
                skipped += 1
                needs_processing = False
            else:
                is_repair = True
                c_v1 = v1_counts.get(prod_id, 0)
                c_v4 = v4_counts.get(prod_id, 0)
                logger.info(
                    "[%d/%d] Repairing incomplete product %s (v1=%d/116, v4=%d/116)...",
                    idx, total, title[:35], c_v1, c_v4,
                )

        if not needs_processing:
            continue

        img_file = resolve_image_path(row, images_dir)
        if img_file is None:
            missing_img += 1
            if missing_img <= 10 or missing_img % 100 == 0:
                logger.warning(
                    "[%d/%d] Image file missing for '%s' (%s)",
                    idx, total, title[:40], slug or "no-slug",
                )
            continue

        t0 = time.perf_counter()
        created_media = False
        if prod_id is None:
            prod_id = uuid.uuid4()

        try:
            with Image.open(img_file) as f_img:
                source_image = f_img.convert("RGB")

            # 1. Single YOLO detection (synchronous in thread, guarded by GPU semaphore)
            async with gpu_semaphore:
                box = await asyncio.to_thread(detector.best_box, source_image)
            if box is not None:
                x1, y1, x2, y2 = box
                w, h = source_image.size
                ix1 = max(0, min(w - 1, int(x1)))
                iy1 = max(0, min(h - 1, int(y1)))
                ix2 = max(0, min(w, int(x2)))
                iy2 = max(0, min(h, int(y2)))
                if ix2 > ix1 and iy2 > iy1:
                    raw_crop = source_image.crop((ix1, iy1, ix2, iy2))
                else:
                    raw_crop = source_image
            else:
                raw_crop = source_image

            # 2. Canonical Letterbox 518x518 (preserves aspect ratio)
            canonical_518 = letterbox_pil(raw_crop, settings.canonical_size_v4)

            # 3. Generate 116-augmentation cloud in RAM
            aug_tuples = generate_augmented_cloud_v3(canonical_518, variants_per_aug=5, seed=42)
            aug_images = [t[2] for t in aug_tuples]

            # 4. Batch compute v1 embeddings (DINOv2, 384d, 116 vectors)
            async with gpu_semaphore:
                v1_vectors = await asyncio.to_thread(v1_embeddings.embed_batch, aug_images, batch_size)

            # 5. Batch compute v4 embeddings (SigLIP 2, 768d, 116 vectors)
            async with gpu_semaphore:
                v4_vectors = await asyncio.to_thread(v4_embeddings.embed_batch, aug_images, batch_size)

            # 6. Save media master files atomically
            source_rel, label_rel = _save_product_media(prod_id, source_image, canonical_518, settings)
            created_media = True

            # 7. Atomically persist Product + 116 v1 vectors + 116 v4 vectors
            async with session_factory() as session:
                if is_repair or force_rebuild:
                    await session.execute(delete(ProductEmbedding).where(ProductEmbedding.product_id == prod_id))
                    await session.execute(delete(ProductEmbeddingV4).where(ProductEmbeddingV4.product_id == prod_id))

                    existing_prod = await session.get(Product, prod_id)
                    if existing_prod is not None:
                        existing_prod.slug = slug or None
                        existing_prod.title = title
                        existing_prod.manufacturer = manufacturer
                        existing_prod.description = description
                        existing_prod.source_image_path = source_rel
                        existing_prod.label_image_path = label_rel
                else:
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

                v1_rows = [
                    ProductEmbedding(
                        product_id=prod_id,
                        image_path=label_rel,
                        sample_type="catalog" if aug_name == "catalog" else "augmented",
                        embedding=vec_v1,
                        embedding_model=settings.embedding_model_name,
                    )
                    for (aug_name, _, _), vec_v1 in zip(aug_tuples, v1_vectors, strict=True)
                ]
                session.add_all(v1_rows)

                v4_rows = [
                    ProductEmbeddingV4(
                        product_id=prod_id,
                        image_path=label_rel,
                        sample_type="catalog" if aug_name == "catalog" else "augmented",
                        aug_name=aug_name,
                        aug_seed=variant_seed,
                        embedding=vec_v4,
                        embedding_model=settings.embedding_model_v4_name,
                    )
                    for (aug_name, variant_seed, _), vec_v4 in zip(aug_tuples, v4_vectors, strict=True)
                ]
                session.add_all(v4_rows)

                await session.commit()

            # Update cache maps
            if slug:
                slug_to_id[slug] = prod_id
            tm_to_id[(title.lower(), manufacturer.lower())] = prod_id
            v1_counts[prod_id] = 116
            v4_counts[prod_id] = 116

            if is_repair:
                repaired += 1
            else:
                ingested += 1

            total_done = ingested + repaired
            if total_done % 10 == 0 or idx == total:
                elapsed_total = time.perf_counter() - t_start
                item_time = time.perf_counter() - t0
                rate = total_done / elapsed_total if elapsed_total > 0 else 0
                logger.info(
                    "[%d/%d] Ingested: %d, Repaired: %d, Skipped: %d, Missing: %d (%.2f s/item, %.1f items/min)",
                    idx,
                    total,
                    ingested,
                    repaired,
                    skipped,
                    missing_img,
                    item_time,
                    rate * 60.0,
                )

        except Exception as e:
            logger.error("[%d/%d] Error processing '%s': %s", idx, total, title, e, exc_info=True)
            failed += 1
            if created_media:
                try:
                    _remove_product_media(prod_id, settings)
                except Exception as cleanup_err:
                    logger.warning("Failed to clean up media for product %s: %s", prod_id, cleanup_err)

    await engine.dispose()
    total_time = time.perf_counter() - t_start
    logger.info(
        "INGESTION COMPLETE! Ingested: %d, Repaired: %d, Skipped: %d, Missing: %d, Failed: %d. Total time: %.1f min",
        ingested,
        repaired,
        skipped,
        missing_img,
        failed,
        total_time / 60.0,
    )
    return 0


def main() -> int:
    default_csv, default_images = get_default_paths()
    parser = argparse.ArgumentParser(description="Unified High-Performance Cascade Catalog Ingestion (v1 + v4)")
    parser.add_argument("--csv", type=Path, default=default_csv, help="CSV catalog path")
    parser.add_argument("--images-dir", type=Path, default=default_images, help="Images directory")
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding mini-batch size (default: 32)")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of items to process (0 = all)")
    parser.add_argument("--force-rebuild", action="store_true", help="Force re-generation of embeddings even if existing")
    parser.add_argument("--dry-run", action="store_true", help="Check CSV/image mappings and existing DB state without loading models or writing")
    args = parser.parse_args()

    return asyncio.run(
        ingest_cascade_catalog(
            csv_path=args.csv,
            images_dir=args.images_dir,
            batch_size=args.batch_size,
            limit=args.limit,
            force_rebuild=args.force_rebuild,
            dry_run=args.dry_run,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
