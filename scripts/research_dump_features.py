"""Dump catalog vectors and query embeddings for offline recognition research.

Runs inside the API container (GPU + DB), writes into <media>/research/:
  catalog.npz   product ids/slugs + v1 (384d) and v4 (768d) reference matrices with product index per row
  queries.npz   per query: several embedding variants (same geometry as the cascade + alternatives)
  queries.json  query metadata (set, file, expect in/out/skip, expected slugs, bbox)
  crops/<set>/<file>.png  native-resolution YOLO label crops (for OCR and visual checks)

    docker compose exec api python scripts/research_dump_features.py --sets tmp1=/srv/app/tmp/1 tmp2=/srv/app/tmp/2 real=/media/research/real
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine

from app.core.config import get_settings
from app.db.models.product import Product, ProductEmbedding, ProductEmbeddingV4
from app.services.detector import DetectorService
from app.services.embeddings import EmbeddingService
from app.services.query_prep_v3 import QueryPrepV3, letterbox_pil
from app.services.segmenter import SegmenterService
from app.services.siglip_embeddings import SigLIP2EmbeddingService

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}


async def export_catalog(database_url: str, out: Path) -> None:
    engine = create_async_engine(database_url)
    async with engine.connect() as conn:
        products = (await conn.execute(select(Product.id, Product.slug, Product.title, Product.manufacturer).order_by(Product.id))).all()
        index = {row[0]: i for i, row in enumerate(products)}
        matrices = {}
        for name, model in (("v1", ProductEmbedding), ("v4", ProductEmbeddingV4)):
            rows = (await conn.execute(select(model.product_id, model.image_path, model.embedding))).all()
            matrices[f"{name}_emb"] = np.asarray([np.asarray(r[2], dtype=np.float32) for r in rows], dtype=np.float32)
            matrices[f"{name}_pid"] = np.asarray([index[r[0]] for r in rows], dtype=np.int32)
            matrices[f"{name}_path"] = np.asarray([r[1] for r in rows])
            print(f"{name}: {len(rows)} vectors")
    await engine.dispose()
    np.savez(
        out / "catalog.npz",
        product_id=np.asarray([str(r[0]) for r in products]),
        slug=np.asarray([r[1] or "" for r in products]),
        title=np.asarray([r[2] for r in products]),
        manufacturer=np.asarray([r[3] for r in products]),
        **matrices,
    )
    print(f"catalog: {len(products)} products")


def load_expected(folder: Path) -> dict[str, dict]:
    """image_path -> {"expect": "in" | "out" | "skip", "expected": [slug, alternatives...]}.

    mapping.json (tmp packs): expected_slug, empty = not in catalog;
    labels.tsv: image_path, expected_slug, status (in_catalog | not_in_catalog | unsure), alt_slugs.
    """
    labels: dict[str, dict] = {}
    mapping = folder / "mapping.json"
    if mapping.is_file():
        for case in json.loads(mapping.read_text(encoding="utf-8")).get("cases", []):
            slug = case.get("expected_slug") or ""
            labels[case["image_path"]] = {"expect": "in" if slug else "out", "expected": [slug] if slug else []}
    tsv = folder / "labels.tsv"
    if tsv.is_file():
        with tsv.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                slug = (row.get("expected_slug") or "").strip()
                status = row.get("status") or ("in_catalog" if slug else "not_in_catalog")
                alts = [a for a in (row.get("alt_slugs") or "").split(",") if a]
                expect = {"in_catalog": "in", "not_in_catalog": "out"}.get(status, "skip")
                labels[row["image_path"]] = {"expect": expect, "expected": [slug, *alts] if slug else []}
    return labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sets", nargs="+", required=True, help="name=folder; images are read from folder/queries or folder")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--skip-catalog", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    out = args.out or settings.media_dir / "research"
    out.mkdir(parents=True, exist_ok=True)
    if not args.skip_catalog:
        asyncio.run(export_catalog(settings.database_url, out))

    detector = DetectorService(settings.resolved_yolo_model_path, settings.yolo_confidence)
    segmenter = SegmenterService(settings.resolved_yolo_seg_model_path, settings.yolo_seg_confidence)
    v1 = EmbeddingService(settings.resolved_dino_model_path, settings.resolved_dino_base_model_path, settings.embedding_dimension)
    v4 = SigLIP2EmbeddingService(settings.resolved_siglip_v4_model_path, settings.resolved_siglip_v4_base_model_path, settings.embedding_dimension_v4, True)
    prep = QueryPrepV3(detector=detector, segmenter=segmenter, target_size=settings.canonical_size_v4)

    meta, variants = [], {k: [] for k in ("v1_lb", "v1_seg", "v4_seg", "v4_lb", "v4_full", "v1_full")}
    for spec in args.sets:
        name, folder = spec.split("=", 1)
        folder = Path(folder)
        images_dir = folder / "queries" if (folder / "queries").is_dir() else folder
        expected = load_expected(folder)
        crops_dir = out / "crops" / name
        crops_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXT and not p.name.startswith("."))
        for number, path in enumerate(files, 1):
            with Image.open(path) as raw:
                image = ImageOps.exif_transpose(raw).convert("RGB")
            box = detector.best_box(image)
            crop = image
            if box is not None:
                x1, y1, x2, y2 = (int(v) for v in box)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(image.width, x2), min(image.height, y2)
                if x2 > x1 and y2 > y1:
                    crop = image.crop((x1, y1, x2, y2))
            crop.save(crops_dir / f"{path.stem}.png")
            seg_view = prep.prepare_crop(crop).image
            lb_view = letterbox_pil(crop, settings.canonical_size_v4)
            full_view = letterbox_pil(image, settings.canonical_size_v4)
            variants["v1_lb"].append(v1.embed(lb_view))
            variants["v1_seg"].append(v1.embed(seg_view))
            variants["v1_full"].append(v1.embed(full_view))
            variants["v4_seg"].append(v4.embed(seg_view))
            variants["v4_lb"].append(v4.embed(lb_view))
            variants["v4_full"].append(v4.embed(full_view))
            label = expected.get(path.name, {"expect": None, "expected": []})
            meta.append({"set": name, "file": path.name, **label, "bbox": list(box) if box else None, "size": list(image.size)})
            print(f"\r{name}: {number}/{len(files)}", end="", flush=True)
        print()

    np.savez(out / "queries.npz", **{k: np.asarray(v, dtype=np.float32) for k, v in variants.items()})
    (out / "queries.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"queries: {len(meta)} -> {out}")


if __name__ == "__main__":
    main()
