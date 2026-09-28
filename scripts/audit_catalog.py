"""Catalog audit: products whose reference label images are (almost) the same picture.

Such pairs cannot be told apart by any image model: the cascade answers «похоже» for them and a
test photo of one of them may be scored as an error. Typical causes: the same wine loaded twice
(«Мускат» and «Бельбек Мускат»), a label shared by several wines of a series, or a wrong photo
attached to a product. The report is meant for manual cleanup of the catalog.

Uses the un-augmented SigLIP 2 reference vector of every product (aug_name = 'catalog').

    docker compose exec api python scripts/audit_catalog.py --threshold 0.97
    # -> /media/artifacts/catalog_audit.tsv
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine

from app.core.config import get_settings
from app.db.models.product import Product, ProductEmbeddingV4


async def load(database_url: str) -> tuple[list[tuple[str, str, str]], np.ndarray]:
    engine = create_async_engine(database_url)
    async with engine.connect() as conn:
        rows = (
            await conn.execute(
                select(Product.slug, Product.title, Product.manufacturer, ProductEmbeddingV4.embedding)
                .join(ProductEmbeddingV4, ProductEmbeddingV4.product_id == Product.id)
                .where(ProductEmbeddingV4.aug_name == "catalog")
                .order_by(Product.slug)
            )
        ).all()
    await engine.dispose()
    seen, products, vectors = set(), [], []
    for slug, title, manufacturer, vector in rows:
        if slug in seen:
            continue
        seen.add(slug)
        products.append((slug or "", title, manufacturer))
        vectors.append(np.asarray(vector, dtype=np.float32))
    matrix = np.stack(vectors) if vectors else np.zeros((0, 768), dtype=np.float32)
    return products, matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-9)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--threshold", type=float, default=0.97, help="cosine similarity of catalog labels to report")
    parser.add_argument("--out", type=Path, default=None, help="default: <media>/artifacts/catalog_audit.tsv")
    args = parser.parse_args()

    settings = get_settings()
    products, matrix = asyncio.run(load(settings.database_url))
    similarity = matrix @ matrix.T
    pairs = [(float(similarity[i, j]), i, j) for i, j in zip(*np.where(np.triu(similarity, 1) >= args.threshold), strict=True)]
    pairs.sort(reverse=True)

    out = args.out or settings.media_dir / "artifacts" / "catalog_audit.tsv"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = ["similarity\tsame_manufacturer\tslug_a\ttitle_a\tslug_b\ttitle_b\tmanufacturer_a\tmanufacturer_b"]
    for score, i, j in pairs:
        a, b = products[i], products[j]
        lines.append(f"{score:.4f}\t{a[2] == b[2]}\t{a[0]}\t{a[1]}\t{b[0]}\t{b[1]}\t{a[2]}\t{b[2]}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(products)} products, {len(pairs)} pairs with label similarity >= {args.threshold} -> {out}")
    for line in lines[1:11]:
        print("  " + line)


if __name__ == "__main__":
    main()
