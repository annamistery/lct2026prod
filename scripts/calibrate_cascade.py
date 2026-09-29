"""Offline calibration of the cascade on dumped vectors (no GPU or DB needed).

Input: the folder written by scripts/research_dump_features.py (catalog.npz, queries.npz, queries.json).
Replays the production cascade exactly (DINOv2 candidates by the averaged embedding, fusion of the two
views of both models) and reports:

* top-1 accuracy on labelled in-catalog photos;
* answer zones for in-catalog photos, real not-in-catalog photos and simulated ones (leave-one-out:
  the true product is removed from the catalog, its same-producer twins stay — the hardest case);
* recommended thresholds:
    reject_below_similarity  = lowest SigLIP 2 similarity of an in-catalog photo − margin
                               (so «нет в каталоге» was never wrong on the validation photos);
    found_min_similarity     = highest similarity of a real not-in-catalog photo + margin.

    python scripts/calibrate_cascade.py media/research --v1-weight 0.3 --pool 30
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def normalize(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("research_dir", type=Path)
    parser.add_argument("--v1-weight", type=float, default=0.3)
    parser.add_argument("--pool", type=int, default=30)
    parser.add_argument("--reject-below", type=float, default=0.55)
    parser.add_argument("--found-similarity", type=float, default=0.83)
    parser.add_argument("--found-margin", type=float, default=0.15)
    parser.add_argument("--safety", type=float, default=0.02, help="margin added to recommended thresholds")
    args = parser.parse_args()

    catalog = np.load(args.research_dir / "catalog.npz")
    queries = np.load(args.research_dir / "queries.npz")
    meta = json.loads((args.research_dir / "queries.json").read_text(encoding="utf-8"))
    slug_index = {slug: i for i, slug in enumerate(catalog["slug"]) if slug}
    n_products = len(catalog["slug"])
    refs = {m: (normalize(catalog[f"{m}_emb"]), catalog[f"{m}_pid"]) for m in ("v1", "v4")}

    def per_product(model: str, query: np.ndarray) -> np.ndarray:
        matrix, pid = refs[model]
        best = np.full(n_products, -1.0, dtype=np.float32)
        np.maximum.at(best, pid, matrix @ normalize(query))
        return best

    def answer(i: int, exclude: set[int]) -> dict:
        views = {v: per_product(v[:2], queries[v][i]) for v in ("v1_lb", "v1_seg", "v4_lb", "v4_seg")}
        averaged = per_product("v1", normalize(queries["v1_lb"][i]) + normalize(queries["v1_seg"][i]))
        for p in exclude:
            averaged[p] = -2.0
        pool = np.argsort(-averaged)[: args.pool]
        fusion = views["v4_lb"][pool] + views["v4_seg"][pool] + args.v1_weight * (views["v1_lb"][pool] + views["v1_seg"][pool])
        order = np.argsort(-fusion)
        best = int(pool[order[0]])
        return {
            "top": best,
            "similarity": float(max(views["v4_lb"][best], views["v4_seg"][best])),
            "margin": float(fusion[order[0]] - fusion[order[1]]) if len(order) > 1 else float("inf"),
        }

    def zone(a: dict) -> str:
        if a["similarity"] < args.reject_below:
            return "not_in_catalog"
        if a["similarity"] >= args.found_similarity and a["margin"] >= args.found_margin:
            return "found"
        return "probable"

    positives, real_negatives, simulated = [], [], []
    for i, m in enumerate(meta):
        known = {slug_index[s] for s in m.get("expected", []) if s in slug_index}
        if m.get("expect") == "in" and known:
            a = answer(i, set())
            a["correct"] = a["top"] in known
            positives.append(a)
            simulated.append(answer(i, known))
        elif m.get("expect") in ("in", "out"):
            real_negatives.append(answer(i, set()))

    correct = sum(a["correct"] for a in positives)
    print(f"in-catalog photos: {len(positives)}, top-1 correct {correct} ({100 * correct / max(1, len(positives)):.1f}%)")
    for name, items in (("in-catalog", positives), ("real not-in-catalog", real_negatives), ("simulated not-in-catalog", simulated)):
        zones = Counter(zone(a) for a in items)
        extra = ""
        if name == "in-catalog":
            found_ok = sum(a["correct"] for a in items if zone(a) == "found")
            extra = f" (found correct: {found_ok}/{zones['found']})"
        print(f"  {name:25s} found {zones['found']:3d}  probable {zones['probable']:3d}  not_in_catalog {zones['not_in_catalog']:3d}{extra}")

    if positives:
        weakest = min(a["similarity"] for a in positives)
        print(f"\nlowest similarity of an in-catalog photo: {weakest:.3f} -> reject_below_similarity <= {weakest - args.safety:.3f}")
    if real_negatives:
        strongest = max(a["similarity"] for a in real_negatives)
        print(f"highest similarity of a real not-in-catalog photo: {strongest:.3f} -> found_min_similarity >= {strongest + args.safety:.3f}")


if __name__ == "__main__":
    main()
