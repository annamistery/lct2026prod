"""Recognition quality of the running service on labelled photo sets (HTTP, no GPU access needed).

A set is a folder with photos (directly or in queries/) and one of the label files:
  mapping.json — test packs format: cases[].image_path, cases[].expected_slug (empty = not in catalog);
  labels.tsv   — image_path, expected_slug, status (in_catalog | not_in_catalog | unsure), alt_slugs (comma separated).

Metrics:
  in-catalog photos: top-1 accuracy (answer slug is the expected one or an accepted alternative), answer zones,
                     wrong «нет в каталоге» answers;
  not-in-catalog photos: how many got «нет в каталоге», «похоже» or a false «найдено»;
  latency.

    python scripts/eval_recognition.py --api http://127.0.0.1:8030 --sets tmp1=tmp/1 tmp2=tmp/2 real="/data/Реальные фото"
"""

from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import statistics
import sys
import time
import urllib.parse
import urllib.request
import uuid
from collections import Counter
from pathlib import Path

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def load_labels(folder: Path) -> dict[str, tuple[str, set[str]]]:
    """image_path -> (kind, accepted slugs); kind is 'in', 'out' or 'skip'."""
    labels: dict[str, tuple[str, set[str]]] = {}
    mapping = folder / "mapping.json"
    if mapping.is_file():
        for case in json.loads(mapping.read_text(encoding="utf-8")).get("cases", []):
            slug = case.get("expected_slug") or ""
            labels[case["image_path"]] = ("in", {slug}) if slug else ("out", set())
    tsv = folder / "labels.tsv"
    if tsv.is_file():
        with tsv.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                status = row.get("status") or ("in_catalog" if row.get("expected_slug") else "not_in_catalog")
                if status == "in_catalog":
                    slugs = {row["expected_slug"], *filter(None, (row.get("alt_slugs") or "").split(","))}
                    labels[row["image_path"]] = ("in", slugs)
                else:
                    labels[row["image_path"]] = ("out" if status == "not_in_catalog" else "skip", set())
    return labels


def catalog_has(api: str, slug: str, cache: dict[str, bool]) -> bool:
    """Expected slugs missing from the catalog turn the photo into a not-in-catalog case."""
    if slug not in cache:
        url = f"{api}/api/products?" + urllib.parse.urlencode({"q": slug, "limit": 100})
        with urllib.request.urlopen(url, timeout=30) as response:
            cache[slug] = any(product["slug"] == slug for product in json.load(response)["products"])
    return cache[slug]


def predict(api: str, path: Path) -> tuple[dict, float]:
    boundary = uuid.uuid4().hex
    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    body = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="{path.name}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode() + path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    request = urllib.request.Request(f"{api}/api/cascade/predict", data=body, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=120) as response:
        answer = json.load(response)
    return answer, (time.perf_counter() - started) * 1000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--api", default="http://127.0.0.1:8030")
    parser.add_argument("--sets", nargs="+", required=True, help="name=folder")
    parser.add_argument("--out", type=Path, default=None, help="JSONL with every answer")
    args = parser.parse_args()

    rows = []
    known: dict[str, bool] = {}
    for spec in args.sets:
        name, folder_text = spec.split("=", 1)
        folder = Path(folder_text)
        images_dir = folder / "queries" if (folder / "queries").is_dir() else folder
        labels = load_labels(folder)
        if not labels:
            sys.exit(f"{folder}: no mapping.json or labels.tsv")
        for image_path, (kind, slugs) in sorted(labels.items()):
            if kind == "skip":
                continue
            if kind == "in":
                slugs = {slug for slug in slugs if catalog_has(args.api, slug, known)}
                kind = "in" if slugs else "out"
            path = images_dir / image_path
            if path.suffix.lower() not in IMAGE_EXT or not path.is_file():
                print(f"skip missing {path}", file=sys.stderr)
                continue
            answer, latency = predict(args.api, path)
            rows.append({"set": name, "image_path": image_path, "kind": kind, "expected": sorted(slugs), "latency_ms": round(latency), **answer})
            print(f"\r{name}: {len(rows)}", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    if args.out:
        args.out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

    for name in [*dict.fromkeys(row["set"] for row in rows), "ALL"]:
        part = [row for row in rows if name in ("ALL", row["set"])]
        inside = [row for row in part if row["kind"] == "in"]
        outside = [row for row in part if row["kind"] == "out"]
        print(f"\n== {name}")
        if inside:
            correct = [row for row in inside if row["slug"] in row["expected"]]
            zones = Counter(row["status"] for row in inside)
            found = [row for row in inside if row["status"] == "found"]
            found_ok = sum(row["slug"] in row["expected"] for row in found)
            print(f"  в каталоге: {len(inside)} фото, top-1 верно {len(correct)} ({100 * len(correct) / len(inside):.1f}%)")
            print(f"    найдено {zones['found']} (верно {found_ok}), похоже {zones['probable']}, ошибочно «нет в каталоге» {zones['not_in_catalog']}")
        if outside:
            zones = Counter(row["status"] for row in outside)
            print(f"  нет в каталоге: {len(outside)} фото -> «нет в каталоге» {zones['not_in_catalog']}, похоже {zones['probable']}, ложно «найдено» {zones['found']}")
        latencies = sorted(row["latency_ms"] for row in part)
        if latencies:
            print(f"  время, мс: медиана {statistics.median(latencies):.0f}, p95 {latencies[max(0, int(len(latencies) * 0.95) - 1)]}, макс {latencies[-1]}")


if __name__ == "__main__":
    main()
