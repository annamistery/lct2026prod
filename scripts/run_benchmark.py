#!/usr/bin/env python3
"""Unified benchmark runner: evaluates v1, cascade, v2 and v3 endpoints across all test packs.

Writes predictions and reports to a writable media directory (/media/benchmark),
preventing read-only filesystem errors in Docker.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent


def get_writable_dir(custom_path: Path | None = None) -> Path:
    """Finds a guaranteed writable directory for benchmark outputs."""
    if custom_path is not None:
        custom_path.mkdir(parents=True, exist_ok=True)
        return custom_path

    candidates = [
        Path("/media/benchmark"),
        ROOT_DIR / "media" / "benchmark",
        Path("/tmp/benchmark"),
        Path("media/benchmark"),
    ]

    for cand in candidates:
        try:
            cand.mkdir(parents=True, exist_ok=True)
            # Test write access
            test_file = cand / ".write_test"
            test_file.write_text("ok", encoding="utf-8")
            test_file.unlink(missing_ok=True)
            return cand
        except Exception:
            continue

    # Fallback to current working directory
    fallback = Path("benchmark_results")
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def find_pack_dir(pack_name: str) -> Path | None:
    candidates = [
        ROOT_DIR / "tmp" / pack_name,
        Path("/srv/app/tmp") / pack_name,
        Path("tmp") / pack_name,
    ]
    for cand in candidates:
        if cand.is_dir() and (cand / "queries.tsv").is_file():
            return cand
    return None


def run_participant_test(pack_dir: Path, endpoint: str, output_file: Path, cwd: Path) -> bool:
    """Executes participant_test.sh inside a writable working directory."""
    script = pack_dir / "participant_test.sh"
    manifest = pack_dir / "queries.tsv"
    images = pack_dir / "queries"

    output_file.unlink(missing_ok=True)

    # PATH lookup: on Windows a bare "bash" resolves to System32\bash.exe (WSL) before Git Bash.
    cmd = [
        shutil.which("bash") or "bash",
        str(script),
        "--images-dir",
        str(images),
        "--manifest",
        str(manifest),
        "--endpoint",
        endpoint,
        "--output",
        str(output_file),
    ]

    print(f"  > Запуск: {script.name} -> {endpoint} ...")
    start = time.perf_counter()
    res = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    elapsed = time.perf_counter() - start

    if res.returncode != 0:
        print(f"  [ОШИБКА] Скрипт завершился с кодом {res.returncode}:\n{res.stderr.strip()}", file=sys.stderr)
        return False

    print(f"  ✓ Завершено за {elapsed:.1f}с. Сохранено в {output_file.name}")
    return True


def evaluate_pack(mapping_file: Path, preds_file: Path) -> dict:
    if not mapping_file.is_file() or not preds_file.is_file():
        return {"total": 0, "correct": 0, "pct": 0.0, "avg_lat": 0.0, "details": {}}

    try:
        cases = json.loads(mapping_file.read_text(encoding="utf-8")).get("cases", [])
        mapping = {c["query_id"]: c.get("expected_slug") for c in cases}
    except Exception:
        return {"total": 0, "correct": 0, "pct": 0.0, "avg_lat": 0.0, "details": {}}

    lines = [line.strip() for line in preds_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    details = {}
    latencies = []
    correct = 0

    for line in lines:
        try:
            p = json.loads(line)
        except Exception:
            continue
        qid = p.get("query_id", "")
        exp = mapping.get(qid)
        got = p.get("predicted_slug")
        lat = p.get("latency_ms", 0)
        if isinstance(lat, (int, float)):
            latencies.append(lat)

        is_ok = exp is not None and got is not None and exp == got
        if is_ok:
            correct += 1

        details[qid] = {
            "expected": exp,
            "predicted": got,
            "latency": lat,
            "is_ok": is_ok,
        }

    total = len(details)
    pct = (correct / total * 100.0) if total > 0 else 0.0
    avg_lat = (sum(latencies) / len(latencies)) if latencies else 0.0

    return {
        "total": total,
        "correct": correct,
        "pct": pct,
        "avg_lat": avg_lat,
        "details": details,
    }


# Benchmarked endpoints: (key for file names, column label, path). v1 is the baseline of the diffs.
ENDPOINTS = [
    ("v1", "v1 (каскад)", "/api/v1/eval/predict"),
    ("cascade_eval", "cascade/eval/predict", "/api/cascade/eval/predict"),
    ("cascade", "cascade/predict", "/api/cascade/predict"),
    ("v2", "v2", "/api/v2/eval/predict"),
    ("v3", "v3 (base 768d)", "/api/v3/eval/predict"),
]

PACKS = [
    ("1", "Пакет 1 (set48 / каталог)"),
    ("2", "Пакет 2 (vina / полевые)"),
]


def main():
    parser = argparse.ArgumentParser(description="Run the evaluation benchmark: v1, cascade, v2 and v3 endpoints")
    parser.add_argument("--host", default="http://127.0.0.1:8030", help="Base URL of backend (default: http://127.0.0.1:8030)")
    parser.add_argument("--out-dir", type=Path, default=None, help="Directory for output predictions (defaults to /media/benchmark)")
    parser.add_argument("--skip-run", action="store_true", help="Skip running participant_test.sh, analyze existing predictions")
    parser.add_argument("--skip-v2", action="store_true", help="Skip v2 tests")
    parser.add_argument("--skip-v3", action="store_true", help="Skip v3 tests (useful if v3 model is not yet loaded)")
    args = parser.parse_args()

    packs = []
    for pack_name, pack_label in PACKS:
        pack_dir = find_pack_dir(pack_name)
        if not pack_dir:
            print(f"ОШИБКА: Не найдена директория с тестовым пакетом tmp/{pack_name}!", file=sys.stderr)
            sys.exit(1)
        packs.append((pack_name, pack_label, pack_dir))

    skipped = {key for key, flag in (("v2", args.skip_v2), ("v3", args.skip_v3)) if flag}
    endpoints = [ep for ep in ENDPOINTS if ep[0] not in skipped]

    out_dir = get_writable_dir(args.out_dir)
    print(f"Директория для сохранения результатов бенчмарка: {out_dir}")

    def out_file(pack_name: str, key: str) -> Path:
        return out_dir / f"p{pack_name}_predictions_{key}.jsonl"

    host = args.host.rstrip("/")
    labels = ", ".join(label for _, label, _ in endpoints)

    if not args.skip_run:
        print("\n" + "=" * 80)
        print(f" ЗАПУСК ЕДИНОГО ТЕСТИРОВАНИЯ ({labels}) — хост {host}")
        print("=" * 80)

        total_steps = len(packs) * len(endpoints)
        step = 0
        for pack_name, pack_label, pack_dir in packs:
            for key, label, path in endpoints:
                step += 1
                print(f"\n[{step}/{total_steps}] {pack_label} -> {label}...")
                run_participant_test(pack_dir, f"{host}{path}", out_file(pack_name, key), cwd=out_dir)

    # Evaluate results: results[key][pack_name]
    results = {key: {pack_name: evaluate_pack(pack_dir / "mapping.json", out_file(pack_name, key)) for pack_name, _, pack_dir in packs} for key, _, _ in endpoints}

    def totals(per_pack: dict) -> dict:
        c = sum(r["correct"] for r in per_pack.values())
        t = sum(r["total"] for r in per_pack.values())
        pct = (c / t * 100.0) if t else 0.0
        lat = (sum(r["avg_lat"] * r["total"] for r in per_pack.values()) / t) if t else 0.0
        return {"correct": c, "total": t, "pct": pct, "avg_lat": lat}

    def fmt(r):
        return f"{r['correct']}/{r['total']} ({r['pct']:.1f}%) [{r['avg_lat']:.0f}мс]"

    # Print summary table
    col_w = 28 + 27 * len(endpoints)
    print("\n" + "=" * col_w)
    print(f" СВОДНЫЙ ОТЧЁТ: {labels}")
    print("=" * col_w)
    print(f"{'Тестовый набор':<28}" + "".join(f" | {label:<24}" for _, label, _ in endpoints))
    print("-" * col_w)
    for pack_name, pack_label, _ in packs:
        print(f"{pack_label:<28}" + "".join(f" | {fmt(results[key][pack_name]):<24}" for key, _, _ in endpoints))
    print("-" * col_w)
    query_count = totals(results[endpoints[0][0]])["total"]
    print(f"{f'ИТОГО (запросов: {query_count})':<28}" + "".join(f" | {fmt(totals(results[key])):<24}" for key, _, _ in endpoints))
    print("=" * col_w)

    # Detailed differential analysis: v1 vs every other endpoint
    base_key, base_label, _ = endpoints[0]
    for key, label, _ in endpoints[1:]:
        for pack_name, pack_label, _ in packs:
            print(f"\n--- Детальная динамика {base_label} -> {label}: {pack_label} ---")
            _print_diff(results[base_key][pack_name], results[key][pack_name], base_label, label)

    print("\n" + "=" * col_w + "\n")


def _print_diff(r1: dict, r2: dict, label1: str, label2: str):
    fixed = []
    broken = []
    still_fail = []

    all_qids = sorted(set(r1["details"].keys()) | set(r2["details"].keys()))
    for qid in all_qids:
        d1 = r1["details"].get(qid, {})
        d2 = r2["details"].get(qid, {})
        ok1 = d1.get("is_ok", False)
        ok2 = d2.get("is_ok", False)
        exp = d1.get("expected") or d2.get("expected")

        if not ok1 and ok2:
            fixed.append((qid, exp, d1.get("predicted"), d2.get("predicted")))
        elif ok1 and not ok2:
            broken.append((qid, exp, d1.get("predicted"), d2.get("predicted")))
        elif not ok1 and not ok2:
            still_fail.append((qid, exp, d1.get("predicted"), d2.get("predicted")))

    if fixed:
        print(f"  + ИСПРАВЛЕНО в {label2} ({len(fixed)} шт.):")
        for qid, exp, p1, p2 in fixed:
            print(f"     [+] {qid}: ожидался {exp} | {label1}='{p1}' -> {label2}='{p2}'")
    else:
        print(f"  - Нет запросов, исправленных в {label2}.")

    if broken:
        print(f"  ! РЕГРЕССИИ в {label2} ({len(broken)} шт.):")
        for qid, exp, p1, p2 in broken:
            print(f"     [-] {qid}: ожидался {exp} | {label1}='{p1}' (OK) -> {label2}='{p2}' (ERR)")
    else:
        print(f"  + Регрессий в {label2} нет.")

    if still_fail:
        print(f"  x Ошибки в обеих ({len(still_fail)} шт.):")
        for qid, exp, p1, p2 in still_fail:
            print(f"     [x] {qid}: ожидался {exp} | {label1}='{p1}' | {label2}='{p2}'")


if __name__ == "__main__":
    main()
