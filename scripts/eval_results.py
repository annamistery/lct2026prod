#!/usr/bin/env python3
"""Evaluate prediction accuracy against mapping.json."""

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Hit@1 accuracy from predictions.jsonl and mapping.json")
    parser.add_argument("--mapping", type=Path, default=Path("mapping.json"), help="Path to mapping.json")
    parser.add_argument("--predictions", type=Path, default=Path("predictions.jsonl"), help="Path to predictions.jsonl")
    args = parser.parse_args()

    if not args.mapping.is_file():
        print(f"ОШИБКА: Файл разметки не найден: {args.mapping}", file=sys.stderr)
        return 1

    if not args.predictions.is_file():
        print(f"ОШИБКА: Файл предсказаний не найден: {args.predictions}", file=sys.stderr)
        return 1

    try:
        mapping_data = json.loads(args.mapping.read_text(encoding="utf-8"))
        cases = mapping_data.get("cases", [])
        mapping = {c["query_id"]: c.get("expected_slug") for c in cases}
    except Exception as e:
        print(f"ОШИБКА чтения mapping.json: {e}", file=sys.stderr)
        return 1

    lines = [l.strip() for l in args.predictions.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not lines:
        print(f"ОШИБКА: Файл predictions.jsonl пуст: {args.predictions}", file=sys.stderr)
        return 1

    preds = []
    for line in lines:
        try:
            preds.append(json.loads(line))
        except Exception:
            continue

    total = len(preds)
    correct = 0
    latencies = []

    print("\n" + "=" * 70)
    print(" ДЕТАЛЬНЫЙ ОТЧЁТ ПО КАЖДОМУ ЗАПРОСУ:")
    print("=" * 70)

    for p in preds:
        qid = p.get("query_id", "")
        exp = mapping.get(qid)
        got = p.get("predicted_slug")
        lat = p.get("latency_ms", 0)
        if isinstance(lat, (int, float)):
            latencies.append(lat)

        if exp and got and exp == got:
            status = "✓ OK  "
            correct += 1
            print(f"[{status}] {qid}: {got} ({lat}ms)")
        else:
            status = "✗ FAIL"
            print(f"[{status}] {qid}: ожидался={exp} | получен={got} ({lat}ms)")

    pct = (correct / total * 100.0) if total > 0 else 0.0
    avg_lat = sum(latencies) / len(latencies) if latencies else 0.0

    print("=" * 70)
    print(f" ИТОГОВАЯ ТОЧНОСТЬ Hit@1: {correct}/{total} ({pct:.1f}%)")
    print(f" Среднее время отклика: {avg_lat:.0f} мс (мин: {min(latencies) if latencies else 0}мс, макс: {max(latencies) if latencies else 0}мс)")
    print("=" * 70 + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
