"""Сборка каталога: wines_integrated.csv -> data/catalog.json (+ отчёт о полноте разметки)."""
import json
import os
import re
import sys
from collections import Counter

import pandas as pd

from .features import extract, norm

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(HERE, "data", "wines_integrated.csv")
CATALOG_JSON = os.path.join(HERE, "data", "catalog.json")

# Чистка поля «Сорт винограда»: не сорта убираем, синонимы одного сорта сводим, составные значения делим
NOT_GRAPES = {"молдова", "белые сорта винограда", "красные сорта винограда"}
GRAPE_ALIAS = {"пино черный": "Пино Нуар", "пино чёрный": "Пино Нуар", "пино чёрный (пино нуар)": "Пино Нуар",
               "пино нуар": "Пино Нуар", "сира (шираз)": "Сира", "шираз": "Сира", "мюллер тургау": "Мюллер-Тургау",
               "мюллер-тургау": "Мюллер-Тургау", "совиньон": "Совиньон Блан"}


def clean_grapes(raw):
    out = []
    for part in re.split(r",|\s+и\s+|/", raw or ""):
        p = part.strip()
        k = norm(p)
        if not p or k in NOT_GRAPES:
            continue
        # «Сира (Шираз)» -> Сира; «Пино чёрный (Пино нуар)» -> Пино Нуар
        out.append(GRAPE_ALIAS.get(k) or GRAPE_ALIAS.get(re.sub(r"\s*\(.*?\)", "", k)) or re.sub(r"\s*\(.*?\)", "", p))
    seen = []
    for g in out:
        if g not in seen:
            seen.append(g)
    return seen


# Для явной подписи типа вина по признакам из базы
def wine_type(w):
    sw = w["features"]["sweetness"]
    sp = w["features"]["sparkling"]
    sw_txt = {0: "сухое", 1: "полусухое", 2: "полусладкое", 3: "сладкое"}.get(sw, "сладость не указана")
    base = w["category"].lower()
    return f"{base} {'игристое ' if sp else ''}{sw_txt}".replace("  ", " ")


def build(csv_path=DEFAULT_CSV, out=CATALOG_JSON):
    """csv -> список вин. `out=None` пропускает запись кэша (нужно, когда каталог рядом с кодом смонтирован
    read-only, как в проде): сборка из ~2000 строк занимает доли секунды, кэш на диске не обязателен."""
    df = pd.read_csv(csv_path, encoding="utf-8-sig").fillna("")
    wines = []
    for _, r in df.iterrows():
        desc = str(r["Описание"]).strip()
        f, ev = extract(r["Название вина"], r["Slug"], r["Категория"], desc)
        w = {
            "id": r["Slug"],
            "name": str(r["Название вина"]).strip(),
            "category": r["Категория"],
            "color": r["Цвет"],
            "region": r["Регион"],
            "grape": r["Сорт винограда"],
            "grapes": clean_grapes(r["Сорт винограда"]),
            "winery": r["Винодельня"],
            "description": desc,
            "url": r["Ссылка на страницу"] or None,
            "image_file": r["Файл в wines_images"] or None,
            "image_url": r["Ссылка на изображение"] or None,
            "features": f,
            "evidence": {k: v for k, v in ev.items() if v},
        }
        w["type"] = wine_type(w)
        wines.append(w)
    if out is not None:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(wines, fh, ensure_ascii=False, indent=1)
    return wines


def load(path=CATALOG_JSON):
    if not os.path.exists(path):
        return build()
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def coverage_report(wines):
    n = len(wines)
    rep = {}
    for k in ["sweetness", "body", "acidity", "tannin", "aging_months", "serve_temp", "alcohol"]:
        rep[k] = sum(w["features"][k] is not None for w in wines)
    rep["sparkling"] = sum(w["features"]["sparkling"] for w in wines)
    rep["oak_mentioned"] = sum(w["features"]["oak"] for w in wines)
    rep["aroma_any"] = sum(bool(w["features"]["aroma"]) for w in wines)
    rep["food_in_desc"] = sum(bool(w["features"]["food_in_desc"]) for w in wines)
    return {k: f"{v} ({v / n:.0%})" for k, v in rep.items()}


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV
    ws = build(csv_path)
    print("вин:", len(ws))
    for k, v in coverage_report(ws).items():
        print(f"  {k:15s} {v}")
    print(Counter(w["type"] for w in ws).most_common(20))
