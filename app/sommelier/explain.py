"""Слой 5: объяснение на уровне гостя. Каждая фраза выводится из вкладов правил и цитат из описания вина."""
import re

from . import guardrails
from .dishes import describe_axes
from .engine import acid_words, body_words, tannin_words
from .features import SWEET_LABEL, norm

CAT_ADJ = {"Красное": "красное", "Белое": "белое", "Розовое": "розовое", "Оранжевое": "оранжевое"}


def snippet(text, n):
    """Обрезка по границе предложения/слова, чтобы цитата не обрывалась на полуслове."""
    s = " ".join((text or "").split())
    if len(s) <= n:
        return s
    cut = s[:n]
    end = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    return (cut[:end + 1] if end > n * 0.5 else cut[:cut.rfind(" ")].rstrip(",;:—- ") + "…")


def _q(ev, key, n=110):
    s = (ev or {}).get(key)
    if not s:
        return ""
    s = s.replace("\n", " ").strip()
    return s if len(s) <= n else s[:n].rstrip() + "…"


def reason_line(c, w, ctx, level):
    f, ev = w["features"], w["evidence"]
    r, novice = c["rule"], level == "novice"
    if r == "weight":
        wb, di = body_words(f["body"]), c["dish_i"]
        wb = wb + " по насыщенности" if wb == "среднее" else wb
        dw = "лёгкое" if di < 1.7 else "среднее по насыщенности" if di < 2.8 else "насыщенное"
        base = f"по «весу» они совпадают: блюдо {dw}, вино {wb}" if c["value"] > 0.3 else \
               f"по «весу» есть расхождение: блюдо {dw}, вино {wb}"
        return base + (f" (в описании: «{_q(ev, 'body', 60)}»)" if not novice and ev.get("body") else "")
    if r == "fat_acid":
        if c["value"] > 0:
            return ("жирность блюда «режет» " + ("освежающая кислотность вина" if novice else f"кислотность вина ({acid_words(f['acidity'])})")
                    + (f"; в описании: «{_q(ev, 'acidity', 70)}»" if ev.get("acidity") else ""))
        return "кислотность вина невысокая, а блюдо жирное — освежения будет меньше"
    if r == "sweet":
        wl = SWEET_LABEL.get(f["sweetness"], "сладость не указана")
        if c.get("contrast"):
            return f"вино ({wl}) создаёт приятный контраст с острым/солёным вкусом блюда"
        if c["value"] < 0:
            if f["sweetness"] is None:
                return "в базе не указана сладость вина, а блюдо сладкое — сладость не подтверждена"
            return f"вино ({wl}) менее сладкое, чем блюдо: рядом с ним оно покажется кислым и «пустым»"
        return f"сладость вина ({wl}) не ниже сладости блюда — вино не потеряется рядом с ним"
    if r == "tannin":
        if c["value"] > 0:
            if c.get("est"):
                return "танины красного вина связываются с белком и жиром блюда (в описании танины не упомянуты — оценка по типичным красным каталога)"
            return "танины связываются с белком и жиром блюда и становятся мягче" + ("" if novice else f" (в описании: «{_q(ev, 'tannin', 60)}»)")
        return "танины вина рядом с острым/солёным могут стать горькими и металлическими"
    if r == "region":
        return f"и вино, и кухня связаны с регионом «{c['region']}» — «что растёт вместе, то сочетается»"
    if r == "aroma":
        if c.get("foods"):
            return f"описание вина само называет такие блюда: {', '.join(c['foods'])}" + (f" («{_q(ev, 'food_in_desc', 110)}»)" if ev.get("food_in_desc") else "")
        return f"{c['why'] or 'ароматы вина сочетаются с блюдом'} (в описании: {', '.join(c['aromas'])})"
    if r == "occasion":
        return c["msg"]
    if r == "prefs":
        return "учтено ваше предпочтение: " + c["msg"]
    return c["msg"]


def explain_pick(r, ctx, level):
    w = r["wine"]
    comps = sorted(r["comps"], key=lambda c: -abs(c["value"] * c["weight"]))
    pros = [reason_line(c, w, ctx, level) for c in comps if c["value"] > 0.15][:3]
    cons = [reason_line(c, w, ctx, level) for c in comps if c["value"] < -0.15][:2]
    return pros, cons


def wine_facts(w, level):
    f = w["features"]
    bits = [f"{w['category'].lower()}", SWEET_LABEL.get(f["sweetness"], "сладость в базе не указана")]
    if f["sparkling"]:
        bits.insert(1, "игристое")
    line = ", ".join(bits)
    meta = [x for x in (w["grape"], w["region"], w["winery"]) if x]
    if level != "novice" and meta:
        line += " · " + " · ".join(meta)
    elif level == "novice" and w["grape"]:
        line += f" · сорт: {w['grape']}"
    extra = []
    if level == "expert":
        if f["aging_months"]:
            extra.append(f"выдержка {f['aging_months']} мес.")
        if f["oak"]:
            extra.append("в описании есть дубовые/бочковые тона")
        if f["serve_temp"]:
            extra.append(f"подача {f['serve_temp'][0]}–{f['serve_temp'][1]}°C")
        if f["alcohol"]:
            extra.append(f"алкоголь {f['alcohol']}%")
        if f["tannin"] is not None:
            extra.append(f"танины: {tannin_words(f['tannin'])}")
        if f["acidity"] is not None:
            extra.append(f"кислотность: {acid_words(f['acidity'])}")
    return line + ("; " + "; ".join(extra) if extra else "")


# ---- оценка «вино + блюдо» для инфографики сканера ------------------------------------------------------
RULE_TITLES = dict(weight="Вес и насыщенность", fat_acid="Кислотность и жир", sweet="Сладость", tannin="Танины",
                   region="Регион", aroma="Ароматы", occasion="Повод", prefs="Ваши пожелания")
# Вердикт — по месту вина среди всего каталога к этому блюду: абсолютный балл у разных блюд несопоставим
# (к стейку лучшее вино набирает 84, к утке с соусом — 76).
VERDICTS = ((0.9, "great", "Отличная пара"), (0.7, "good", "Хорошая пара"), (0.4, "ok", "Допустимо"),
            (0.0, "poor", "Не лучший выбор"))
_QUOTE = re.compile(r"\s*(?:;\s*в описании:\s*|\((?:в описании:\s*)?)«([^»]+)»\)?$")


def _split_quote(line):
    """«довод (в описании: «цитата»)» -> (довод, цитата); цитата из карточки показывается отдельно."""
    m = _QUOTE.search(line)
    return (line[:m.start()].rstrip(" ;"), m.group(1)) if m else (line, None)


def _trait(t):
    """«интенсивность насыщенное» -> «насыщенное»; «среднее» без оси непонятно — «средней насыщенности»."""
    t = t.removeprefix("интенсивность ")
    return "средней насыщенности" if t == "среднее" else t


def pairing_profile(w, ctx, score, comps, unknown, rank, better):
    """Структура для инфографики сочетания: вердикт, вклад правил «за/против», профиль блюда, лучшие блюда к вину.

    Всё выводится из тех же вкладов правил, что и текстовый ответ; новых фактов о вине нет.
    """
    level, title = next((lv, t) for lim, lv, t in VERDICTS if rank["beats"] >= lim)
    if any(c["rule"] == "sweet" and c["value"] <= -0.6 for c in comps):   # вино заметно суше блюда — главный конфликт
        level, title = VERDICTS[-1][1:]
    dish = ctx.dish
    factors = []
    for c in sorted(comps, key=lambda c: -abs(c["value"] * c["weight"])):
        text, quote = _split_quote(reason_line(c, w, ctx, "novice"))
        sign = "plus" if c["value"] > 0.15 else "minus" if c["value"] < -0.15 else "neutral"
        rule_title = "Карточка называет это блюдо" if c["rule"] == "aroma" and c.get("foods") else RULE_TITLES.get(c["rule"], c["rule"])
        factors.append(dict(rule=c["rule"], title=rule_title, sign=sign,
                            value=round(c["value"], 2), weight=c["weight"], text=text[:1].upper() + text[1:], quote=quote))
    return dict(
        dish=dict(text=ctx.dish_text or ", ".join(sorted({m.split(" («")[0] for m in dish["matched_main"]})),
                  traits=[_trait(t) for t in describe_axes(dish).split(", ")]),
        score=score, beats=rank["beats"], best=rank["best"],
        verdict=dict(level=level, title=title),
        factors=factors,
        unknown=list(unknown),
        better_dishes=better,
        card_foods=list(w["features"]["food_in_desc"]),
        note=guardrails.NEUTRAL_FOOTER,
    )


def pick_card(r, ctx, level):
    """Поля карточки вина в выдаче чата: тип, сорт, регион, доводы без цитат, короткая цитата описания.

    Те же доводы, что в текстовом ответе (explain_pick); цитата из описания идёт отдельно, одной на вино.
    """
    w, f = r["wine"], r["wine"]["features"]
    pros, cons = explain_pick(r, ctx, level)
    clean = lambda lines: [(t[:1].upper() + t[1:]) for t, _ in map(_split_quote, lines)]
    return dict(
        winery=w["winery"] or "", category=w["category"] or "", grape=w["grape"] or "", region=w["region"] or "",
        sweetness=SWEET_LABEL.get(f["sweetness"]), sparkling=bool(f["sparkling"]), url=w["url"],
        pros=clean(pros), cons=clean(cons),
        quote=snippet(w["description"], 180) if (w.get("description") or "").strip() else None,
    )


def dish_traits(dish):
    """Признаки блюда для чипов: «насыщенное», «жирное», «сладкое»…"""
    return [_trait(t) for t in describe_axes(dish).split(", ")] if dish else []


def render_recommendation(rec, ctx, type_rows, level):
    """Готовый текст ответа (без LLM). LLM-слой может переписать его, но опирается на те же факты."""
    L = []
    if ctx.dish:
        L.append(f"Разобрал блюдо: {describe_axes(ctx.dish)}.")
    if not rec["picks"]:
        L.append(f"По вашим условиям в каталоге ({rec['total']} вин) не осталось подходящих. "
                 f"Что отсеялось: {', '.join(f'{k} — {v}' for k, v in list(rec['dropped'].items())[:4])}. "
                 "Можно смягчить одно из условий (цвет, сладость, сорт).")
        return "\n".join(L)
    L.append(f"Из {rec['total']} вин каталога после ваших ограничений осталось {rec['pool_size']}. Три варианта разных стилей:")
    for i, r in enumerate(rec["picks"], 1):
        w = r["wine"]
        pros, cons = explain_pick(r, ctx, level)
        L.append(f"\n{i}. {w['name']} — {r['role']}")
        L.append(f"   {wine_facts(w, level)}")
        if pros:
            L.append("   Почему подходит: " + "; ".join(pros) + ".")
        if cons:
            L.append("   Оговорка: " + "; ".join(cons) + ".")
        if r["unknown"]:
            L.append(f"   В базе не указано: {', '.join(r['unknown'])} — оценка по этим пунктам не делалась.")
        L.append(f"   Описание из базы: «{snippet(w['description'], 240)}»")
        if w["url"]:
            L.append(f"   Карточка: {w['url']}")
    if type_rows:
        L.append("\nУместность типов вина к этой ситуации (по 5 лучшим винам каждого типа в каталоге):")
        for t in type_rows[:8]:
            name = f"{t['cat'].lower()}{' игристое' if t['sparkling'] else ''} {t['sweet']}"
            why = ""
            if t["verdict"] != "подходит" and t["worst"]:
                why = {"sweet": " (по сладости уступает блюду)", "weight": " (не совпадает по «весу»)",
                       "tannin": " (танины конфликтуют с блюдом)", "fat_acid": " (мало кислотности для жирного)"}.get(t["worst"], "")
            L.append(f"  • {name}: {t['verdict']}{why}")
    if "budget_unsupported" in ctx.notes:
        L.append("\nБюджет учесть не могу: в базе нет цен и наличия.")
    return "\n".join(L)


def render_profile_comparison(pa, pb):
    def fmt_mean(t, words):
        v, n = t
        return "нет данных" if v is None else f"{words(v)} (по {n} описаниям)"
    L = [f"Сравнение по описаниям каталога: «{pa['label']}» ({pa['n']} вин) и «{pb['label']}» ({pb['n']} вин)."]
    rows = [
        ("тело", fmt_mean(pa["body"], body_words), fmt_mean(pb["body"], body_words)),
        ("кислотность", fmt_mean(pa["acidity"], acid_words), fmt_mean(pb["acidity"], acid_words)),
        ("танины", f"упомянуты у {pa['tannin_mention']:.0%}; " + fmt_mean(pa["tannin"], tannin_words),
                   f"упомянуты у {pb['tannin_mention']:.0%}; " + fmt_mean(pb["tannin"], tannin_words)),
        ("сладкие/полусладкие", f"{pa['sweet_share']:.0%}", f"{pb['sweet_share']:.0%}"),
        ("дубовые/бочковые тона", f"{pa['oak_share']:.0%}", f"{pb['oak_share']:.0%}"),
    ]
    for name, a, b in rows:
        L.append(f"  • {name}: {a}  ↔  {b}")
    tags = set(pa["aroma_share"]) | set(pb["aroma_share"])
    diffs = sorted(((pa["aroma_share"].get(t, 0) - pb["aroma_share"].get(t, 0), t) for t in tags), key=lambda x: -abs(x[0]))
    diffs = [(d, t) for d, t in diffs if abs(d) >= 0.10][:4]
    if diffs:
        L.append("  • ароматы, которые встречаются заметно чаще: " + "; ".join(
            f"«{t}» — у {pa['label'] if d > 0 else pb['label']} (+{abs(d) * 100:.0f} п.п.)" for d, t in diffs))
    else:
        L.append("  • по ароматам описания этих групп почти не различаются.")
    same = [n for n, a, b in rows if a.split("(")[0] == b.split("(")[0]]
    if len(same) >= 3:
        L.append("Вывод по базе: заметных различий по указанным признакам немного; сильнее различаются отдельные вина внутри каждой группы.")
    small = [p["label"] for p in (pa, pb) if p["n"] < 10]
    if small:
        L.append(f"Внимание: по «{', '.join(small)}» в каталоге мало вин — выводы ориентировочные.")
    return "\n".join(L)


def render_profile(p):
    L = [f"«{p['label']}» — {p['n']} вин в каталоге."]
    if p["body"][0] is not None:
        L.append(f"  • тело по описаниям: {body_words(p['body'][0])}")
    if p["acidity"][0] is not None:
        L.append(f"  • кислотность: {acid_words(p['acidity'][0])}")
    if p["tannin_mention"] >= 0.05:
        L.append(f"  • танины упомянуты в {p['tannin_mention']:.0%} описаний" + (f", чаще {tannin_words(p['tannin'][0])}" if p["tannin"][0] else ""))
    else:
        L.append("  • танины в описаниях практически не упоминаются")
    L.append(f"  • сухие: {p['dry_share']:.0%}, сладкие и полусладкие: {p['sweet_share']:.0%}")
    if p["top_aroma"]:
        L.append("  • типичные ароматы: " + ", ".join(p["top_aroma"]))
    if p["distinct_aroma"]:
        L.append("  • отличает от остальных: " + ", ".join(p["distinct_aroma"]))
    if p["foods"]:
        L.append("  • сами описания чаще всего называют к нему: " + ", ".join(p["foods"]))
    L.append("  • регионы: " + ", ".join(f"{r} ({n})" for r, n in p["regions"]))
    L.append("  • примеры: " + "; ".join(p["examples"]))
    if p["n"] < 10:
        L.append("  Внимание: вин мало — выводы ориентировочные.")
    return "\n".join(L)


# ---- структурированная карточка вина (для интерфейса сканера) ----------------------------------------
# Разбор описания из базы на смысловые блоки — только перегруппировка фраз карточки, без добавления фактов.
DESC_SECTIONS = (
    ("color", "Цвет", r"\bцвет\w*|\bокрас\w*"),
    ("aroma", "Аромат", r"аромат\w*|\bбукет\w*|\bв носу\b"),
    ("taste", "Вкус", r"вкус\w*|послевкус\w*|танин\w*|текстур\w*|\bфиниш\w*|\bфинал\w*|\bво рту\b"),
    ("food", "Гастрономия", r"гастроном\w*|сочета\w*|аперитив\w*|подава\w*|\bподойд\w*|\bподходит к\b|\bк (?:мяс|рыб|сыр|птиц|блюд|дич|морепр|десерт|паст|закуск|салат|фрукт|шашлык|стейк)\w*"),
    ("tech", "Технология", r"винификац\w*|фермент\w*|мацерац\w*|\bвыдерж\w*|\bбочк\w*|купаж\w*|\bурожа\w*|\bсбор\w*|брожен\w*|\bсусл\w*|\bлоз\w*|виноградник\w*|\bразлит\w*"),
)
DESC_TECH = re.compile(r"^\s*(технолог\w*|техника винификации|особенности технологии|технологические операции|технология производства)\s*:")
DESC_OTHER = re.compile(r"^\s*(награды|сортовой состав|сорта винограда|тираж|декантац\w*)\s*:")
DESC_LABEL = re.compile(r"^\s*(вкус и аромат|вкус|аромат|цвет|сочетания|гастрономия|финиш|послевкусие)\s*:\s*", re.I)
SCALE_WORDS = {
    "sweetness": ("Сладость", "сухое", "сладкое"),
    "body": ("Тело", "лёгкое", "плотное"),
    "tannin": ("Танины", "мягкие", "выраженные"),
    "acidity": ("Кислотность", "низкая", "высокая"),
}


def split_description(desc):
    """Описание из базы -> [{key, title, text}] в порядке цвет/аромат/вкус/гастрономия/о вине. Фразы не меняются."""
    # «аромат …; вкус …» — разные блоки, поэтому режем и по «;»
    sentences = [s.strip() for s in re.split(r"(?<=[.!?…;])\s+|\n+", desc or "") if s.strip()]
    buckets, last = {}, "other"
    for s in sentences:
        low = norm(s)
        hits = [(m.start(), key) for key, _, rx in DESC_SECTIONS for m in [re.search(rx, low)] if m]
        key = "tech" if DESC_TECH.match(low) else "other" if DESC_OTHER.match(low) else min(hits)[1] if hits else last
        s = DESC_TECH.sub("", s)
        s = DESC_LABEL.sub("", s).strip().rstrip(";,").strip()
        if s:
            s = s[:1].upper() + s[1:]
            buckets.setdefault(key, []).append(s if s[-1] in ".!?…»\"" else s + ".")
        last = key
    titles = {key: title for key, title, _ in DESC_SECTIONS} | {"other": "О вине"}
    out = []
    for key in [k for k, _, _ in DESC_SECTIONS] + ["other"]:
        if buckets.get(key):
            text = " ".join(buckets[key])
            out.append(dict(key=key, title=titles[key], text=text))
    return out


def wine_profile(w):
    """Структура для карточки: блоки описания, шкалы 0..1, ароматы, блюда из карточки, факты подачи."""
    f = w["features"]
    scales = []
    for key, (label, left, right) in SCALE_WORDS.items():
        v = f.get(key)
        if key == "sweetness":
            pos = None if v is None else v / 3
            word = SWEET_LABEL.get(v) if v is not None else None
        else:
            pos = None if v is None else max(0.0, min(1.0, (v - 1) / 2))
            word = {"body": body_words, "tannin": tannin_words, "acidity": acid_words}[key](v)
        scales.append(dict(key=key, label=label, left=left, right=right, value=None if pos is None else round(pos, 2), word=word))
    sections = split_description(w.get("description"))
    # цвет из поля карточки — отдельным блоком с образцом, если в описании о цвете ничего нет
    if w.get("color") and len(w["color"]) <= 60 and not any(s["key"] == "color" for s in sections):
        sections.insert(0, dict(key="color", title="Цвет", text=w["color"]))
    facts = []
    if f.get("serve_temp"):
        facts.append(dict(label="Подача", value=f"{f['serve_temp'][0]}–{f['serve_temp'][1]} °C"))
    if f.get("alcohol"):
        facts.append(dict(label="Алкоголь", value=f"{f['alcohol']}%"))
    if f.get("aging_months"):
        facts.append(dict(label="Выдержка", value=f"{f['aging_months']} мес."))
    if f.get("oak"):
        facts.append(dict(label="Дуб", value="есть в описании"))
    return dict(category=w.get("category") or "", sparkling=bool(f.get("sparkling")), sections=sections,
                scales=scales, aromas=list(f.get("aroma") or []), foods=list(f.get("food_in_desc") or []), facts=facts)
