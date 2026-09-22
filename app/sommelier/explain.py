"""Слой 5: объяснение на уровне гостя. Каждая фраза выводится из вкладов правил и цитат из описания вина."""
from .dishes import describe_axes
from .engine import acid_words, body_words, tannin_words
from .features import SWEET_LABEL

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
