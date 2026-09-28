"""Оркестратор: guardrails -> интент -> разбор -> движок -> объяснение. Диалог с памятью и обратной связью."""
import difflib
import json
import logging
import os
import re
import time

from . import catalog, guardrails
from .context import Parser
from .engine import Context, Engine
from .explain import render_profile, render_profile_comparison, render_recommendation
from .features import norm

logger = logging.getLogger(__name__)

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_FEEDBACK_LOG = os.path.join(HERE, "data", "feedback.jsonl")

COMPARE_RX = r"(чем отличается|разниц\w+ между|отличи\w+|сравни|в чем разница|в чём разница|или)\b"
DESCRIBE_RX = r"(расскажи|что такое|что за|опиши|какой вкус у|чем характеризуется|что известно про|что за сорт)"
GRAPE_ALIASES = {"игристое": ("sparkling", True), "красное": ("cat", "Красное"), "белое": ("cat", "Белое"),
                 "розовое": ("cat", "Розовое"), "оранжевое": ("cat", "Оранжевое"), "сладкое": ("sweet", {2, 3}), "сухое": ("sweet", {0})}


class Sommelier:
    def __init__(self, wines=None, feedback_log=None):
        self.wines = wines or catalog.load()
        self.engine = Engine(self.wines)
        self.parser = Parser(self.wines)
        self.ctx = Context()
        self.last = None            # последняя выдача (для обратной связи)
        # По умолчанию — рядом с кодом (удобно для CLI); сервис передаёт путь в writable-каталог,
        # так как в проде app/ смонтирован read-only.
        self.feedback_log = feedback_log or DEFAULT_FEEDBACK_LOG

    # ---- публичный вход -------------------------------------------------------------------------
    def reset(self):
        self.ctx = Context()
        self.last = None

    def ask(self, text):
        code = guardrails.check(text)
        if code:
            return dict(kind="guardrail", code=code, text=guardrails.MESSAGES[code])
        t = norm(text)
        fb = self._feedback(t, text)
        if fb:
            return fb
        if re.search(r"этикетк|распознан|на фото|по фото", t) and self._find_wine(text):
            return self.about_wine(text, self._dish_from(text))
        grapes = self.parser.grapes_in(t)
        if re.search(COMPARE_RX, t) and (len(grapes) >= 2 or len(self._groups_in(t)) >= 2):
            return self.compare(text)
        if re.search(DESCRIBE_RX, t) and (grapes or self._groups_in(t)) and not self._dish_from(text):
            return self.describe(text)
        return self.recommend(text)

    # ---- рекомендации ---------------------------------------------------------------------------
    def _dish_from(self, text):
        from .dishes import parse_dish
        return parse_dish(text)

    def recommend(self, text, k=3):
        new = self.parser.parse(text)
        if not (new.dish or new.occasion or new.colors or new.sparkling is not None or new.sweet_ok
                or new.grapes_like or new.regions or new.avoid_oak or new.tannin_pref or new.body_pref):
            if self.ctx.dish is None:
                return dict(kind="clarify", text="Уточните, пожалуйста: к какому блюду или по какому поводу подбираем "
                            "и есть ли пожелания (цвет, сухое/сладкое, что не любите)? Также могу объяснить разницу между сортами.")
        self.ctx.merge(new)
        ctx = self.ctx
        rec = self.engine.recommend(ctx, k)
        # уточняющий вопрос вместо выдачи «вслепую»: нет ни блюда, ни повода, ни предпочтений по стилю
        types = self.engine.type_fit(ctx) if ctx.dish else []
        self.last = rec
        text_out = render_recommendation(rec, ctx, types, ctx.experience)
        tone = guardrails.check_tone(text_out)
        assert not tone, tone
        text_out += "\n\n" + guardrails.NEUTRAL_FOOTER
        return dict(kind="recommendation", text=text_out, context=_ctx_dict(ctx),
                    picks=[dict(id=r["wine"]["id"], name=r["wine"]["name"], score=r["score"], role=r["role"],
                                unknown=r["unknown"], reasons=[c for c in r["comps"]]) for r in rec["picks"]],
                    types=types, pool=rec["pool_size"])

    # ---- вино с этикетки: «что это и с чем сочетается» -----------------------------------------------
    def _find_wine(self, text, cutoff=0.6):
        t = norm(text)
        names = {norm(w["name"]): w for w in self.wines}
        hits = [w for n, w in names.items() if len(n) > 5 and n in t]
        if hits:
            return max(hits, key=lambda w: len(w["name"]))
        m = difflib.get_close_matches(re.sub(r"[^\w\s]", "", t), list(names), n=1, cutoff=cutoff)
        return names[m[0]] if m else None

    def about_wine(self, text_or_id, dish=None):
        """Точка интеграции с распознаванием этикеток: получает slug/название, возвращает справку по вину."""
        w = self.engine.by_id.get(text_or_id) or self._find_wine(text_or_id)
        if not w:
            return dict(kind="not_found", text="Такого вина в базе нет — ничего сказать о нём не могу.")
        f = w["features"]
        from .explain import snippet, wine_facts
        L = [f"{w['name']}", wine_facts(w, "expert"), f"Описание из базы: «{snippet(w['description'], 700)}»"]
        if f["food_in_desc"]:
            L.append("Блюда, которые упоминает сама карточка: " + ", ".join(f["food_in_desc"]))
        if dish:
            ctx = Context(dish=dish)
            s, comps, unk = self.engine.score(w, ctx)
            from .explain import reason_line
            pros = [reason_line(c, w, ctx, "novice") for c in comps if c["value"] > 0.15]
            cons = [reason_line(c, w, ctx, "novice") for c in comps if c["value"] < -0.15]
            L.append(f"К этому блюду: оценка по правилам {s}/100.")
            if pros:
                L.append("  за: " + "; ".join(pros))
            if cons:
                L.append("  против: " + "; ".join(cons))
            if unk:
                L.append(f"  в базе не указано: {', '.join(unk)}")
        L.append("\n" + guardrails.NEUTRAL_FOOTER)
        card = dict(name=w["name"], winery=w.get("winery") or "", category=w.get("category") or "",
                    color=w.get("color") or "", region=w.get("region") or "", grape=w.get("grape") or "")
        return dict(kind="wine_info", id=w["id"], text="\n".join(L), card=card)

    def alternatives(self, slug, k=5):
        """Точка интеграции с распознаванием: вина каталога, похожие по стилю на распознанное (slug)."""
        w = self.engine.by_id.get(slug)
        if not w:
            return dict(kind="not_found", id=None, items=[])
        from .explain import wine_facts
        items = [
            dict(id=p["wine"]["id"], name=p["wine"]["name"], winery=p["wine"]["winery"],
                 facts=wine_facts(p["wine"], "novice"), score=p["score"], reasons=p["reasons"])
            for p in self.engine.similar(w, k)
        ]
        return dict(kind="alternatives", id=w["id"], items=items)

    # ---- различия между сортами / типами --------------------------------------------------------------
    def _groups_in(self, t):
        out = []
        for g in self.parser.grapes_in(t):
            out.append(("grape", g))
        for word, _spec in GRAPE_ALIASES.items():
            if re.search(word[:5], t) and not self.parser.grapes_in(t):
                out.append(("alias", word))
        return out

    def _group(self, kind, key):
        if kind == "grape":
            ws = [w for w in self.wines if key in self.engine.grapes_of(w)]
            return self.engine.group_profile(ws, key)
        typ, val = GRAPE_ALIASES[key]
        if typ == "sparkling":
            ws = [w for w in self.wines if w["features"]["sparkling"]]
        elif typ == "cat":
            ws = [w for w in self.wines if w["category"] == val and not w["features"]["sparkling"]]
        else:
            ws = [w for w in self.wines if w["features"]["sweetness"] in val]
        return self.engine.group_profile(ws, key)

    def compare(self, text):
        t = norm(text)
        groups = self._groups_in(t)
        # «Рислинг» может дать и «Рислинг Рейнский» — берём максимум два самых частотных разных сорта
        groups = sorted(groups, key=lambda g: -self.engine.grape_count.get(g[1], 0))[:2]
        if len(groups) < 2:
            return dict(kind="clarify", text="Назовите два сорта или типа вина для сравнения, например «Мерло и Каберне Совиньон».")
        pa, pb = self._group(*groups[0]), self._group(*groups[1])
        return dict(kind="comparison", text=render_profile_comparison(pa, pb) + "\n\n" + guardrails.INFO_FOOTER)

    def describe(self, text):
        t = norm(text)
        groups = self._groups_in(t)
        if not groups:
            return dict(kind="clarify", text="О каком сорте или типе вина рассказать?")
        gs = sorted(groups, key=lambda g: -self.engine.grape_count.get(g[1], 0))[:1]
        return dict(kind="profile", text=render_profile(self._group(*gs[0])) + "\n\n" + guardrails.INFO_FOOTER)

    # ---- обратная связь ---------------------------------------------------------------------------
    def _feedback(self, t, raw):
        if not self.last or not self.last["picks"]:
            return None
        m = re.search(r"(не понравил\w*|понравил\w*|не подошл\w*|подошл\w*|беру|возьму)\s*(?:вариант\s*)?(\d)?", t)
        if not m or not re.match(r"^\s*(да|нет|мне|это|вариант|номер|первое|второе|третье|не|понравил|подошл|беру|возьму|\d)", t):
            return None
        neg = m.group(1).startswith("не")
        idx = int(m.group(2)) - 1 if m.group(2) else None
        picks = self.last["picks"]
        targets = [picks[idx]] if idx is not None and 0 <= idx < len(picks) else picks
        try:
            os.makedirs(os.path.dirname(self.feedback_log), exist_ok=True)
            with open(self.feedback_log, "a", encoding="utf-8") as fh:
                for r in targets:
                    fh.write(json.dumps(dict(ts=int(time.time()), wine=r["wine"]["id"], liked=not neg,
                                             dish=self.ctx.dish_text, score=r["score"]), ensure_ascii=False) + "\n")
        except OSError:
            logger.warning("could not write sommelier feedback log at %s", self.feedback_log, exc_info=True)
        if neg:
            self.ctx.excluded_ids |= {r["wine"]["id"] for r in targets}
            return dict(kind="feedback", text="Учёл: эти вина больше не предлагаю. Скажите, что не подошло (слишком сладкое, "
                        "плотное, кислое) — и я сдвину подбор.")
        return dict(kind="feedback", text="Записал вашу оценку. Она нужна только для настройки подбора.")

    def feedback_stats(self):
        return self.ctx


def _ctx_dict(c):
    d = {k: (sorted(v) if isinstance(v, set) else v) for k, v in vars(c).items() if k != "dish"}
    d["dish"] = c.dish
    return d
