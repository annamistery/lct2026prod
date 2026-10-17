"""Поведенческие проверки движка сомелье (вендорирован в app/sommelier, см. app/sommelier/README.md)."""
import re
import tempfile
from pathlib import Path

from app.sommelier import guardrails
from app.sommelier.llm import GeminiLLM
from app.sommelier.sommelier import Sommelier

# feedback_log уводим во временный каталог, чтобы тесты не писали в рабочее дерево репозитория.
S = Sommelier(feedback_log=str(Path(tempfile.gettempdir()) / "lct2026prod_sommelier_test_feedback.jsonl"))
IDS = {w["id"] for w in S.wines}


def picks(q, fresh=True):
    if fresh:
        S.reset()
    r = S.ask(q)
    assert r["kind"] == "recommendation", r
    return r


def feats(r):
    return [S.engine.by_id[p["id"]] for p in r["picks"]]


def test_only_catalog_wines():
    for q in ["стейк с перечным соусом", "устрицы", "тирамису на десерт", "острый том ям", "плов на ужин"]:
        assert {p["id"] for p in picks(q)["picks"]} <= IDS


def test_dessert_not_dry():
    r = picks("к десерту, тирамису")
    assert all((w["features"]["sweetness"] or 0) >= 2 for w in feats(r)[:2]), [w["name"] for w in feats(r)]
    verdicts = {(t["cat"], t["sweet"], t["sparkling"]): t["verdict"] for t in r["types"]}
    assert verdicts.get(("Красное", "сухое", False)) == "лучше избегать"


def test_spicy_avoids_high_tannin():
    r = picks("острое тайское карри")
    for w in feats(r):
        assert (w["features"]["tannin"] or 0) < 3, w["name"]


def test_oysters_light():
    r = picks("устрицы")
    assert all((w["features"]["body"] or 0) <= 2.6 for w in feats(r)), [(w["name"], w["features"]["body"]) for w in feats(r)]


def test_hard_filters():
    r = picks("стейк, только красное, не люблю дуб, не люблю мерло")
    for w in feats(r):
        assert w["category"] == "Красное" and not w["features"]["oak"] and "Мерло" not in w["grape"]


def test_diversity():
    r = picks("стейк рибай")
    ws = feats(r)
    assert len({w["grape"] for w in ws}) >= 2 and len({w["name"] for w in ws}) == 3


def test_guardrails():
    for q, code in [("мне 16 лет, что выпить", "minor"), ("я за рулем, подбери вино", "driving"),
                    ("хочу напиться", "intoxicate"), ("не хочу вино", "decline"), ("я беременна, к рыбе?", "pregnant")]:
        r = S.ask(q)
        assert r["kind"] == "guardrail" and r["code"] == code, (q, r)


def test_tone_clean():
    for q in ["утка с вишнёвым соусом, годовщина", "устрицы", "борщ"]:
        assert not guardrails.check_tone(picks(q)["text"])


def test_budget_not_invented():
    r = picks("стейк, недорого")
    assert "нет цен" in r["text"]


def test_compare_and_profile():
    S.reset()
    r = S.ask("чем отличается Мерло от Каберне Совиньон?")
    assert r["kind"] == "comparison" and "Мерло" in r["text"] and "Каберне Совиньон" in r["text"]
    r = S.ask("расскажи про Рислинг")
    assert r["kind"] == "profile"


def test_feedback_excludes():
    S.reset()
    r = S.ask("стейк")
    first = r["picks"][0]["id"]
    S.ask("не понравилось 1")
    r2 = S.ask("стейк")
    assert first not in {p["id"] for p in r2["picks"]}


def test_about_wine_label_integration():
    """Точка интеграции с распознаванием этикеток: slug -> справка, опционально с оценкой под блюдо."""
    some_id = next(iter(IDS))
    r = S.about_wine(some_id)
    assert r["kind"] == "wine_info" and r["id"] == some_id
    wine = S.engine.by_id[some_id]
    assert r["card"]["name"] == wine["name"] and r["card"]["category"] == wine["category"]  # карточка для сканера

    r = S.about_wine("несуществующий-слаг-вина-xyz")
    assert r["kind"] == "not_found"


def test_alternatives_same_style_other_wines():
    base = next(w for w in S.wines if w["category"] == "Розовое" and not w["features"]["sparkling"] and w["features"]["sweetness"] == 0)
    r = S.alternatives(base["id"], k=5)
    assert r["kind"] == "alternatives" and r["id"] == base["id"]
    assert 0 < len(r["items"]) <= 5
    alts = [S.engine.by_id[item["id"]] for item in r["items"]]
    assert base["id"] not in {w["id"] for w in alts}
    for w in alts:                                   # жёсткие условия стиля
        assert w["category"] == base["category"] and w["features"]["sparkling"] is False
        assert w["features"]["sweetness"] is None or abs(w["features"]["sweetness"] - base["features"]["sweetness"]) <= 1
    wineries = [w["winery"] for w in alts]
    assert max(wineries.count(x) for x in wineries) <= 2  # не больше двух вин одной винодельни
    assert all(item["reasons"] for item in r["items"])  # каждый выбор объяснён


def test_alternatives_unknown_slug():
    assert S.alternatives("no-such-wine") == dict(kind="not_found", id=None, items=[])


def test_type_comparison_any_word_form_and_dish_goes_to_recommendation():
    for q in ("В чём разница между красным и белым?", "Чем розовое отличается от белого?", "Сравни сухое и сладкое"):
        S.reset()
        assert S.ask(q)["kind"] == "comparison", q
    for q in ("Что лучше к стейку: красное или белое?", "Курица с белым соусом", "Что взять к сухофруктам и сыру?"):
        S.reset()
        assert S.ask(q)["kind"] == "recommendation", q


# ---- LLM-слой (Gemini) — без сети: подменяем вызов модели ----------------------------------------
class FakeLLM(GeminiLLM):
    def __init__(self, reply):
        super().__init__("test-key")
        self.reply = reply

    def _generate(self, system, user, max_tokens, json_mode=False):
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply(user) if callable(self.reply) else self.reply


def with_llm(llm, q="стейк рибай"):
    s = Sommelier(wines=S.wines, feedback_log=S.feedback_log, llm=llm)
    return s.ask(q), Sommelier(wines=S.wines, feedback_log=S.feedback_log).ask(q)


def test_llm_answer_used_when_it_names_all_picks():
    r, base = with_llm(FakeLLM(lambda user: "Живой ответ: " + ", ".join(p["name"] for p in base_picks)))
    assert r["text"].startswith("Живой ответ:")
    assert r["text"].endswith(guardrails.NEUTRAL_FOOTER)
    assert [p["id"] for p in r["picks"]] == [p["id"] for p in base["picks"]]


def test_llm_answer_rejected_falls_back_to_template():
    for reply in ("Ответ без названий вин", OSError("network down")):
        r, base = with_llm(FakeLLM(reply))
        assert r["text"] == base["text"]


def test_llm_answers_free_question_and_falls_back_without_it():
    s = Sommelier(wines=S.wines, feedback_log=S.feedback_log, llm=FakeLLM("Танины — вещества из кожицы и косточек."))
    r = s.ask("Что такое танины?")
    assert r["kind"] == "answer" and r["text"].endswith(guardrails.INFO_FOOTER)
    r = Sommelier(wines=S.wines, feedback_log=S.feedback_log, llm=FakeLLM(OSError("down"))).ask("Что такое танины?")
    assert r["kind"] == "clarify"


def test_llm_retells_profile_and_comparison_with_fallback():
    def ok(user):
        stats = user.split("GROUP_STATS", 1)[1]
        labels = re.findall(r"«([^»]+)»", stats)
        counts = re.findall(r"(\d+) вин", stats)
        return "Живой рассказ: " + "; ".join(labels) + " — по " + ", ".join(counts) + " винам."
    for q, kind in (("Расскажи про Мерло", "profile"), ("Чем отличается Мерло от Каберне Совиньон?", "comparison")):
        base = Sommelier(wines=S.wines, feedback_log=S.feedback_log).ask(q)
        r = Sommelier(wines=S.wines, feedback_log=S.feedback_log, llm=FakeLLM(ok)).ask(q)
        assert r["kind"] == kind and r["text"].startswith("Живой рассказ:"), r["text"]
        assert r["text"].endswith(guardrails.INFO_FOOTER)
        bad = Sommelier(wines=S.wines, feedback_log=S.feedback_log, llm=FakeLLM("Рассказ без названий и чисел")).ask(q)
        assert bad["text"] == base["text"]


base_picks = Sommelier(wines=S.wines, feedback_log=S.feedback_log).ask("стейк рибай")["picks"]


def test_wine_profile_splits_description_without_inventing_text():
    from app.sommelier.explain import split_description
    blocks = split_description("Аромат с нотами вишни; вкус округлый, с мягкими танинами. К мясным блюдам. Награды: золото")
    assert [b["key"] for b in blocks] == ["aroma", "taste", "food", "other"]
    w = S.engine.by_id["belbek-belbek-merlo-krasnoe-suhoe-129"]
    p = S.about_wine(w["id"])["profile"]
    words = set(re.findall(r"\w+", (w["description"] + " " + w["color"]).lower()))  # блок «Цвет» может прийти из поля карточки
    assert all(set(re.findall(r"\w+", b["text"].lower())) <= words for b in p["sections"])
    assert {s["key"] for s in p["scales"]} == {"sweetness", "body", "tannin", "acidity"}
    assert all(s["value"] is None or 0 <= s["value"] <= 1 for s in p["scales"])


def test_pairing_infographic_matches_engine_score():
    """`pairing` для инфографики сканера: тот же балл, что у движка, вердикт по месту в каталоге, лучшие блюда."""
    from app.sommelier.dishes import parse_dish
    from app.sommelier.engine import Context
    slug = "abrau-dyurso-abrau-dyurso-pino-nuar-krasnoe-suhoe-13"
    assert S.about_wine(slug)["pairing"] is None  # без блюда — только справка

    dish = parse_dish("стейк рибай")
    p = S.about_wine(slug, dish, "стейк рибай")["pairing"]
    score, comps, _ = S.engine.score(S.engine.by_id[slug], Context(dish=dish))
    assert p["score"] == score and p["dish"]["text"] == "стейк рибай"
    assert 0 <= p["beats"] <= 1 and p["best"] >= score
    assert len(p["factors"]) == len(comps) and all(f["sign"] in ("plus", "minus", "neutral") for f in p["factors"])
    assert all(f["quote"] is None or "«" not in f["text"][-3:] for f in p["factors"])  # цитата вынесена из довода
    assert p["better_dishes"] and all(d["label"] != "Стейк" for d in p["better_dishes"])  # текущее блюдо не предлагается
    assert p["better_dishes"] == sorted(p["better_dishes"], key=lambda d: -d["score"])

    # сухое вино к сладкому десерту — главный конфликт пары, вердикт не мягче «не лучший выбор»
    assert S.about_wine(slug, parse_dish("тирамису"), "тирамису")["pairing"]["verdict"]["level"] == "poor"


def test_recommendation_picks_carry_display_card():
    """Карточки вин в чате: факты карточки, доводы без цитат (те же, что в тексте), короткая цитата описания."""
    r = picks("утка с вишнёвым соусом")
    assert r["dish_traits"] and r["note"] == guardrails.NEUTRAL_FOOTER and r["total"] == len(S.wines)
    for p in r["picks"]:
        w = S.engine.by_id[p["id"]]
        assert p["category"] == w["category"] and p["grape"] == (w["grape"] or "") and p["url"] == w["url"]
        assert p["pros"] and all("«" not in line[-2:] for line in p["pros"] + p["cons"])
        assert p["quote"] is None or p["quote"].split("…")[0][:40] in w["description"]


def test_about_wine_passport_fields():
    """«О вине»: ссылка на карточку, лучшие типовые блюда по убыванию балла, пометка-справка."""
    slug = "abrau-dyurso-abrau-dyurso-pino-nuar-krasnoe-suhoe-13"
    r = S.about_wine(slug)
    assert r["card"]["url"] == S.engine.by_id[slug]["url"] and r["note"] == guardrails.INFO_FOOTER
    assert len(r["best_dishes"]) == 4 and r["best_dishes"] == sorted(r["best_dishes"], key=lambda d: -d["score"])
