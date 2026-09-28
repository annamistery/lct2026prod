"""Поведенческие проверки движка сомелье (вендорирован в app/sommelier, см. app/sommelier/README.md)."""
import tempfile
from pathlib import Path

from app.sommelier import guardrails
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
