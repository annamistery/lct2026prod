"""Необязательный LLM-слой (диалог и «живые» объяснения). Движок остаётся единственным источником выбора и фактов.

Требует: pip install anthropic; переменная окружения ANTHROPIC_API_KEY. Без них Sommelier работает на шаблонах.
Проверки безопасности: (1) в ответе допустимы только названия вин из выдачи движка; (2) нет побуждающих оборотов;
(3) при нарушении — откат к шаблонному тексту. НЕ запускалось против живого API в этой среде (нет ключа).
"""
import json
import os
import re

from . import guardrails

MODEL = os.environ.get("SOMMELIER_MODEL", "claude-sonnet-5")
HERE = os.path.dirname(os.path.abspath(__file__))
SYSTEM_PROMPT = open(os.path.join(HERE, "SYSTEM_PROMPT.md"), encoding="utf-8").read()

CONTEXT_SCHEMA = """Верни ТОЛЬКО JSON: {"dish": строка|null, "occasion": "celebration|aperitif|date|business|gift|family|casual"|null,
"colors": ["Красное"|"Белое"|"Розовое"|"Оранжевое"], "sparkling": true|false|null, "sweet_ok": [0..3]|null,
"grapes_like": [..], "grapes_avoid": [..], "avoid_oak": bool, "tannin_pref": "low|high"|null, "body_pref": "light|full"|null,
"experience": "novice|enthusiast|expert", "budget_text": строка|null}. Ничего не выдумывай: нет в тексте — null/пусто."""


def _client():
    import anthropic  # ленивый импорт: пакет необязателен
    return anthropic.Anthropic()


def extract_context(text, history=""):
    """Слой 1 через LLM: свободный текст -> поля Context. Блюдо всё равно разбирается dishes.parse_dish."""
    msg = _client().messages.create(model=MODEL, max_tokens=500, system=CONTEXT_SCHEMA,
                                    messages=[{"role": "user", "content": f"{history}\n{text}".strip()}])
    raw = msg.content[0].text
    return json.loads(re.search(r"\{.*\}", raw, re.S).group(0))


def _payload(result):
    picks = []
    for p in result["picks"]:
        picks.append(dict(name=p["name"], role=p["role"], score=p["score"], unknown=p["unknown"],
                          rule_contributions=[dict(rule=c["rule"], value=round(c["value"], 2), weight=c["weight"], note=c["msg"])
                                              for c in p["reasons"]]))
    return dict(CANDIDATES=picks, TYPE_FIT=result.get("types", [])[:8], CONTEXT=result["context"])


def polish(result, catalog_by_id, level="novice"):
    """Переписывает шаблонный ответ живым языком. Возвращает текст; при любом нарушении — исходный шаблон."""
    base = result["text"]
    try:
        cands = []
        for p in result["picks"]:
            w = catalog_by_id[p["id"]]
            cands.append(dict(name=w["name"], category=w["category"], grape=w["grape"], region=w["region"], winery=w["winery"],
                              features=w["features"], description=w["description"]))
        payload = _payload(result) | {"CANDIDATES_FACTS": cands}
        msg = _client().messages.create(
            model=MODEL, max_tokens=1200, system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"Уровень гостя: {level}.\n"
                       f"Данные движка (единственный источник фактов):\n{json.dumps(payload, ensure_ascii=False)}\n\n"
                       "Сформулируй ответ по правилам системного промпта."}])
        out = msg.content[0].text
    except Exception:
        return base
    allowed = {c["name"] for c in cands}
    if not all(n in out for n in allowed if n):                     # все вина выдачи должны быть названы
        return base
    if guardrails.check_tone(out):                                   # побуждающая лексика
        return base
    if re.search(r"\b\d{3,6}\s*(руб|₽)", out):                       # цен в базе нет
        return base
    return out
