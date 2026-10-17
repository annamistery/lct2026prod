"""Необязательный LLM-слой (живые формулировки ответа) на Google Gemini. Движок остаётся единственным источником выбора и фактов.

Требует: переменная окружения GOOGLE_API_KEY (модель — SOMMELIER_LLM_MODEL). Без ключа Sommelier работает на шаблонах.
Вызов — через REST API Gemini стандартной библиотекой (без дополнительных пакетов).
Проверки безопасности: (1) в ответе должны быть названы все вина из выдачи движка; (2) нет побуждающих оборотов;
(3) нет цен; (4) при любой ошибке или нарушении — откат к шаблонному тексту.
"""
import json
import logging
import os
import re
import urllib.error
import urllib.request

from . import guardrails
from .features import norm

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-flash-latest"
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
HERE = os.path.dirname(os.path.abspath(__file__))
SYSTEM_PROMPT = open(os.path.join(HERE, "SYSTEM_PROMPT.md"), encoding="utf-8").read()

CONTEXT_SCHEMA = """Верни ТОЛЬКО JSON: {"dish": строка|null, "occasion": "celebration|aperitif|date|business|gift|family|casual"|null,
"colors": ["Красное"|"Белое"|"Розовое"|"Оранжевое"], "sparkling": true|false|null, "sweet_ok": [0..3]|null,
"grapes_like": [..], "grapes_avoid": [..], "avoid_oak": bool, "tannin_pref": "low|high"|null, "body_pref": "light|full"|null,
"experience": "novice|enthusiast|expert", "budget_text": строка|null}. Ничего не выдумывай: нет в тексте — null/пусто."""


ANSWER_PROMPT = """Ты — AI-сомелье сервиса с каталогом российских вин. Отвечаешь на русском, коротко (до 8 предложений), без markdown.
Что умеешь в сервисе: подобрать вино из каталога к блюду или поводу (гость пишет блюдо/повод и пожелания: цвет, сухое/сладкое,
что не любит), сравнить сорта или типы вина, рассказать о сорте по данным каталога, дать справку по вину с распознанной этикетки.
Правила:
- Отвечай на общие вопросы о вине (термины, подача, хранение, температура, бокалы, сочетания в целом) по общеизвестным фактам.
- Не называй конкретные вина, винодельни, бренды и цены — подбор конкретных вин делает каталог; если гость хочет конкретику,
  предложи написать блюдо или повод.
- Тон информирующий, без побуждений пить («обязательно попробуйте», «порадуйте себя» и т. п.), без превосходных степеней.
- Не помогаешь несовершеннолетним, беременным, водителям, людям на лекарствах или в зависимости; не советуешь количество.
- Вопросы не про вино, еду и напитки — вежливо скажи, что ты сомелье, и предложи, с чем можешь помочь.
- Не добавляй дисклеймер в конце — его добавит система."""


class GeminiLLM:
    def __init__(self, api_key, model=DEFAULT_MODEL, timeout=20.0):
        self.api_key = api_key
        self.model = model or DEFAULT_MODEL
        self.timeout = timeout

    def _generate(self, system, user, max_tokens, json_mode=False):
        # Токены «размышлений» Gemini считаются в maxOutputTokens: без ограничения ответ обрезается и идёт ~30 с.
        # Модели 3.x принимают thinkingLevel; у 2.x его нет — им просто оставляем запас по токенам.
        cfg = {"maxOutputTokens": max_tokens + 2000, "temperature": 0.3}
        if not self.model.startswith("gemini-2"):
            cfg["thinkingConfig"] = {"thinkingLevel": "low"}
        if json_mode:
            cfg["responseMimeType"] = "application/json"
        body = {"systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": cfg}
        req = urllib.request.Request(API_URL.format(model=self.model), data=json.dumps(body).encode("utf-8"), method="POST",
                                     headers={"Content-Type": "application/json", "x-goog-api-key": self.api_key})
        for attempt in (1, 2):                       # один повтор при сетевом сбое (не при ответе API с ошибкой)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.load(resp)
                break
            except urllib.error.HTTPError:
                raise
            except (urllib.error.URLError, TimeoutError):
                if attempt == 2:
                    raise
        parts = data["candidates"][0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts if not p.get("thought")).strip()

    def extract_context(self, text, history=""):
        """Слой 1 через LLM: свободный текст -> поля Context. Блюдо всё равно разбирается dishes.parse_dish."""
        raw = self._generate(CONTEXT_SCHEMA, f"{history}\n{text}".strip(), 500, json_mode=True)
        return json.loads(re.search(r"\{.*\}", raw, re.S).group(0))

    def answer(self, question, dialog_context=""):
        """Свободный ответ на вопрос, который движок не разобрал (не подбор). None — если LLM недоступна или ответ не прошёл проверки."""
        user = (f"Контекст диалога: {dialog_context}\n\n" if dialog_context else "") + f"Вопрос гостя: {question}"
        try:
            out = self._generate(ANSWER_PROMPT, user, 800)
        except (urllib.error.URLError, TimeoutError, OSError, KeyError, IndexError, ValueError) as exc:
            logger.warning("sommelier LLM answer failed, using template: %s", exc)
            return None
        if not out or guardrails.check_tone(out) or re.search(r"\b\d{3,6}\s*(руб|₽)", out):
            logger.info("sommelier LLM free answer rejected")
            return None
        return out

    def retell(self, base, profiles, level="novice"):
        """Пересказ профиля сорта/типа или сравнения двух (GROUP_STATS из шаблона). При любом нарушении — исходный шаблон.

        profiles — словари engine.group_profile: в ответе обязаны остаться название каждой группы и число вин, по которым сделан вывод.
        """
        try:
            out = self._generate(
                SYSTEM_PROMPT,
                f"Уровень гостя: {level}.\n"
                f"GROUP_STATS (статистика по описаниям каталога, единственный источник фактов):\n{base}\n\n"
                "Перескажи это живым языком по правилам системного промпта: что это значит для вкуса и с чем такое вино обычно "
                "сочетается по данным выше. Не добавляй фактов, которых нет в GROUP_STATS, доли и числа не меняй. "
                "Названия сортов пиши точно как в GROUP_STATS; типы (красное, белое, сухое…) склоняй по-русски и без кавычек: «у красных вин». Обязательно укажи, по скольким винам сделан вывод. "
                "Примеры вин называй только из GROUP_STATS. Нейтральную пометку в конце не добавляй. Без markdown-разметки.",
                1200)
        except (urllib.error.URLError, TimeoutError, OSError, KeyError, IndexError, ValueError) as exc:
            logger.warning("sommelier LLM retell failed, using template: %s", exc)
            return base
        if not out:
            return base
        nout = norm(out)
        if not all(_label_in(p["label"], nout) and re.search(rf"(?<!\d){p['n']}(?!\d)", out) for p in profiles):
            logger.info("sommelier LLM retell rejected: group name or wine count missing")
            return base
        if guardrails.check_tone(out) or re.search(r"\b\d{3,6}\s*(руб|₽)", out):
            logger.info("sommelier LLM retell rejected: tone/price")
            return base
        return out

    def polish(self, result, catalog_by_id, level="novice"):
        """Переписывает шаблонный ответ живым языком. Возвращает текст; при любом нарушении — исходный шаблон."""
        base = result["text"]
        try:
            cands = []
            for p in result["picks"]:
                w = catalog_by_id[p["id"]]
                cands.append(dict(name=w["name"], category=w["category"], grape=w["grape"], region=w["region"], winery=w["winery"],
                                  features=w["features"], description=w["description"]))
            payload = _payload(result) | {"CANDIDATES_FACTS": cands}
            out = self._generate(
                SYSTEM_PROMPT,
                f"Уровень гостя: {level}.\n"
                f"Данные движка (единственный источник фактов):\n{json.dumps(payload, ensure_ascii=False, default=str)}\n\n"
                f"Шаблонный ответ движка (перескажи его живым языком, не меняя фактов и порядка вин):\n{base}\n\n"
                "Сформулируй ответ по правилам системного промпта. Названия вин пиши точно как в CANDIDATES. "
                "Нейтральную пометку в конце не добавляй — её добавит система. Без markdown-разметки.",
                1500)
        except (urllib.error.URLError, TimeoutError, OSError, KeyError, IndexError, ValueError) as exc:
            logger.warning("sommelier LLM call failed, using template: %s", exc)
            return base
        if not out:
            return base
        nout = norm(out)
        if not all(norm(c["name"]) in nout for c in cands if c["name"]):   # все вина выдачи должны быть названы
            logger.info("sommelier LLM answer rejected: not all picked wines named")
            return base
        if guardrails.check_tone(out):                                       # побуждающая лексика
            logger.info("sommelier LLM answer rejected: tone")
            return base
        if re.search(r"\b\d{3,6}\s*(руб|₽)", out):                           # цен в базе нет
            logger.info("sommelier LLM answer rejected: price")
            return base
        return out


def _label_in(label, nout):
    """Сорт («Мерло») — дословно; тип вина из нижнего регистра («красное») — по основе, в любой форме: «красных», «белым»."""
    lab = norm(label)
    if label != lab:
        return lab in nout
    return re.search(r"\b" + re.escape(lab[:-2]) + r"(ое|ого|ому|ым|ом|ые|ых|ыми|ая|ой|ую|ие|им|ими|их)\b", nout) is not None


def _payload(result):
    picks = []
    for p in result["picks"]:
        picks.append(dict(name=p["name"], role=p["role"], score=p["score"], unknown=p["unknown"],
                          rule_contributions=[dict(rule=c["rule"], value=round(c["value"], 2), weight=c["weight"], note=c["msg"])
                                              for c in p["reasons"]]))
    return dict(CANDIDATES=picks, TYPE_FIT=result.get("types", [])[:8], CONTEXT=result["context"])
