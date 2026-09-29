"""Reads the text of a wine label with a local vision-language model served by Ollama.

Used only for «probable» answers of the cascade (see pipelines/search/cascade/label_check.py).
Any failure (Ollama down, timeout, broken JSON) returns None: the cascade then keeps its image-only answer.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import urllib.request

from PIL import Image

logger = logging.getLogger(__name__)

PROMPT = (
    "На фото этикетка бутылки вина. Прочитай только то, что НАПИСАНО на этикетке, ничего не придумывай. "
    "Верни JSON с полями:\n"
    '"producer" — винодельня/бренд;\n'
    '"name" — название вина или линейки (без сорта и без слов про цвет/сладость), пусто если нет;\n'
    '"grapes" — список сортов винограда, написанных на этикетке (пустой список, если сорт не написан);\n'
    '"color" — "красное" | "белое" | "розовое" | "оранжевое" | "" (только если написано на этикетке);\n'
    '"sweetness" — "сухое" | "полусухое" | "полусладкое" | "сладкое" | "брют" | "экстра брют" | "" (только если написано);\n'
    '"sparkling" — true, если написано игристое/шампанское/брют/pet-nat, иначе false;\n'
    '"year" — год урожая числом или null;\n'
    '"text" — весь читаемый текст этикетки одной строкой.'
)


class LabelReader:
    def __init__(self, url: str, model: str, timeout_s: float = 20.0, max_side: int = 1024) -> None:
        self.url = url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.max_side = max_side

    async def read(self, crop: Image.Image) -> dict | None:
        image = crop.convert("RGB")
        image.thumbnail((self.max_side, self.max_side))
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=90)
        payload = {
            "model": self.model,
            "stream": False,
            "format": "json",
            "keep_alive": "60m",
            "options": {"temperature": 0, "num_ctx": 4096, "num_predict": 400},
            "messages": [{"role": "user", "content": PROMPT, "images": [base64.b64encode(buf.getvalue()).decode("ascii")]}],
        }
        try:
            answer = await asyncio.to_thread(self._post, "/api/chat", payload, self.timeout_s)
            data = json.loads(answer["message"]["content"])
            return data if isinstance(data, dict) else None
        except Exception as exc:  # the check is optional: never fail the search because of it
            logger.warning("label reader failed: %s", exc)
            return None

    def warm_up(self) -> None:
        """Load the model into memory so the first «probable» photo does not pay for it."""
        try:
            self._post("/api/generate", {"model": self.model, "keep_alive": "60m"}, 120)
        except Exception as exc:
            logger.warning("label reader warm-up failed: %s", exc)

    def _post(self, path: str, payload: dict, timeout_s: float) -> dict:
        request = urllib.request.Request(f"{self.url}{path}", data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=timeout_s) as response:  # URL comes from settings, not from requests
            return json.load(response)
