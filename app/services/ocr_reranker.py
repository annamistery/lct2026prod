"""OCR-based second-stage reranker for search pipeline v4.

Algorithm
---------
1. Run PaddleOCR (preferred) or EasyOCR on the query crop to extract text.
2. Extract a vintage year YYYY (1950–2029) from the OCR text via regex.
3. For each candidate, derive vintage from Product.description via the same regex.
4. Score each candidate:
       FinalScore = w_sim * cosine_sim
                  + w_vintage * vintage_match   (1.0 if years match, 0.0 otherwise)
                  + w_text    * text_overlap     (normalised Jaccard of OCR tokens)
5. Re-sort candidates by FinalScore descending.

If neither PaddleOCR nor EasyOCR is installed the reranker raises ImportError at
construction time; lifespan.py catches this and runs v4 without OCR reranking
(only vector similarity determines order).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from PIL import Image

if TYPE_CHECKING:
    from app.db.repositories.products import ProductCandidateV4

_YEAR_RE = re.compile(r"\b(1[89]\d{2}|20[012]\d)\b")


def _extract_vintage(text: str) -> str | None:
    """Return the first 4-digit year in [1800, 2029] found in text, or None."""
    m = _YEAR_RE.search(text)
    return m.group(0) if m else None


def _tokenise(text: str) -> set[str]:
    """Lower-case alphanumeric tokens, length ≥ 2."""
    return {t for t in re.findall(r"[a-zа-яё0-9]{2,}", text.lower()) if t}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _ocr_text(image: Image.Image, engine: str) -> str:
    """Extract all text from image using the chosen OCR engine."""
    if engine == "paddle":
        from paddleocr import PaddleOCR  # type: ignore[import]

        ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
        result = ocr.ocr(image, cls=True)
        texts: list[str] = []
        if result:
            for line_group in result:
                if line_group:
                    for item in line_group:
                        if item and len(item) >= 2 and item[1]:
                            texts.append(str(item[1][0]))
        return " ".join(texts)

    if engine == "easyocr":
        import easyocr  # type: ignore[import]
        import numpy as np

        reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        result = reader.readtext(np.array(image))
        return " ".join(item[1] for item in result)

    raise RuntimeError(f"Unknown OCR engine: {engine!r}")


@dataclass
class OcrReranker:
    """Reranks top-K candidates using OCR vintage & text matching."""

    w_sim: float = 0.5
    w_vintage: float = 0.3
    w_text: float = 0.2

    def __post_init__(self) -> None:
        # Probe which engine is available; raises ImportError if neither is present
        self._engine = self._detect_engine()

    @staticmethod
    def _detect_engine() -> str:
        try:
            import paddleocr  # noqa: F401
            return "paddle"
        except ImportError:
            pass
        try:
            import easyocr  # noqa: F401
            return "easyocr"
        except ImportError:
            pass
        raise ImportError(
            "OCR reranker requires paddleocr or easyocr. "
            "Install one of them, or set ENABLE_OCR_RERANK_V4=false."
        )

    def rerank(
        self,
        query_image: Image.Image,
        candidates: list[ProductCandidateV4],
        dino_similarities: list[float],
    ) -> tuple[list[ProductCandidateV4], str | None, list[float]]:
        """Return (reranked_candidates, vintage_detected, final_scores)."""
        if not candidates:
            return candidates, None, []

        try:
            query_text = _ocr_text(query_image, self._engine)
        except Exception:
            query_text = ""

        vintage_query = _extract_vintage(query_text)
        query_tokens = _tokenise(query_text)

        scored: list[tuple[float, ProductCandidateV4]] = []
        final_scores: list[float] = []

        for cand, sim in zip(candidates, dino_similarities, strict=False):
            prod_text = cand.product.description or ""
            vintage_prod = _extract_vintage(prod_text)
            prod_tokens = _tokenise(prod_text + " " + (cand.product.title or ""))

            vintage_match = (
                1.0
                if (vintage_query and vintage_prod and vintage_query == vintage_prod)
                else 0.0
            )
            text_overlap = _jaccard(query_tokens, prod_tokens)

            score = (
                self.w_sim * sim
                + self.w_vintage * vintage_match
                + self.w_text * text_overlap
            )
            scored.append((score, cand))
            final_scores.append(score)

        # Sort by score descending, preserving original order on tie
        order = sorted(range(len(scored)), key=lambda i: -scored[i][0])
        reranked = [scored[i][1] for i in order]
        reranked_scores = [scored[i][0] for i in order]

        return reranked, vintage_query, reranked_scores
