"""Answer zones of the cascade: found / probable / not_in_catalog.

similarity — SigLIP 2 cosine similarity of the best candidate (best of the two query views);
margin — fusion score gap between the first and the second candidate.

* not_in_catalog: similarity < 0.55. Raised from 0.485 after the manual check of real photos
  (report 2026-09-29): false matches on wines not in the catalog drop from 19 to 10 of 39,
  at the cost of 1 of 73 correct in-catalog answers (similarity 0.505).
* found: similarity and margin above every not-in-catalog validation photo — the top-1 was
  always right there.
* probable: everything in between — the best candidate is shown, but the user should check it
  (same-producer twins, other vintages, wines of the same series that are not in the catalog).
"""

from __future__ import annotations

from dataclasses import dataclass

FOUND = "found"
PROBABLE = "probable"
NOT_IN_CATALOG = "not_in_catalog"

MESSAGES = {
    FOUND: "Вино найдено в каталоге",
    PROBABLE: "Похоже на это вино — сверьте название, год и цвет с этикеткой",
    NOT_IN_CATALOG: "Данного вина нет в каталоге",
}


@dataclass(frozen=True)
class RecognitionThresholds:
    found_min_similarity: float = 0.83
    found_min_margin: float = 0.15
    reject_below_similarity: float = 0.55


def classify(similarity: float | None, margin: float | None, thresholds: RecognitionThresholds) -> tuple[str, str]:
    """Returns (status, human-readable reason)."""
    if similarity is None:
        return PROBABLE, "SigLIP 2 недоступен: ответ только по DINOv2, уверенность не оценивается"
    if similarity < thresholds.reject_below_similarity:
        return NOT_IN_CATALOG, (
            f"Сходство с лучшим вином каталога {similarity:.3f} < {thresholds.reject_below_similarity:.3f}: "
            "ниже порога «нет в каталоге»"
        )
    gap = margin if margin is not None else float("inf")
    if similarity >= thresholds.found_min_similarity and gap >= thresholds.found_min_margin:
        return FOUND, (
            f"Сходство {similarity:.3f} ≥ {thresholds.found_min_similarity:.3f} "
            f"и отрыв от второго кандидата {gap:.3f} ≥ {thresholds.found_min_margin:.3f}"
        )
    if similarity < thresholds.found_min_similarity:
        return PROBABLE, f"Сходство {similarity:.3f} ниже порога уверенного ответа {thresholds.found_min_similarity:.3f}"
    return PROBABLE, f"Близкий второй кандидат: отрыв {gap:.3f} < {thresholds.found_min_margin:.3f}"
