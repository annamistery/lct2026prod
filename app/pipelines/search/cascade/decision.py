from __future__ import annotations

from dataclasses import dataclass

from app.schemas.search.cascade import CascadeNeighborItem, V1DecisionDetails
from app.schemas.search.v1 import SearchResult


@dataclass
class CascadeDecisionEngine:
    confidence_margin: float = 0.05
    min_confidence_score: float = 0.65
    neighbor_window: float = 0.08
    max_neighbors: int = 5

    def evaluate(self, v1_results: list[SearchResult]) -> tuple[V1DecisionDetails, list[CascadeNeighborItem]]:
        """Evaluate whether v1 output has an unambiguous top-1 winner or requires v4 arbitration."""
        if not v1_results:
            details = V1DecisionDetails(
                is_confident=False,
                top1_similarity=0.0,
                top2_similarity=None,
                margin=None,
                reason="Кандидаты v1 не найдены в базе данных",
            )
            return details, []

        top1 = v1_results[0]
        sim1 = float(top1.dino_similarity)

        if len(v1_results) == 1:
            details = V1DecisionDetails(
                is_confident=True,
                top1_similarity=sim1,
                top2_similarity=None,
                margin=None,
                reason="Единственный кандидат в каталоге",
            )
            return details, []

        top2 = v1_results[1]
        sim2 = float(top2.dino_similarity)
        margin = round(sim1 - sim2, 4)

        # Check manufacturer conflict (wine twins from the same winery/series)
        m1 = top1.manufacturer.strip().lower() if top1.manufacturer else ""
        m2 = top2.manufacturer.strip().lower() if top2.manufacturer else ""
        same_winery = bool(m1 and m2 and m1 == m2)

        if same_winery and margin < 0.12:
            is_confident = False
            reason = (
                f"Обнаружены вина одного производителя '{top1.manufacturer}' "
                f"с близким сходством (отрыв {margin:.4f} < 0.12)"
            )
        elif margin >= self.confidence_margin and sim1 >= self.min_confidence_score:
            is_confident = True
            reason = (
                f"Ярко выраженный ТОП-1: отрыв {margin:.4f} >= {self.confidence_margin:.4f}, "
                f"сходство {sim1:.4f}"
            )
        else:
            is_confident = False
            reason = (
                f"Малый отрыв между кандидатами: {margin:.4f} < {self.confidence_margin:.4f} "
                f"(топ-1: {sim1:.4f}, топ-2: {sim2:.4f})"
            )

        details = V1DecisionDetails(
            is_confident=is_confident,
            top1_similarity=sim1,
            top2_similarity=sim2,
            margin=margin,
            reason=reason,
        )

        if is_confident:
            return details, []

        # Collect neighbor candidates for Stage 2 (v4)
        threshold = sim1 - self.neighbor_window
        neighbors: list[CascadeNeighborItem] = []
        for rank, cand in enumerate(v1_results, 1):
            cand_sim = float(cand.dino_similarity)
            cand_m = cand.manufacturer.strip().lower() if cand.manufacturer else ""
            is_close_score = cand_sim >= threshold
            is_same_brand = bool(same_winery and cand_m == m1)

            if rank <= 2 or is_close_score or is_same_brand:
                neighbors.append(
                    CascadeNeighborItem(
                        product_id=cand.product_id,
                        slug=cand.slug,
                        title=cand.title,
                        manufacturer=cand.manufacturer,
                        dino_similarity=cand_sim,
                        rank_v1=rank,
                    )
                )
            if len(neighbors) >= self.max_neighbors:
                break

        return details, neighbors
