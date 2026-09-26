"""Cascade Search Pipeline: Fast v1 (DINOv2) coarse retrieval + selective v4 (SigLIP 2/OCR) neighbor refinement."""

from __future__ import annotations

import asyncio
import base64
import io
import time
from dataclasses import dataclass
from typing import Any

from PIL import Image

from app.db.repositories.products import ProductRepository
from app.pipelines.search.cascade.decision import CascadeDecisionEngine
from app.pipelines.search.v1.pipeline import SearchPipelineV1
from app.pipelines.search.v4.pipeline import SearchPipelineV4
from app.schemas.search.cascade import (
    CascadePredictResponse,
    CascadeSearchResponse,
    CascadeTimings,
)
from app.schemas.search.v1 import SearchResult
from app.schemas.search.v4 import SearchResultV4
from app.services.detector import DetectorService
from app.services.images import ImageService


@dataclass
class CascadeSearchPipeline:
    detector: DetectorService
    pipeline_v1: SearchPipelineV1
    pipeline_v4: SearchPipelineV4 | None
    decision_engine: CascadeDecisionEngine
    images: ImageService
    predict_threshold: float | None = None

    async def run(
        self,
        image: Image.Image,
        repository: ProductRepository,
        k: int = 5,
        is_already_crop: bool = False,
        threshold: float | None = None,
    ) -> CascadeSearchResponse:
        started = time.perf_counter()

        # Step 1: Detect and extract bounding box crop in full original resolution
        bbox_started = time.perf_counter()
        if is_already_crop:
            raw_crop = image
            bbox_detect_ms = 0.0
        else:
            box = await asyncio.to_thread(self.detector.best_box, image)
            bbox_detect_ms = self._elapsed(bbox_started)
            if box is not None:
                try:
                    raw_crop = self.images.crop(image, box)
                except Exception:
                    raw_crop = image
            else:
                raw_crop = image

        # Step 2: Run coarse Stage 1 (v1: DINOv2-small LoRA + pgvector cosine similarity)
        v1_started = time.perf_counter()
        v1_response = await self.pipeline_v1.run(
            raw_crop, repository, k=max(k, 10), query_crop=None,
        )
        v1_total_ms = self._elapsed(v1_started)
        v1_results: list[SearchResult] = v1_response.results

        # Step 3: Evaluate decision engine (unambiguous Top-1 vs neighbor twins)
        dec_started = time.perf_counter()
        decision, v1_neighbors = self.decision_engine.evaluate(v1_results)
        decision_ms = self._elapsed(dec_started)

        # Step 4: Branching — either return Top-1 or refine neighbors via v4
        v4_results: list[SearchResultV4] | None = None
        v4_total_ms: float | None = None
        v4_query_crop: str | None = None
        vintage_detected: str | None = None

        if decision.is_confident or not v1_neighbors or self.pipeline_v4 is None:
            stage_reached = "v1_confident"
            winner = v1_results[0] if v1_results else None
            final_results: list[Any] = v1_results[:k]
            if self.pipeline_v4 is None and not decision.is_confident:
                decision.reason += " (v4 модель недоступна, возврат лучшего из v1)"
        else:
            stage_reached = "v4_refined"
            v4_started = time.perf_counter()

            neighbor_ids = [n.product_id for n in v1_neighbors]
            v4_response = await self.pipeline_v4.run(
                raw_crop,
                repository,
                k=k,
                is_already_crop=True,
                product_ids=neighbor_ids,
            )
            v4_total_ms = self._elapsed(v4_started)
            v4_results = v4_response.results
            v4_query_crop = v4_response.query_crop
            vintage_detected = v4_response.vintage_detected
            winner = v4_response.winner

            # Assemble final results: winner + v4 candidates, filled up with remaining v1 results
            final_results = []
            seen_ids = set()
            if v4_results:
                for item in v4_results:
                    final_results.append(item)
                    seen_ids.add(item.product_id)
            for item in v1_results:
                if len(final_results) >= k:
                    break
                if item.product_id not in seen_ids:
                    final_results.append(item)
                    seen_ids.add(item.product_id)

        # Step 5: Encode BBox crop as data URL
        bbox_crop_url = self._encode_image(raw_crop)

        timings = CascadeTimings(
            bbox_detect_ms=bbox_detect_ms,
            v1_total_ms=v1_total_ms,
            decision_ms=decision_ms,
            v4_total_ms=v4_total_ms,
            total_ms=self._elapsed(started),
        )

        # Apply predict confidence threshold consistently across both endpoints
        effective_threshold = threshold if threshold is not None else self.predict_threshold
        if effective_threshold is not None and winner is not None:
            winner_confidence = (
                getattr(winner, "final_score", None)
                or getattr(winner, "dino_similarity", None)
            )
            if winner_confidence is not None and winner_confidence < effective_threshold:
                winner = None
                final_results = []
                stage_reached = "rejected_low_confidence"
                decision.reason += f" (confidence {winner_confidence:.4f} < threshold {effective_threshold:.4f})"

        return CascadeSearchResponse(
            stage_reached=stage_reached,
            winner=winner,
            final_results=final_results,
            decision=decision,
            v1_results=v1_results[:k],
            v1_neighbors=v1_neighbors,
            v4_results=v4_results,
            timings=timings,
            bbox_crop=bbox_crop_url,
            v4_query_crop=v4_query_crop,
            vintage_detected=vintage_detected,
        )

    async def predict_top1(
        self,
        image: Image.Image,
        repository: ProductRepository,
        is_already_crop: bool = False,
        threshold: float | None = None,
    ) -> CascadePredictResponse:
        """Fast prediction for benchmarks returning only top-1 wine slug and confidence.

        The threshold is applied inside ``run``; if the winner confidence is below
        it ``slug`` is ``None`` per customer requirements.
        """
        effective_threshold = threshold if threshold is not None else self.predict_threshold
        response = await self.run(
            image, repository, k=1, is_already_crop=is_already_crop, threshold=effective_threshold,
        )
        winner = response.winner
        confidence = (
            getattr(winner, "final_score", None)
            or getattr(winner, "dino_similarity", None)
            if winner
            else None
        )
        return CascadePredictResponse(
            slug=winner.slug if winner else None,
            stage_reached=response.stage_reached,
            confidence=round(confidence, 4) if confidence is not None else None,
        )

    @staticmethod
    def _encode_image(image: Image.Image) -> str:
        buf = io.BytesIO()
        image.save(buf, format="WEBP", quality=88)
        return f"data:image/webp;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"

    @staticmethod
    def _elapsed(started: float) -> float:
        return round((time.perf_counter() - started) * 1000, 2)
