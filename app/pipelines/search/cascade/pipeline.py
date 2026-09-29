"""Cascade search: DINOv2 candidate retrieval + SigLIP 2 / DINOv2 score fusion + answer zones.

1. YOLO finds the label; two query views are built from the crop: letterbox 518 of the bbox
   crop and the segmented (masked) letterbox. Both models embed both views.
2. DINOv2 (v1) with the averaged embedding of the two views retrieves `candidate_pool`
   products from pgvector (per-product best reference vector).
3. For every candidate the best similarity of each view is read for both models; candidates are
   ranked by fusion = v4(letterbox) + v4(segmented) + w · (v1(letterbox) + v1(segmented)).
4. The SigLIP 2 similarity of the winner and the fusion gap to the runner-up decide the answer
   zone (found / probable / not_in_catalog), see decision.py.

5. Optional label check (label_check.py): for «probable» answers a local vision-language model
   reads the label and its producer / name / grapes / sweetness are compared with the catalog
   cards of the nearest candidates: keep the answer, switch to a lower candidate or answer
   «нет в каталоге». Plain OCR is not used: it did not improve accuracy (docs/RECOGNITION_QUALITY.md).
"""

from __future__ import annotations

import asyncio
import base64
import io
import time
import uuid
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from app.db.models.product import Product, ProductEmbedding, ProductEmbeddingV4
from app.db.repositories.products import ProductRepository
from app.pipelines.search.cascade.decision import MESSAGES, NOT_IN_CATALOG, PROBABLE, RecognitionThresholds, classify
from app.pipelines.search.cascade.label_check import REJECT, SWITCH, CatalogCard, LabelReading, decide
from app.schemas.search.cascade import (
    CascadeCandidate,
    CascadeDecision,
    CascadeLabelCheck,
    CascadePredictResponse,
    CascadeSearchResponse,
    CascadeTimings,
)
from app.schemas.search.v1 import SearchResult
from app.services.detector import DetectorService
from app.services.embeddings import EmbeddingService
from app.services.label_reader import LabelReader
from app.services.query_prep_v3 import QueryPrepV3, letterbox_pil
from app.services.siglip_embeddings import SigLIP2EmbeddingService


@dataclass
class CascadeSearchPipeline:
    detector: DetectorService
    v1_embeddings: EmbeddingService
    v4_embeddings: SigLIP2EmbeddingService | None
    query_prep: QueryPrepV3
    gpu_semaphore: asyncio.Semaphore
    thresholds: RecognitionThresholds = field(default_factory=RecognitionThresholds)
    candidate_pool: int = 30
    v1_weight: float = 0.3
    predict_threshold: float | None = None
    target_size: int = 518
    label_reader: LabelReader | None = None
    label_cards: dict[str, CatalogCard] = field(default_factory=dict)
    label_top_k: int = 5

    async def run(
        self,
        image: Image.Image,
        repository: ProductRepository,
        k: int = 5,
        is_already_crop: bool = False,
        threshold: float | None = None,
        include_images: bool = True,
    ) -> CascadeSearchResponse:
        started = time.perf_counter()

        # 1. Label crop (full resolution) and the two query views
        bbox_detect_ms: float | None = None
        raw_crop = image
        if not is_already_crop:
            detect_started = time.perf_counter()
            box = await asyncio.to_thread(self.detector.best_box, image)
            bbox_detect_ms = self._elapsed(detect_started)
            raw_crop = self._crop(image, box)
        prep_started = time.perf_counter()
        letterbox_view = letterbox_pil(raw_crop, self.target_size)
        segmented_view = (await asyncio.to_thread(self.query_prep.prepare_crop, raw_crop)).image
        views = [letterbox_view, segmented_view]
        query_prep_ms = self._elapsed(prep_started)

        # 2. DINOv2: candidates by the averaged embedding, then per-view similarities
        v1_started = time.perf_counter()
        async with self.gpu_semaphore:
            v1_vectors = await asyncio.to_thread(self.v1_embeddings.embed_batch, views)
        averaged = self._average(v1_vectors)
        v1_candidates = await repository.nearest(averaged, max(k, self.candidate_pool))
        products = {candidate.product.id: candidate.product for candidate in v1_candidates}
        product_ids = list(products)
        v1_per_view = [await repository.max_similarity_by_product(ProductEmbedding, vector, product_ids) for vector in v1_vectors]
        v1_total_ms = self._elapsed(v1_started)

        # 3. SigLIP 2 per-view similarities for the same candidates
        v4_total_ms: float | None = None
        v4_per_view: list[dict[uuid.UUID, float]] | None = None
        if self.v4_embeddings is not None and product_ids:
            v4_started = time.perf_counter()
            async with self.gpu_semaphore:
                v4_vectors = await asyncio.to_thread(self.v4_embeddings.embed_batch, views)
            v4_per_view = [await repository.max_similarity_by_product(ProductEmbeddingV4, vector, product_ids) for vector in v4_vectors]
            v4_total_ms = self._elapsed(v4_started)

        # 4. Fusion ranking and answer zone
        ranked = self._rank(products, v1_per_view, v4_per_view)
        best = ranked[0] if ranked else None
        margin = round(best.fusion_score - ranked[1].fusion_score, 4) if len(ranked) > 1 else None
        similarity = best.v4_similarity if best is not None else None
        if best is None:
            status, reason = NOT_IN_CATALOG, "Каталог пуст: кандидатов нет"
        else:
            status, reason = classify(similarity, margin, self.thresholds)
        label_check: CascadeLabelCheck | None = None
        label_check_ms: float | None = None
        if status == PROBABLE and self.label_reader is not None and best is not None:
            label_started = time.perf_counter()
            best, status, reason, label_check = await self._check_label(raw_crop, ranked, best, status, reason)
            label_check_ms = self._elapsed(label_started)

        effective_threshold = threshold if threshold is not None else self.predict_threshold
        confidence = similarity if similarity is not None else (best.v1_similarity if best is not None else None)
        if status != NOT_IN_CATALOG and effective_threshold is not None and confidence is not None and confidence < effective_threshold:
            status, reason = NOT_IN_CATALOG, f"Уверенность {confidence:.3f} ниже заданного порога {effective_threshold:.3f}"

        v1_results = [
            SearchResult(
                product_id=candidate.product.id,
                slug=candidate.product.slug,
                title=candidate.product.title,
                manufacturer=candidate.product.manufacturer,
                description=candidate.product.description,
                image_url=f"/api/media/{candidate.product.label_image_path}",
                dino_similarity=round(1.0 - candidate.distance, 4),
                pgvector_rank=rank,
                final_rank=rank,
            )
            for rank, candidate in enumerate(v1_candidates[:k], 1)
        ]
        return CascadeSearchResponse(
            status=status,
            message=MESSAGES[status],
            confidence=round(confidence, 4) if confidence is not None else None,
            stage_reached="fusion" if v4_per_view is not None else "v1_only",
            winner=best if status != NOT_IN_CATALOG else None,
            final_results=ranked[:k],
            decision=CascadeDecision(status=status, similarity=similarity, margin=margin, reason=reason, label_check=label_check),
            v1_results=v1_results,
            timings=CascadeTimings(
                bbox_detect_ms=bbox_detect_ms,
                query_prep_ms=query_prep_ms,
                v1_total_ms=v1_total_ms,
                v4_total_ms=v4_total_ms,
                label_check_ms=label_check_ms,
                total_ms=self._elapsed(started),
            ),
            bbox_crop=self._encode_image(raw_crop) if include_images else None,
            v4_query_crop=self._encode_image(segmented_view) if include_images else None,
        )

    async def predict_top1(
        self,
        image: Image.Image,
        repository: ProductRepository,
        is_already_crop: bool = False,
        threshold: float | None = None,
    ) -> CascadePredictResponse:
        """Fast prediction for benchmarks: slug of the winner, or null when the wine is not in the catalog."""
        response = await self.run(image, repository, k=1, is_already_crop=is_already_crop, threshold=threshold, include_images=False)
        return CascadePredictResponse(
            slug=response.winner.slug if response.winner else None,
            status=response.status,
            stage_reached=response.stage_reached,
            confidence=response.confidence,
        )

    async def _check_label(
        self, crop: Image.Image, ranked: list[CascadeCandidate], best: CascadeCandidate, status: str, reason: str
    ) -> tuple[CascadeCandidate, str, str, CascadeLabelCheck]:
        """Label text vs catalog cards of the nearest candidates; on any failure the image answer stays."""
        reading = await self.label_reader.read(crop)
        cards = [self.label_cards[c.slug] for c in ranked[: self.label_top_k] if c.slug in self.label_cards]
        if reading is None or not cards or cards[0].slug != best.slug:
            return best, status, reason, CascadeLabelCheck(action="skipped", reason="этикетку прочитать не удалось", reading=reading)
        verdict = decide(LabelReading.from_dict(reading), cards, self.label_top_k)
        check = CascadeLabelCheck(action=verdict.action, reason=verdict.reason, reading=reading)
        if verdict.action == REJECT:
            return best, NOT_IN_CATALOG, f"{reason}. Проверка этикетки: {verdict.reason}", check
        if verdict.action == SWITCH:
            chosen = next(c for c in ranked if c.slug == verdict.slug)
            return chosen, status, f"{reason}. Проверка этикетки: {verdict.reason}", check
        return best, status, reason, check

    def warm_up(self) -> None:
        """First GPU inference takes ~15 s (kernel selection): pay it at startup, not on the first user photo."""
        sample = Image.new("RGB", (640, 900), "white")
        self.detector.best_box(sample)
        views = [letterbox_pil(sample, self.target_size), self.query_prep.prepare_crop(sample).image]
        self.v1_embeddings.embed_batch(views)
        if self.v4_embeddings is not None:
            self.v4_embeddings.embed_batch(views)

    def _rank(
        self,
        products: dict[uuid.UUID, Product],
        v1_per_view: list[dict[uuid.UUID, float]],
        v4_per_view: list[dict[uuid.UUID, float]] | None,
    ) -> list[CascadeCandidate]:
        scored = []
        for product_id, product in products.items():
            v1 = [view.get(product_id, -1.0) for view in v1_per_view]
            fusion = self.v1_weight * sum(v1)
            v4_best: float | None = None
            if v4_per_view is not None:
                v4 = [view.get(product_id, -1.0) for view in v4_per_view]
                fusion += sum(v4)
                v4_best = max(v4)
            scored.append((fusion, max(v1), v4_best, product))
        scored.sort(key=lambda item: (-item[0], str(item[3].id)))
        return [
            CascadeCandidate(
                product_id=product.id,
                slug=product.slug,
                title=product.title,
                manufacturer=product.manufacturer,
                description=product.description,
                image_url=f"/api/media/{product.label_image_path}",
                v1_similarity=round(v1_best, 4),
                v4_similarity=round(v4_best, 4) if v4_best is not None else None,
                fusion_score=round(fusion, 4),
                rank=rank,
                dino_similarity=round(v1_best, 4),
                final_score=round(v4_best if v4_best is not None else v1_best, 4),
            )
            for rank, (fusion, v1_best, v4_best, product) in enumerate(scored, 1)
        ]

    @staticmethod
    def _crop(image: Image.Image, box: tuple[float, float, float, float] | None) -> Image.Image:
        if box is None:
            return image
        x1, y1, x2, y2 = box
        left, top = max(0, int(x1)), max(0, int(y1))
        right, bottom = min(image.width, int(x2)), min(image.height, int(y2))
        if right <= left or bottom <= top:
            return image
        return image.crop((left, top, right, bottom))

    @staticmethod
    def _average(vectors: list[list[float]]) -> list[float]:
        mean = np.mean(np.asarray(vectors, dtype=np.float32), axis=0)
        return (mean / (np.linalg.norm(mean) or 1.0)).tolist()

    @staticmethod
    def _encode_image(image: Image.Image) -> str:
        buf = io.BytesIO()
        image.save(buf, format="WEBP", quality=88)
        return f"data:image/webp;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"

    @staticmethod
    def _elapsed(started: float) -> float:
        return round((time.perf_counter() - started) * 1000, 2)
