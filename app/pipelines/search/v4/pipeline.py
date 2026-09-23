"""Search pipeline v4: SigLIP 2 768d + vote-count ranking + OCR vintage reranker."""

from __future__ import annotations

import asyncio
import base64
import io
import time
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from app.db.repositories.products import ProductCandidateV4, ProductRepository
from app.schemas.search.v4 import SearchResponseV4, SearchResultV4, SearchTimingsV4
from app.services.images import ImageService
from app.services.ocr_reranker import OcrReranker
from app.services.query_prep_v3 import QueryPrepV3
from app.services.siglip_embeddings import SigLIP2EmbeddingService


@dataclass
class SearchPipelineV4:
    embeddings: SigLIP2EmbeddingService
    images: ImageService
    query_prep: QueryPrepV3
    gpu_semaphore: asyncio.Semaphore
    candidate_pool_size: int
    vote_pool_size: int
    enable_ocr_rerank: bool
    ocr_reranker: OcrReranker | None = field(default=None)

    async def run(
        self,
        image: Image.Image,
        repository: ProductRepository,
        k: int = 5,
        is_already_crop: bool = False,
    ) -> SearchResponseV4:
        started = time.perf_counter()

        # 1. Query preparation: bbox → seg → clean hull → letterbox 512
        prep_started = time.perf_counter()
        if is_already_crop:
            prep_result = await asyncio.to_thread(self.query_prep.prepare_crop, image)
        else:
            prep_result = await asyncio.to_thread(self.query_prep.prepare, image)
        prep_ms = self._elapsed(prep_started)

        # 2. SigLIP 2 embedding (768d)
        async with self.gpu_semaphore:
            emb_started = time.perf_counter()
            embedding = await asyncio.to_thread(self.embeddings.embed, prep_result.image)
            emb_ms = self._elapsed(emb_started)

        # 3. pgvector nearest search with vote counting
        db_started = time.perf_counter()
        candidates = await repository.nearest_v4_with_votes(
            embedding, max(k, self.candidate_pool_size), vote_pool=self.vote_pool_size,
        )
        pgvector_ms = self._elapsed(db_started)

        # 4. Optional OCR rerank on top-k candidates
        ocr_ms: float | None = None
        vintage_detected: str | None = None
        final_scores: list[float] = [1.0 - c.distance for c in candidates[:k]]

        if self.enable_ocr_rerank and self.ocr_reranker is not None and candidates:
            top_k = candidates[:k]
            sims = [1.0 - c.distance for c in top_k]
            ocr_started = time.perf_counter()
            try:
                top_k, vintage_detected, final_scores = await asyncio.to_thread(
                    self.ocr_reranker.rerank, prep_result.image, top_k, sims,
                )
                candidates = top_k + candidates[k:]
            except Exception:
                pass  # OCR failure → keep original order
            ocr_ms = self._elapsed(ocr_started)

        # 5. Build response
        results = self._build_results(
            candidates[:k], final_scores, self.vote_pool_size
        )

        timings = SearchTimingsV4(
            query_prep_ms=prep_ms,
            embedding_ms=emb_ms,
            pgvector_ms=pgvector_ms,
            ocr_ms=ocr_ms,
            total_ms=self._elapsed(started),
        )

        query_crop_url = self._encode_image(prep_result.image)
        bbox_crop_url: str | None = None
        if prep_result.bbox is not None and not is_already_crop:
            try:
                x1, y1, x2, y2 = prep_result.bbox
                raw_crop = image.crop((x1, y1, x2, y2))
                bbox_crop_url = self._encode_image(raw_crop)
            except Exception:
                pass

        return SearchResponseV4(
            winner=results[0] if results else None,
            results=results,
            timings=timings,
            query_crop=query_crop_url,
            bbox_crop=bbox_crop_url,
            seg_polygon=prep_result.seg_polygon,
            is_fallback=prep_result.is_fallback,
            vintage_detected=vintage_detected,
        )

    def _build_results(
        self,
        candidates: list[ProductCandidateV4],
        final_scores: list[float],
        vote_pool: int,
    ) -> list[SearchResultV4]:
        # Pad final_scores if OCR reranker was skipped for some candidates
        while len(final_scores) < len(candidates):
            final_scores.append(1.0 - candidates[len(final_scores)].distance)

        results: list[SearchResultV4] = []
        for idx, (cand, fscore) in enumerate(zip(candidates, final_scores), 1):
            aug_image_url: str | None = None
            try:
                # Read the pre-built augmented variant that matched
                aug_path = self.images.resolve(cand.image_path)
                aug_image = Image.open(aug_path).convert("RGB")
                aug_image_url = self._encode_image(aug_image)
            except Exception:
                pass

            # Clean undistorted catalog reference: catalog.webp in product's directory
            cand_p = Path(cand.image_path)
            catalog_rel = str(cand_p.parent / "catalog.webp").replace("\\", "/")
            try:
                self.images.resolve(catalog_rel)
                catalog_url = f"/api/media/{catalog_rel}"
            except Exception:
                catalog_url = f"/api/media/{cand.image_path}"

            ocr_vintage_match: bool | None = None

            results.append(SearchResultV4(
                product_id=cand.product.id,
                slug=cand.product.slug,
                title=cand.product.title,
                manufacturer=cand.product.manufacturer,
                description=cand.product.description,
                image_url=catalog_url,
                matched_aug_image=aug_image_url,
                dino_similarity=round(1.0 - cand.distance, 4),
                vote_count=cand.vote_count,
                vote_ratio=round(cand.vote_count / max(1, vote_pool), 4),
                matched_aug_name=cand.aug_name,
                ocr_vintage_match=ocr_vintage_match,
                ocr_score=None,
                final_score=round(fscore, 4),
                rank=idx,
            ))
        return results

    @staticmethod
    def _encode_image(image: Image.Image) -> str:
        buf = io.BytesIO()
        image.save(buf, format="WEBP", quality=92)
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/webp;base64,{encoded}"

    @staticmethod
    def _elapsed(started: float) -> float:
        return round((time.perf_counter() - started) * 1000, 2)
