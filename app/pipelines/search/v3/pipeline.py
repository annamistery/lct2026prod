"""Search pipeline v3: DINOv2-base 768d with seg→letterbox 518, vote-count ranking, SIFT rerank on augmented crops."""

import asyncio
import base64
import io
import time
from dataclasses import dataclass

from PIL import Image

from app.db.repositories.products import ProductCandidateV3, ProductRepository
from app.pipelines.search.v3.reranking import SiftRerankerV3
from app.schemas.search.v3 import SearchResponseV3, SearchResultV3, SearchTimingsV3
from app.services.embeddings import EmbeddingService
from app.services.images import ImageService
from app.services.query_prep_v3 import QueryPrepV3


@dataclass
class SearchPipelineV3:
    embeddings: EmbeddingService
    images: ImageService
    query_prep: QueryPrepV3
    reranker: SiftRerankerV3
    gpu_semaphore: asyncio.Semaphore
    candidate_pool_size: int
    enable_sift_rerank: bool
    sift_threshold: float
    vote_pool_size: int

    async def run(
        self,
        image: Image.Image,
        repository: ProductRepository,
        k: int = 5,
        is_already_crop: bool = False,
    ) -> SearchResponseV3:
        started = time.perf_counter()

        # 1. Query preparation: bbox → seg → clean hull → crop → letterbox 518
        prep_started = time.perf_counter()
        if is_already_crop:
            prep_result = await asyncio.to_thread(self.query_prep.prepare_crop, image)
        else:
            prep_result = await asyncio.to_thread(self.query_prep.prepare, image)
        prep_ms = self._elapsed(prep_started)

        # 2. DINOv2-base embedding (768d)
        async with self.gpu_semaphore:
            emb_started = time.perf_counter()
            embedding = await asyncio.to_thread(self.embeddings.embed, prep_result.image)
            emb_ms = self._elapsed(emb_started)

        # 3. pgvector nearest search with vote counting
        db_started = time.perf_counter()
        candidates = await repository.nearest_v3_with_votes(
            embedding, max(k, self.candidate_pool_size), vote_pool=self.vote_pool_size,
        )
        pgvector_ms = self._elapsed(db_started)

        # 4. Optional SIFT rerank on top candidates
        sift_ms = None
        if self.enable_sift_rerank and candidates:
            top1_sim = 1.0 - candidates[0].distance
            if top1_sim >= self.sift_threshold:
                sift_started = time.perf_counter()
                candidates = await self._sift_rerank(prep_result.image, candidates[:k])
                sift_ms = self._elapsed(sift_started)

        # 5. Build response
        results = self._build_results(candidates[:k], self.vote_pool_size)

        timings = SearchTimingsV3(
            query_prep_ms=prep_ms,
            embedding_ms=emb_ms,
            pgvector_ms=pgvector_ms,
            sift_ms=sift_ms,
            total_ms=self._elapsed(started),
        )

        query_crop_url = self.encode_image_data_url(prep_result.image)
        bbox_crop_url = None
        if prep_result.bbox is not None and not is_already_crop:
            try:
                x1, y1, x2, y2 = prep_result.bbox
                raw_crop = image.crop((x1, y1, x2, y2))
                bbox_crop_url = self.encode_image_data_url(raw_crop)
            except Exception:
                pass

        return SearchResponseV3(
            winner=results[0] if results else None,
            results=results,
            timings=timings,
            query_crop=query_crop_url,
            bbox_crop=bbox_crop_url,
            seg_polygon=prep_result.seg_polygon,
            is_fallback=prep_result.is_fallback,
        )

    async def _sift_rerank(
        self,
        query_image: Image.Image,
        candidates: list[ProductCandidateV3],
    ) -> list[ProductCandidateV3]:
        """SIFT rerank: compare query against the regenerated augmented crop."""
        scored: list[tuple[float, ProductCandidateV3]] = []
        for cand in candidates:
            try:
                # Use v3 letterbox crop (labels_v3/{id}.webp), not the old squashed label
                v3_label_path = self.images.resolve(cand.image_path)
                label_image = Image.open(v3_label_path).convert("RGB")
            except Exception:
                scored.append((1.0 - cand.distance, cand))
                continue

            dino_sim = 1.0 - cand.distance
            result = await asyncio.to_thread(
                self.reranker.compare,
                query_image, label_image,
                cand.aug_name, cand.aug_seed, dino_sim,
            )
            scored.append((result.score, cand))

        scored.sort(key=lambda x: -x[0])
        return [cand for _, cand in scored]

    def _build_results(self, candidates: list[ProductCandidateV3], vote_pool: int) -> list[SearchResultV3]:
        results = []
        for idx, cand in enumerate(candidates, 1):
            aug_image_url = None
            try:
                # Use v3 letterbox crop, not the old squashed label
                v3_label_path = self.images.resolve(cand.image_path)
                label_image = Image.open(v3_label_path).convert("RGB")
                aug_image = self.reranker.regenerate_augment(label_image, cand.aug_name, cand.aug_seed)
                aug_image_url = self.encode_image_data_url(aug_image)
            except Exception:
                pass

            results.append(SearchResultV3(
                product_id=cand.product.id,
                slug=cand.product.slug,
                title=cand.product.title,
                manufacturer=cand.product.manufacturer,
                description=cand.product.description,
                image_url=f"/api/media/{cand.image_path}",
                matched_aug_image=aug_image_url,
                dino_similarity=round(1.0 - cand.distance, 4),
                vote_count=cand.vote_count,
                vote_ratio=round(cand.vote_count / max(1, vote_pool), 4),
                matched_aug_name=cand.aug_name,
                rank=idx,
            ))
        return results

    @staticmethod
    def encode_image_data_url(image: Image.Image) -> str:
        buf = io.BytesIO()
        image.save(buf, format="WEBP", quality=92)
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/webp;base64,{encoded}"

    @staticmethod
    def _elapsed(started: float) -> float:
        return round((time.perf_counter() - started) * 1000, 2)
