import asyncio
import time
from dataclasses import dataclass

from PIL import Image

from app.db.repositories.products import ProductRepository
from app.schemas.search.v2 import SearchResponseV2, SearchResultV2, SearchTimingsV2
from app.services.embeddings import EmbeddingService
from app.services.images import ImageService
from app.services.rectification import RectificationResult, RectificationService


@dataclass
class SearchPipelineV2:
    embeddings: EmbeddingService
    images: ImageService
    rectifier: RectificationService
    gpu_semaphore: asyncio.Semaphore
    candidate_pool_size: int

    async def run(
        self,
        image: Image.Image,
        repository: ProductRepository,
        k: int = 5,
        is_already_crop: bool = False,
        query_crop: str | None = None,
    ) -> SearchResponseV2:
        started = time.perf_counter()

        # 1. Cascade Rectification (BBox -> Seg on Crop -> Homography Matrix)
        rect_started = time.perf_counter()
        if is_already_crop:
            rect_result: RectificationResult = await asyncio.to_thread(self.rectifier.rectify_crop, image)
        else:
            rect_result = await asyncio.to_thread(self.rectifier.rectify, image)
        rect_ms = self._elapsed(rect_started)

        # 2. DINOv2 Embedding on normalized matrix
        async with self.gpu_semaphore:
            emb_started = time.perf_counter()
            embedding = await asyncio.to_thread(self.embeddings.embed, rect_result.rectified_image)
            emb_ms = self._elapsed(emb_started)

        # 3. Database Nearest Search in product_embeddings_v2
        db_started = time.perf_counter()
        candidates = await repository.nearest_v2(embedding, max(k, self.candidate_pool_size))
        pgvector_ms = self._elapsed(db_started)

        results = [
            SearchResultV2(
                product_id=cand.product.id,
                slug=cand.product.slug,
                title=cand.product.title,
                manufacturer=cand.product.manufacturer,
                description=cand.product.description,
                image_url=f"/api/media/{cand.image_path}",
                dino_similarity=round(1.0 - cand.distance, 4),
                rank=idx,
            )
            for idx, cand in enumerate(candidates[:k], 1)
        ]

        timings = SearchTimingsV2(
            rectification_ms=rect_ms,
            embedding_ms=emb_ms,
            pgvector_ms=pgvector_ms,
            total_ms=self._elapsed(started),
        )

        return SearchResponseV2(
            winner=results[0] if results else None,
            results=results,
            timings=timings,
            query_crop=query_crop,
            is_fallback=rect_result.is_fallback,
        )

    @staticmethod
    def _elapsed(started: float) -> float:
        return round((time.perf_counter() - started) * 1000, 2)
