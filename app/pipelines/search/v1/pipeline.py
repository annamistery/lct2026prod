import asyncio
import time
from dataclasses import dataclass

from PIL import Image

from app.db.repositories.products import ProductRepository
from app.pipelines.search.v1.reranking import SiftReranker
from app.schemas.search.v1 import SearchResponse, SearchResult, SearchTimings
from app.services.embeddings import EmbeddingService
from app.services.images import ImageService


@dataclass
class SearchPipelineV1:
    embeddings: EmbeddingService
    images: ImageService
    reranker: SiftReranker
    gpu_semaphore: asyncio.Semaphore
    sift_semaphore: asyncio.Semaphore
    candidate_pool_size: int
    enable_sift_rerank: bool = False

    async def run(
        self,
        image: Image.Image,
        repository: ProductRepository,
        k: int,
        query_crop: str | None = None,
        augmentation_applied: list[str] | None = None,
    ) -> SearchResponse:
        started = time.perf_counter()
        async with self.gpu_semaphore:
            embedding_started = time.perf_counter()
            embedding = await asyncio.to_thread(self.embeddings.embed, image)
            embedding_ms = self._elapsed(embedding_started)
        database_started = time.perf_counter()
        candidates = await repository.nearest(embedding, max(k, self.candidate_pool_size if self.enable_sift_rerank else k))
        pgvector_ms = self._elapsed(database_started)

        if not self.enable_sift_rerank:
            results = [
                SearchResult(
                    product_id=cand.product.id,
                    slug=cand.product.slug,
                    title=cand.product.title,
                    manufacturer=cand.product.manufacturer,
                    description=cand.product.description,
                    image_url=f"/api/media/{cand.image_path}",
                    dino_similarity=round(1.0 - cand.distance, 4),
                    sift_score=round(1.0 - cand.distance, 4),
                    inliers=0,
                    inlier_ratio=0.0,
                    pgvector_rank=rank,
                    final_rank=rank,
                )
                for rank, cand in enumerate(candidates[:k], 1)
            ]
            timings = SearchTimings(embedding_ms=embedding_ms, pgvector_ms=pgvector_ms, sift_ms=0.0, total_ms=self._elapsed(started))
            return SearchResponse(
                winner=results[0] if results else None,
                results=results,
                timings=timings,
                query_crop=query_crop,
                augmentation_applied=augmentation_applied,
            )

        sift_started = time.perf_counter()

        async def score(rank: int, candidate):
            try:
                path = self.images.resolve(candidate.image_path)
                candidate_image = await asyncio.to_thread(self._open_image, path)
                similarity = 1.0 - candidate.distance
                async with self.sift_semaphore:
                    sift = await asyncio.to_thread(self.reranker.compare, image, candidate_image, similarity)
                return rank, candidate.product, similarity, sift, candidate.image_path
            except (FileNotFoundError, OSError):
                return None

        scored = [item for item in await asyncio.gather(*(score(rank, candidate) for rank, candidate in enumerate(candidates, 1))) if item]
        scored.sort(key=lambda item: (-item[3].score, item[0]))
        sift_ms = self._elapsed(sift_started)
        results = [
            SearchResult(
                product_id=item[1].id,
                slug=item[1].slug,
                title=item[1].title,
                manufacturer=item[1].manufacturer,
                description=item[1].description,
                image_url=f"/api/media/{item[4]}",
                dino_similarity=item[2],
                sift_score=item[3].score,
                inliers=item[3].inliers,
                inlier_ratio=item[3].inlier_ratio,
                pgvector_rank=item[0],
                final_rank=rank,
            )
            for rank, item in enumerate(scored[:k], 1)
        ]
        timings = SearchTimings(embedding_ms=embedding_ms, pgvector_ms=pgvector_ms, sift_ms=sift_ms, total_ms=self._elapsed(started))
        return SearchResponse(
            winner=results[0] if results else None,
            results=results,
            timings=timings,
            query_crop=query_crop,
            augmentation_applied=augmentation_applied,
        )

    @staticmethod
    def _open_image(path) -> Image.Image:
        with Image.open(path) as image:
            return image.convert("RGB")

    @staticmethod
    def _elapsed(started: float) -> float:
        return round((time.perf_counter() - started) * 1000, 2)
