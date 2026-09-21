import asyncio
import base64
import io
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

        # 4. Generate base64 previews for UI
        matrix_data_url = self.encode_image_data_url(rect_result.rectified_image)

        bbox_data_url = None
        if rect_result.crop_bbox is not None and not is_already_crop:
            try:
                x1, y1, x2, y2 = rect_result.crop_bbox
                raw_crop = image.crop((x1, y1, x2, y2))
                bbox_data_url = self.encode_image_data_url(raw_crop)
            except Exception:
                bbox_data_url = None

        return SearchResponseV2(
            winner=results[0] if results else None,
            results=results,
            timings=timings,
            query_crop=matrix_data_url,
            bbox_crop=bbox_data_url,
            quad_corners=rect_result.quad_corners_orig,
            is_fallback=rect_result.is_fallback,
        )

    @staticmethod
    def encode_image_data_url(image: Image.Image) -> str:
        buf = io.BytesIO()
        image.save(buf, format="WEBP", quality=92)
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/webp;base64,{encoded}"

    @staticmethod
    def _elapsed(started: float) -> float:
        return round((time.perf_counter() - started) * 1000, 2)
