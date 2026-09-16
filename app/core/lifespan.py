import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.config import get_settings
from app.db.session import create_engine_and_session_factory
from app.pipelines.search.v1.pipeline import SearchPipelineV1
from app.pipelines.search.v1.reranking import SiftReranker
from app.services.detector import DetectorService
from app.services.embeddings import EmbeddingService
from app.services.images import ImageService

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    engine, session_factory = create_engine_and_session_factory(settings.database_url)
    images = ImageService(settings.media_dir, settings.canonical_size, settings.max_upload_bytes, settings.max_image_pixels)
    app.state.engine = engine
    app.state.session_factory = session_factory
    app.state.images = images
    app.state.detector = None
    app.state.pipeline_v1 = None
    try:
        detector, embeddings = await asyncio.gather(
            asyncio.to_thread(DetectorService, settings.yolo_model_path, settings.yolo_confidence),
            asyncio.to_thread(EmbeddingService, settings.dino_model_path, settings.dino_base_model_path, settings.embedding_dimension),
        )
        app.state.detector = detector
        app.state.pipeline_v1 = SearchPipelineV1(embeddings, images, SiftReranker(), asyncio.Semaphore(settings.gpu_concurrency), asyncio.Semaphore(settings.sift_concurrency), settings.candidate_pool_size)
    except Exception:
        logger.exception("ML services failed to load")
    yield
    await engine.dispose()
