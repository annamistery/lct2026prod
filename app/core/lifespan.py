import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.config import get_settings
from app.db.session import create_engine_and_session_factory
from app.pipelines.search.v1.pipeline import SearchPipelineV1
from app.pipelines.search.v1.reranking import SiftReranker
from app.pipelines.search.v2.pipeline import SearchPipelineV2
from app.pipelines.search.v3.pipeline import SearchPipelineV3
from app.pipelines.search.v3.reranking import SiftRerankerV3
from app.services.batch_import import BatchImportService
from app.services.detector import DetectorService
from app.services.embeddings import EmbeddingService
from app.services.images import ImageService
from app.services.product_ingestion import ProductIngestionService
from app.services.query_prep_v3 import QueryPrepV3
from app.services.rectification import RectificationService
from app.services.segmenter import SegmenterService
from app.services.sommelier_service import SommelierService

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
    app.state.segmenter = None
    app.state.rectification = None
    app.state.pipeline_v1 = None
    app.state.pipeline_v2 = None
    app.state.pipeline_v3 = None
    app.state.ingestion = None
    app.state.batch_import = None
    app.state.import_tasks = set()
    app.state.sommelier = None
    try:
        sommelier_service = await asyncio.to_thread(
            SommelierService,
            settings.resolved_sommelier_csv_path,
            settings.resolved_sommelier_feedback_path,
            settings.sommelier_max_sessions,
            settings.sommelier_session_ttl_seconds,
        )
        app.state.sommelier = sommelier_service
        logger.info("Sommelier catalog loaded: %d wines", len(sommelier_service.wines))
    except Exception:
        logger.exception("Sommelier service failed to load")
    try:
        detector, embeddings = await asyncio.gather(
            asyncio.to_thread(DetectorService, settings.yolo_model_path, settings.yolo_confidence),
            asyncio.to_thread(EmbeddingService, settings.dino_model_path, settings.dino_base_model_path, settings.embedding_dimension),
        )
        app.state.detector = detector

        segmenter = None
        seg_path = settings.resolved_yolo_seg_model_path
        if seg_path.is_file():
            try:
                segmenter = await asyncio.to_thread(SegmenterService, seg_path, settings.yolo_seg_confidence)
                logger.info("YOLO segmentation model loaded from %s", seg_path)
            except Exception:
                logger.warning("YOLO segmentation model failed to initialize from %s", seg_path, exc_info=True)
        else:
            logger.warning("YOLO segmentation model file not found at %s", seg_path)
        app.state.segmenter = segmenter

        rectification = RectificationService(
            detector=detector,
            segmenter=segmenter,
            target_size=settings.canonical_size,
        )
        app.state.rectification = rectification
        pipeline = SearchPipelineV1(
            embeddings=embeddings,
            images=images,
            reranker=SiftReranker(),
            gpu_semaphore=asyncio.Semaphore(settings.gpu_concurrency),
            sift_semaphore=asyncio.Semaphore(settings.sift_concurrency),
            candidate_pool_size=settings.candidate_pool_size,
            enable_sift_rerank=settings.enable_sift_rerank,
        )
        app.state.pipeline_v1 = pipeline

        pipeline_v2 = SearchPipelineV2(
            embeddings=embeddings,
            images=images,
            rectifier=rectification,
            gpu_semaphore=asyncio.Semaphore(settings.gpu_concurrency),
            candidate_pool_size=settings.candidate_pool_size,
        )
        app.state.pipeline_v2 = pipeline_v2

        # v3 pipeline (DINOv2-base 768d, optional — loads only if model exists)
        try:
            v3_model_path = settings.dino_v3_model_path
            v3_base_path = settings.dino_v3_base_model_path
            if v3_model_path.is_dir():
                embeddings_v3 = await asyncio.to_thread(
                    EmbeddingService, v3_model_path, v3_base_path, settings.embedding_dimension_v3,
                )
                query_prep_v3 = QueryPrepV3(
                    detector=detector, segmenter=segmenter, target_size=settings.canonical_size_v3,
                )
                pipeline_v3 = SearchPipelineV3(
                    embeddings=embeddings_v3,
                    images=images,
                    query_prep=query_prep_v3,
                    reranker=SiftRerankerV3(),
                    gpu_semaphore=asyncio.Semaphore(settings.gpu_concurrency),
                    candidate_pool_size=settings.candidate_pool_size,
                    enable_sift_rerank=settings.enable_sift_rerank_v3,
                    sift_threshold=settings.sift_rerank_v3_threshold,
                    vote_pool_size=settings.v3_vote_pool_size,
                )
                app.state.pipeline_v3 = pipeline_v3
                logger.info("v3 pipeline loaded (DINOv2-base 768d, canonical %d)", settings.canonical_size_v3)
            else:
                logger.warning("v3 DINOv2 model not found at %s — v3 pipeline disabled", v3_model_path)
        except Exception:
            logger.exception("v3 pipeline failed to load")

        ingestion = ProductIngestionService(images, detector, pipeline, settings.embedding_model_name)
        app.state.ingestion = ingestion
        batch_import = BatchImportService(settings.import_staging_dir, settings.max_import_items, settings.max_upload_bytes, images, ingestion, session_factory)
        await batch_import.recover_interrupted()
        app.state.batch_import = batch_import
    except Exception:
        logger.exception("ML services failed to load")
    yield
    for task in app.state.import_tasks:
        task.cancel()
    if app.state.import_tasks:
        await asyncio.gather(*app.state.import_tasks, return_exceptions=True)
    await engine.dispose()
