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
from app.pipelines.search.v4.pipeline import SearchPipelineV4
from app.services.batch_import import BatchImportService
from app.services.detector import DetectorService
from app.services.embeddings import EmbeddingService
from app.services.images import ImageService
from app.services.ocr_reranker import OcrReranker
from app.services.product_ingestion import ProductIngestionService
from app.services.query_prep_v3 import QueryPrepV3
from app.services.rectification import RectificationService
from app.services.segmenter import SegmenterService
from app.services.siglip_embeddings import SigLIP2EmbeddingService
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
    app.state.pipeline_v4 = None
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
                    skip_resize=True,
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

        # v4 pipeline (SigLIP 2 768d, optional — loads only if model exists)
        try:
            v4_model_path = settings.resolved_siglip_v4_model_path
            v4_base_path = settings.resolved_siglip_v4_base_model_path
            if v4_model_path.is_dir():
                embeddings_v4 = await asyncio.to_thread(
                    SigLIP2EmbeddingService, v4_model_path, v4_base_path,
                    settings.embedding_dimension_v4, True,
                )
                ocr_reranker: OcrReranker | None = None
                if settings.enable_ocr_rerank_v4:
                    try:
                        ocr_reranker = OcrReranker(
                            w_sim=settings.ocr_rerank_weight_sim,
                            w_vintage=settings.ocr_rerank_weight_vintage,
                            w_text=settings.ocr_rerank_weight_text,
                        )
                        logger.info("OCR reranker initialised for v4")
                    except ImportError:
                        logger.warning("v4 OCR reranker disabled: paddleocr/easyocr not installed")
                query_prep_v4 = QueryPrepV3(
                    detector=detector, segmenter=segmenter, target_size=settings.canonical_size_v4,
                )
                pipeline_v4 = SearchPipelineV4(
                    embeddings=embeddings_v4,
                    images=images,
                    query_prep=query_prep_v4,
                    ocr_reranker=ocr_reranker,
                    gpu_semaphore=asyncio.Semaphore(settings.gpu_concurrency),
                    candidate_pool_size=settings.v4_candidate_pool_size,
                    vote_pool_size=settings.v4_vote_pool_size,
                    enable_ocr_rerank=settings.enable_ocr_rerank_v4,
                )
                app.state.pipeline_v4 = pipeline_v4
                logger.info("v4 pipeline loaded (SigLIP2 768d, canonical %d)", settings.canonical_size_v4)
            else:
                logger.warning("v4 SigLIP2 model not found at %s — v4 pipeline disabled", v4_model_path)
        except Exception:
            logger.exception("v4 pipeline failed to load")

        # Cascade search pipeline (v1 Coarse + optional v4 Refinement)
        try:
            p1 = getattr(app.state, "pipeline_v1", None)
            p4 = getattr(app.state, "pipeline_v4", None)
            if p1 is not None:
                from app.pipelines.search.cascade.decision import CascadeDecisionEngine
                from app.pipelines.search.cascade.pipeline import CascadeSearchPipeline

                decision_engine = CascadeDecisionEngine(
                    confidence_margin=settings.cascade_confidence_margin,
                    min_confidence_score=settings.cascade_min_confidence_score,
                    neighbor_window=settings.cascade_neighbor_score_window,
                    max_neighbors=settings.cascade_max_neighbors,
                )
                pipeline_cascade = CascadeSearchPipeline(
                    detector=detector,
                    pipeline_v1=p1,
                    pipeline_v4=p4,
                    decision_engine=decision_engine,
                    images=images,
                    predict_threshold=settings.cascade_predict_threshold,
                )
                app.state.pipeline_cascade = pipeline_cascade
                if p4 is not None:
                    logger.info("Cascade search pipeline loaded (v1 Coarse + v4 Refinement)")
                else:
                    logger.warning("Cascade search pipeline loaded in fallback mode (v1 Coarse only, v4 unavailable)")
            else:
                app.state.pipeline_cascade = None
        except Exception:
            logger.exception("Cascade pipeline failed to load")

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
