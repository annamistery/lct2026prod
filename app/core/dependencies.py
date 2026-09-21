from collections.abc import AsyncIterator

from fastapi import HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.pipelines.search.v1.pipeline import SearchPipelineV1
from app.pipelines.search.v2.pipeline import SearchPipelineV2
from app.services.batch_import import BatchImportService
from app.services.detector import DetectorService
from app.services.images import ImageService
from app.services.product_ingestion import ProductIngestionService
from app.services.rectification import RectificationService
from app.services.segmenter import SegmenterService


async def get_session(request: Request) -> AsyncIterator[AsyncSession]:
    async with request.app.state.session_factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


def get_images(request: Request) -> ImageService:
    return request.app.state.images


def get_detector(request: Request) -> DetectorService:
    service = request.app.state.detector
    if service is None:
        raise HTTPException(status_code=503, detail="Detector is not ready")
    return service


def get_pipeline_v1(request: Request) -> SearchPipelineV1:
    pipeline = request.app.state.pipeline_v1
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Search pipeline is not ready")
    return pipeline


def get_pipeline_v2(request: Request) -> SearchPipelineV2:
    pipeline = request.app.state.pipeline_v2
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Search pipeline v2 is not ready")
    return pipeline


def get_ingestion(request: Request) -> ProductIngestionService:
    service = request.app.state.ingestion
    if service is None:
        raise HTTPException(status_code=503, detail="Product ingestion is not ready")
    return service


def get_batch_import(request: Request) -> BatchImportService:
    service = request.app.state.batch_import
    if service is None:
        raise HTTPException(status_code=503, detail="Batch import is not ready")
    return service


def get_segmenter(request: Request) -> SegmenterService:
    service = request.app.state.segmenter
    if service is None:
        raise HTTPException(status_code=503, detail="Segmenter is not ready")
    return service


def get_rectification(request: Request) -> RectificationService:
    service = request.app.state.rectification
    if service is None:
        raise HTTPException(status_code=503, detail="Rectification service is not ready")
    return service

