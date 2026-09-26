import asyncio
import base64
import io
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from PIL import Image
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.dependencies import get_detector, get_images, get_pipeline_v1, get_segmenter, get_session
from app.db.repositories.products import ProductRepository
from app.pipelines.search.v1.pipeline import SearchPipelineV1
from app.schemas.search.v1 import PredictResponse, SearchResponse
from app.services.augment import apply_random_hard_augmentation
from app.services.detector import DetectorService
from app.services.images import ImageService, InvalidImage
from app.services.query_prep_v3 import QueryPrepV3
from app.services.segmenter import SegmenterService

router = APIRouter(prefix="/v1", tags=["search-v1"])


def encode_crop_data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="WEBP", quality=90)
    return f"data:image/webp;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"


async def decode_upload(upload: UploadFile, images: ImageService, settings: Settings):
    contents = await upload.read(settings.max_upload_bytes + 1)
    try:
        return await asyncio.to_thread(images.decode, contents)
    except InvalidImage as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/search", response_model=SearchResponse)
async def search(
    image: Annotated[UploadFile, File()],
    k: Annotated[int, Form(ge=1)] = 5,
    session: AsyncSession = Depends(get_session),
    images: ImageService = Depends(get_images),
    detector: DetectorService = Depends(get_detector),
    segmenter: Annotated[SegmenterService | None, Depends(get_segmenter)] = None,
    pipeline: SearchPipelineV1 = Depends(get_pipeline_v1),
    settings: Settings = Depends(get_settings),
) -> SearchResponse:
    if k > settings.max_top_k:
        raise HTTPException(status_code=422, detail=f"k must not exceed {settings.max_top_k}")
    source = await decode_upload(image, images, settings)
    query_prep = QueryPrepV3(
        detector=detector,
        segmenter=segmenter,
        target_size=settings.canonical_size_v4,
        padding_ratio=0.05,
    )
    prep = await asyncio.to_thread(query_prep.prepare, source)
    if prep.bbox is None and prep.is_fallback:
        raise HTTPException(status_code=422, detail="Label was not detected")
    query_crop_b64 = await asyncio.to_thread(encode_crop_data_url, prep.image)
    return await pipeline.run(prep.image, ProductRepository(session), k, query_crop=query_crop_b64)


@router.post("/search-from-crop", response_model=SearchResponse)
async def search_from_crop(
    image: Annotated[UploadFile, File()],
    k: Annotated[int, Form(ge=1)] = 5,
    augment: Annotated[bool, Form()] = False,
    session: AsyncSession = Depends(get_session),
    images: ImageService = Depends(get_images),
    segmenter: Annotated[SegmenterService | None, Depends(get_segmenter)] = None,
    pipeline: SearchPipelineV1 = Depends(get_pipeline_v1),
    settings: Settings = Depends(get_settings),
) -> SearchResponse:
    if k > settings.max_top_k:
        raise HTTPException(status_code=422, detail=f"k must not exceed {settings.max_top_k}")
    crop = await decode_upload(image, images, settings)
    query_prep = QueryPrepV3(
        detector=None,
        segmenter=segmenter,
        target_size=settings.canonical_size_v4,
        padding_ratio=0.05,
    )
    prep = await asyncio.to_thread(query_prep.prepare_crop, crop)
    prepared = prep.image
    applied_effects: list[str] | None = None
    if augment:
        prepared, applied_effects = await asyncio.to_thread(apply_random_hard_augmentation, prepared)
    query_crop_b64 = await asyncio.to_thread(encode_crop_data_url, prepared)
    return await pipeline.run(
        prepared,
        ProductRepository(session),
        k,
        query_crop=query_crop_b64,
        augmentation_applied=applied_effects,
    )


@router.post("/eval/predict", response_model=PredictResponse)
async def predict_eval(
    image: Annotated[UploadFile, File()],
    session: AsyncSession = Depends(get_session),
    images: ImageService = Depends(get_images),
    detector: DetectorService = Depends(get_detector),
    segmenter: Annotated[SegmenterService | None, Depends(get_segmenter)] = None,
    pipeline: SearchPipelineV1 = Depends(get_pipeline_v1),
    settings: Settings = Depends(get_settings),
) -> PredictResponse:
    """Predict top-1 wine slug for verification benchmark script."""
    source = await decode_upload(image, images, settings)
    query_prep = QueryPrepV3(
        detector=detector,
        segmenter=segmenter,
        target_size=settings.canonical_size_v4,
        padding_ratio=0.05,
    )
    prep = await asyncio.to_thread(query_prep.prepare, source)
    response = await pipeline.run(prep.image, ProductRepository(session), k=1)
    if response.winner and response.winner.slug:
        return PredictResponse(slug=response.winner.slug)
    return PredictResponse(slug=None)
