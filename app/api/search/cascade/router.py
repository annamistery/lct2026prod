import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.dependencies import get_images, get_pipeline_cascade, get_session
from app.db.repositories.products import ProductRepository
from app.pipelines.search.cascade.pipeline import CascadeSearchPipeline
from app.schemas.search.cascade import CascadePredictResponse, CascadeSearchResponse
from app.services.images import ImageService, InvalidImage

router = APIRouter(prefix="/cascade", tags=["search-cascade"])


async def _decode_upload(upload: UploadFile, images: ImageService, settings: Settings):
    contents = await upload.read(settings.max_upload_bytes + 1)
    try:
        return await asyncio.to_thread(images.decode, contents)
    except InvalidImage as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/search", response_model=CascadeSearchResponse)
async def search_cascade(
    image: Annotated[UploadFile, File()],
    k: Annotated[int, Form(ge=1)] = 5,
    session: AsyncSession = Depends(get_session),
    images: ImageService = Depends(get_images),
    pipeline: CascadeSearchPipeline = Depends(get_pipeline_cascade),
    settings: Settings = Depends(get_settings),
) -> CascadeSearchResponse:
    """Full diagnostic cascade search: Stage 1 (v1 DINOv2) + conditional Stage 2 (v4 SigLIP 2/OCR) neighbor refinement."""
    if k > settings.max_top_k:
        raise HTTPException(status_code=422, detail=f"k must not exceed {settings.max_top_k}")
    source = await _decode_upload(image, images, settings)
    return await pipeline.run(source, ProductRepository(session), k, is_already_crop=False)


@router.post("/search-from-crop", response_model=CascadeSearchResponse)
async def search_from_crop_cascade(
    crop: Annotated[UploadFile, File()],
    k: Annotated[int, Form(ge=1)] = 5,
    session: AsyncSession = Depends(get_session),
    images: ImageService = Depends(get_images),
    pipeline: CascadeSearchPipeline = Depends(get_pipeline_cascade),
    settings: Settings = Depends(get_settings),
) -> CascadeSearchResponse:
    """Cascade search starting directly from a pre-cropped label."""
    if k > settings.max_top_k:
        raise HTTPException(status_code=422, detail=f"k must not exceed {settings.max_top_k}")
    source = await _decode_upload(crop, images, settings)
    return await pipeline.run(source, ProductRepository(session), k, is_already_crop=True)


@router.post("/predict", response_model=CascadePredictResponse)
async def predict_cascade(
    image: Annotated[UploadFile, File()],
    threshold: Annotated[float | None, Form(ge=0.0, le=1.0)] = None,
    session: AsyncSession = Depends(get_session),
    images: ImageService = Depends(get_images),
    pipeline: CascadeSearchPipeline = Depends(get_pipeline_cascade),
    settings: Settings = Depends(get_settings),
) -> CascadePredictResponse:
    """Benchmark prediction returning only top-1 wine slug and confidence using cascade.

    If ``threshold`` is not provided, the configured ``cascade_predict_threshold`` is used.
    When the winner confidence is below the threshold, ``slug`` is ``null``.
    """
    source = await _decode_upload(image, images, settings)
    effective_threshold = threshold if threshold is not None else settings.cascade_predict_threshold
    return await pipeline.predict_top1(source, ProductRepository(session), is_already_crop=False, threshold=effective_threshold)
