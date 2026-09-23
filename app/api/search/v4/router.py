import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.dependencies import get_images, get_pipeline_v4, get_session
from app.db.repositories.products import ProductRepository
from app.pipelines.search.v4.pipeline import SearchPipelineV4
from app.schemas.search.v4 import PredictResponseV4, SearchResponseV4
from app.services.images import ImageService, InvalidImage

router = APIRouter(prefix="/v4", tags=["search-v4"])


async def _decode_upload(upload: UploadFile, images: ImageService, settings: Settings):
    contents = await upload.read(settings.max_upload_bytes + 1)
    try:
        return await asyncio.to_thread(images.decode, contents)
    except InvalidImage as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/search", response_model=SearchResponseV4)
async def search_v4(
    image: Annotated[UploadFile, File()],
    k: Annotated[int, Form(ge=1)] = 5,
    session: AsyncSession = Depends(get_session),
    images: ImageService = Depends(get_images),
    pipeline: SearchPipelineV4 = Depends(get_pipeline_v4),
    settings: Settings = Depends(get_settings),
) -> SearchResponseV4:
    if k > settings.max_top_k:
        raise HTTPException(status_code=422, detail=f"k must not exceed {settings.max_top_k}")
    source = await _decode_upload(image, images, settings)
    return await pipeline.run(source, ProductRepository(session), k, is_already_crop=False)


@router.post("/search-from-crop", response_model=SearchResponseV4)
async def search_from_crop_v4(
    crop: Annotated[UploadFile, File()],
    k: Annotated[int, Form(ge=1)] = 5,
    session: AsyncSession = Depends(get_session),
    images: ImageService = Depends(get_images),
    pipeline: SearchPipelineV4 = Depends(get_pipeline_v4),
    settings: Settings = Depends(get_settings),
) -> SearchResponseV4:
    if k > settings.max_top_k:
        raise HTTPException(status_code=422, detail=f"k must not exceed {settings.max_top_k}")
    source = await _decode_upload(crop, images, settings)
    return await pipeline.run(source, ProductRepository(session), k, is_already_crop=True)


@router.post("/eval/predict", response_model=PredictResponseV4)
async def predict_eval_v4(
    image: Annotated[UploadFile, File()],
    session: AsyncSession = Depends(get_session),
    images: ImageService = Depends(get_images),
    pipeline: SearchPipelineV4 = Depends(get_pipeline_v4),
    settings: Settings = Depends(get_settings),
) -> PredictResponseV4:
    """Predict top-1 wine slug for benchmark using Pipeline v4 (SigLIP 2 + OCR)."""
    source = await _decode_upload(image, images, settings)
    response = await pipeline.run(source, ProductRepository(session), k=1, is_already_crop=False)
    if response.winner and response.winner.slug:
        return PredictResponseV4(slug=response.winner.slug)
    return PredictResponseV4(slug=None)
