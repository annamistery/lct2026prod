import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.dependencies import get_detector, get_images, get_pipeline_v1, get_session
from app.db.repositories.products import ProductRepository
from app.pipelines.search.v1.pipeline import SearchPipelineV1
from app.schemas.search.v1 import SearchResponse
from app.services.detector import DetectorService
from app.services.images import ImageService, InvalidImage

router = APIRouter(prefix="/v1", tags=["search-v1"])


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
    pipeline: SearchPipelineV1 = Depends(get_pipeline_v1),
    settings: Settings = Depends(get_settings),
) -> SearchResponse:
    if k > settings.max_top_k:
        raise HTTPException(status_code=422, detail=f"k must not exceed {settings.max_top_k}")
    source = await decode_upload(image, images, settings)
    async with pipeline.gpu_semaphore:
        box = await asyncio.to_thread(detector.best_box, source)
    if box is None:
        raise HTTPException(status_code=422, detail="Label was not detected")
    try:
        crop = images.crop(source, box)
    except InvalidImage as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return await pipeline.run(crop, ProductRepository(session), k)


@router.post("/search-from-crop", response_model=SearchResponse)
async def search_from_crop(
    image: Annotated[UploadFile, File()],
    k: Annotated[int, Form(ge=1)] = 5,
    session: AsyncSession = Depends(get_session),
    images: ImageService = Depends(get_images),
    pipeline: SearchPipelineV1 = Depends(get_pipeline_v1),
    settings: Settings = Depends(get_settings),
) -> SearchResponse:
    if k > settings.max_top_k:
        raise HTTPException(status_code=422, detail=f"k must not exceed {settings.max_top_k}")
    crop = images.canonical(await decode_upload(image, images, settings))
    return await pipeline.run(crop, ProductRepository(session), k)
