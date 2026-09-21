import asyncio
import base64
import io
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from PIL import Image
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.dependencies import get_images, get_pipeline_v2, get_session
from app.db.repositories.products import ProductRepository
from app.pipelines.search.v2.pipeline import SearchPipelineV2
from app.schemas.search.v2 import PredictResponseV2, SearchResponseV2
from app.services.images import ImageService, InvalidImage

router = APIRouter(prefix="/v2", tags=["search-v2"])


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


@router.post("/search", response_model=SearchResponseV2)
async def search(
    image: Annotated[UploadFile, File()],
    k: Annotated[int, Form(ge=1)] = 5,
    session: AsyncSession = Depends(get_session),
    images: ImageService = Depends(get_images),
    pipeline: SearchPipelineV2 = Depends(get_pipeline_v2),
    settings: Settings = Depends(get_settings),
) -> SearchResponseV2:
    if k > settings.max_top_k:
        raise HTTPException(status_code=422, detail=f"k must not exceed {settings.max_top_k}")
    source = await decode_upload(image, images, settings)
    return await pipeline.run(
        source,
        ProductRepository(session),
        k,
        is_already_crop=False,
    )


@router.post("/search-from-crop", response_model=SearchResponseV2)
async def search_from_crop(
    crop: Annotated[UploadFile, File()],
    k: Annotated[int, Form(ge=1)] = 5,
    session: AsyncSession = Depends(get_session),
    images: ImageService = Depends(get_images),
    pipeline: SearchPipelineV2 = Depends(get_pipeline_v2),
    settings: Settings = Depends(get_settings),
) -> SearchResponseV2:
    if k > settings.max_top_k:
        raise HTTPException(status_code=422, detail=f"k must not exceed {settings.max_top_k}")
    source = await decode_upload(crop, images, settings)
    return await pipeline.run(
        source,
        ProductRepository(session),
        k,
        is_already_crop=True,
    )


@router.post("/eval/predict", response_model=PredictResponseV2)
async def predict_eval_v2(
    image: Annotated[UploadFile, File()],
    session: AsyncSession = Depends(get_session),
    images: ImageService = Depends(get_images),
    pipeline: SearchPipelineV2 = Depends(get_pipeline_v2),
    settings: Settings = Depends(get_settings),
) -> PredictResponseV2:
    """Predict top-1 wine slug for verification benchmark script using Pipeline v2."""
    source = await decode_upload(image, images, settings)
    response = await pipeline.run(source, ProductRepository(session), k=1, is_already_crop=False)
    if response.winner and response.winner.slug:
        return PredictResponseV2(slug=response.winner.slug)
    return PredictResponseV2(slug=None)
