import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.dependencies import get_images, get_pipeline_cascade, get_session
from app.db.repositories.products import ProductRepository
from app.pipelines.search.cascade.pipeline import CascadeSearchPipeline
from app.schemas.search.cascade import CascadePredictResponse, CascadeSearchResponse, CascadeThresholdsResponse
from app.services.images import ImageService, InvalidImage

router = APIRouter(prefix="/cascade", tags=["search-cascade"])


async def _decode_upload(upload: UploadFile, images: ImageService, settings: Settings):
    contents = await upload.read(settings.max_upload_bytes + 1)
    try:
        return await asyncio.to_thread(images.decode, contents)
    except InvalidImage as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/thresholds", response_model=CascadeThresholdsResponse)
async def cascade_thresholds(pipeline: CascadeSearchPipeline = Depends(get_pipeline_cascade)) -> CascadeThresholdsResponse:
    """Пороги зон ответа: «найдено» / «похоже» / «нет в каталоге». Задаются в .env (CASCADE_*), применяются после перезапуска."""
    thresholds = pipeline.thresholds
    return CascadeThresholdsResponse(
        found_min_similarity=thresholds.found_min_similarity,
        found_min_margin=thresholds.found_min_margin,
        reject_below_similarity=thresholds.reject_below_similarity,
        predict_threshold=pipeline.predict_threshold,
        candidate_pool=pipeline.candidate_pool,
        v1_weight=pipeline.v1_weight,
        stage="fusion" if pipeline.v4_embeddings is not None else "v1_only",
    )


@router.post("/search", response_model=CascadeSearchResponse)
async def search_cascade(
    image: Annotated[UploadFile, File()],
    k: Annotated[int, Form(ge=1)] = 5,
    session: AsyncSession = Depends(get_session),
    images: ImageService = Depends(get_images),
    pipeline: CascadeSearchPipeline = Depends(get_pipeline_cascade),
    settings: Settings = Depends(get_settings),
) -> CascadeSearchResponse:
    """Full diagnostic cascade search: DINOv2 candidates + SigLIP 2 / DINOv2 fusion.

    ``status``: ``found`` — вино найдено; ``probable`` — показан лучший кандидат, его стоит сверить с этикеткой;
    ``not_in_catalog`` — вина нет в каталоге (``winner`` = null, в ``final_results`` — самые похожие этикетки каталога).
    """
    if k > settings.max_top_k:
        raise HTTPException(status_code=422, detail=f"k must not exceed {settings.max_top_k}")
    source = await _decode_upload(image, images, settings)
    return await pipeline.run(
        source, ProductRepository(session), k,
        is_already_crop=False, threshold=settings.cascade_predict_threshold,
    )


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
    return await pipeline.run(
        source, ProductRepository(session), k,
        is_already_crop=True, threshold=settings.cascade_predict_threshold,
    )


@router.post("/predict", response_model=CascadePredictResponse)
@router.post("/eval/predict", response_model=CascadePredictResponse)
async def predict_cascade(
    image: Annotated[UploadFile, File()],
    threshold: Annotated[float | None, Form(ge=0.0, le=1.0)] = None,
    session: AsyncSession = Depends(get_session),
    images: ImageService = Depends(get_images),
    pipeline: CascadeSearchPipeline = Depends(get_pipeline_cascade),
    settings: Settings = Depends(get_settings),
) -> CascadePredictResponse:
    """Benchmark prediction: top-1 wine slug, answer status and confidence.

    ``status`` is ``found`` (``slug`` is the wine) or ``not_in_catalog`` (``slug`` is ``null``); the scanner's
    «похоже» zone is reported as ``found``. The optional
    ``threshold`` (or the configured ``cascade_predict_threshold``) additionally rejects answers whose SigLIP 2
    similarity is below it.
    """
    source = await _decode_upload(image, images, settings)
    effective_threshold = threshold if threshold is not None else settings.cascade_predict_threshold
    return await pipeline.predict_top1(source, ProductRepository(session), is_already_crop=False, threshold=effective_threshold)
