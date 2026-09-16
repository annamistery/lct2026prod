import asyncio
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.dependencies import get_detector, get_images, get_pipeline_v1, get_session
from app.db.models.product import Product, ProductEmbedding
from app.db.repositories.products import ProductRepository
from app.pipelines.search.v1.pipeline import SearchPipelineV1
from app.schemas.products import ProductListResponse, ProductResponse
from app.services.detector import DetectorService
from app.services.images import ImageService, InvalidImage

router = APIRouter(tags=["products"])


def serialize(product: Product) -> ProductResponse:
    return ProductResponse(id=product.id, title=product.title, manufacturer=product.manufacturer, description=product.description, image_url=f"/api/media/{product.source_image_path}", label_url=f"/api/media/{product.label_image_path}", created_at=product.created_at)


@router.get("/products", response_model=ProductListResponse)
async def list_products(limit: Annotated[int, Query(ge=1, le=100)] = 50, offset: Annotated[int, Query(ge=0)] = 0, session: AsyncSession = Depends(get_session)) -> ProductListResponse:
    products = await ProductRepository(session).list(limit, offset)
    return ProductListResponse(products=[serialize(product) for product in products])


@router.get("/products/{product_id}", response_model=ProductResponse)
async def get_product(product_id: uuid.UUID, session: AsyncSession = Depends(get_session)) -> ProductResponse:
    product = await ProductRepository(session).get(product_id)
    if product is None:
        raise HTTPException(status_code=404, detail="Product not found")
    return serialize(product)


@router.post("/products", response_model=ProductResponse, status_code=201)
async def create_product(
    title: Annotated[str, Form(min_length=1, max_length=300)],
    manufacturer: Annotated[str, Form(min_length=1, max_length=300)],
    image: Annotated[UploadFile, File()],
    description: Annotated[str, Form(max_length=5000)] = "",
    session: AsyncSession = Depends(get_session),
    images: ImageService = Depends(get_images),
    detector: DetectorService = Depends(get_detector),
    pipeline: SearchPipelineV1 = Depends(get_pipeline_v1),
    settings: Settings = Depends(get_settings),
) -> ProductResponse:
    clean_title = title.strip()
    clean_manufacturer = manufacturer.strip()
    if not clean_title or not clean_manufacturer:
        raise HTTPException(status_code=422, detail="Title and manufacturer must not be blank")
    contents = await image.read(settings.max_upload_bytes + 1)
    try:
        source = await asyncio.to_thread(images.decode, contents)
    except InvalidImage as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    async with pipeline.gpu_semaphore:
        box = await asyncio.to_thread(detector.best_box, source)
    if box is None:
        raise HTTPException(status_code=422, detail="Label was not detected")
    try:
        label = images.crop(source, box)
    except InvalidImage as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    async with pipeline.gpu_semaphore:
        embedding = await asyncio.to_thread(pipeline.embeddings.embed, label)
    product_id = uuid.uuid4()
    stored = None
    try:
        stored = await asyncio.to_thread(images.save_product, product_id, source, label)
        product = Product(id=product_id, title=clean_title, manufacturer=clean_manufacturer, description=description.strip(), source_image_path=stored.source_path, label_image_path=stored.label_path)
        product_embedding = ProductEmbedding(image_path=stored.label_path, sample_type="catalog", embedding=embedding, embedding_model=settings.embedding_model_name)
        ProductRepository(session).add(product, product_embedding)
        await session.commit()
        await session.refresh(product)
        return serialize(product)
    except Exception:
        await session.rollback()
        if stored is not None:
            await asyncio.to_thread(images.remove_product, product_id)
        raise


@router.get("/media/{path:path}", response_class=FileResponse)
async def media(path: str, images: ImageService = Depends(get_images)) -> FileResponse:
    try:
        return FileResponse(images.resolve(path))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Media file not found") from exc
