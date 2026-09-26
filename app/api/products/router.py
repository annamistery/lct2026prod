import asyncio
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.dependencies import get_images, get_ingestion, get_session
from app.db.models.product import Product
from app.db.repositories.products import ProductRepository
from app.schemas.products import ProductListResponse, ProductResponse
from app.services.images import ImageService, InvalidImage
from app.services.product_ingestion import ProductIngestionService

router = APIRouter(tags=["products"])


def serialize(product: Product) -> ProductResponse:
    return ProductResponse(
        id=product.id,
        slug=product.slug,
        title=product.title,
        manufacturer=product.manufacturer,
        description=product.description,
        image_url=f"/api/media/{product.source_image_path}",
        label_url=f"/api/media/{product.label_image_path}",
        created_at=product.created_at,
    )


@router.get("/products", response_model=ProductListResponse)
async def list_products(
    q: Annotated[str | None, Query(min_length=1, max_length=200)] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
    session: AsyncSession = Depends(get_session),
) -> ProductListResponse:
    repo = ProductRepository(session)
    if q:
        products, total = await repo.search_products(q, limit, offset)
    else:
        products, total = await repo.list_products(limit, offset)
    return ProductListResponse(
        products=[serialize(product) for product in products],
        total=total,
        limit=limit,
        offset=offset,
    )


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
    ingestion: ProductIngestionService = Depends(get_ingestion),
    settings: Settings = Depends(get_settings),
) -> ProductResponse:
    contents = await image.read(settings.max_upload_bytes + 1)
    try:
        source = await asyncio.to_thread(images.decode, contents)
        product = await ingestion.create(session, title, manufacturer, description, source)
    except (InvalidImage, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return serialize(product)


@router.get("/media/{path:path}", response_class=FileResponse)
async def media(path: str, images: ImageService = Depends(get_images)) -> FileResponse:
    try:
        return FileResponse(images.resolve(path))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Media file not found") from exc
