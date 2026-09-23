import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dependencies import get_session
from app.db.models import Product, ProductEmbedding
from app.db.repositories.products import ProductRepository

BASE_DIR = Path(__file__).resolve().parent.parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "app" / "web" / "templates"))

router = APIRouter(include_in_schema=False)


@router.get("/", response_class=RedirectResponse)
async def index():
    return RedirectResponse(url="/products")


@router.get("/products", response_class=HTMLResponse)
async def products_page(
    request: Request,
    limit: Annotated[int, Query(ge=1, le=100)] = 60,
    offset: Annotated[int, Query(ge=0)] = 0,
    session: AsyncSession = Depends(get_session),
):
    repo = ProductRepository(session)
    products = await repo.list_products(limit=limit, offset=offset)
    total = await session.scalar(select(func.count(Product.id))) or 0
    return templates.TemplateResponse(
        "products.html",
        {
            "request": request,
            "active": "products",
            "products": products,
            "total_count": total,
            "limit": limit,
            "offset": offset,
        },
    )


@router.get("/products/{product_id}", response_class=HTMLResponse)
async def product_detail_page(
    product_id: uuid.UUID,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    repo = ProductRepository(session)
    product = await repo.get(product_id)
    if product is None:
        raise HTTPException(status_code=404, detail="Товар не найден")
    return templates.TemplateResponse(
        "product_detail.html",
        {
            "request": request,
            "active": "products",
            "product": product,
        },
    )


@router.get("/add", response_class=HTMLResponse)
async def add_product_page(request: Request):
    return templates.TemplateResponse(
        "add_product.html",
        {
            "request": request,
            "active": "add_product",
        },
    )


@router.get("/search", response_class=HTMLResponse)
async def search_page(request: Request):
    return templates.TemplateResponse(
        "search.html",
        {
            "request": request,
            "active": "search",
        },
    )


@router.get("/search-v2", response_class=HTMLResponse)
async def search_v2_page(request: Request):
    return templates.TemplateResponse(
        "search_v2.html",
        {
            "request": request,
            "active": "search_v2",
        },
    )


@router.get("/search-v3", response_class=HTMLResponse)
async def search_v3_page(request: Request):
    return templates.TemplateResponse(
        "search_v3.html",
        {
            "request": request,
            "active": "search_v3",
        },
    )


@router.get("/search-v4", response_class=HTMLResponse)
async def search_v4_page(request: Request):
    return templates.TemplateResponse(
        "search_v4.html",
        {
            "request": request,
            "active": "search_v4",
        },
    )


@router.get("/import", response_class=HTMLResponse)
async def import_page(
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    total_products = await session.scalar(select(func.count(Product.id))) or 0
    total_embeddings = await session.scalar(select(func.count(ProductEmbedding.id))) or 0
    return templates.TemplateResponse(
        "import.html",
        {
            "request": request,
            "active": "import",
            "total_products": total_products,
            "total_embeddings": total_embeddings,
        },
    )


@router.get("/twins", response_class=HTMLResponse)
async def twins_page(request: Request):
    return templates.TemplateResponse(
        "twins.html",
        {
            "request": request,
            "active": "twins",
        },
    )
