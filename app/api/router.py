from fastapi import APIRouter

from app.api.health.router import router as health_router
from app.api.imports.router import router as imports_router
from app.api.products.router import router as products_router
from app.api.search.v1.router import router as search_v1_router

router = APIRouter(prefix="/api")
router.include_router(health_router)
router.include_router(products_router)
router.include_router(imports_router)
router.include_router(search_v1_router)
