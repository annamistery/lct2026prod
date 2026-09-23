from fastapi import APIRouter

from app.api.health.router import router as health_router
from app.api.imports.router import router as imports_router
from app.api.products.router import router as products_router
from app.api.search.v1.router import router as search_v1_router
from app.api.search.v2.router import router as search_v2_router
from app.api.search.v3.router import router as search_v3_router
from app.api.sommelier.router import router as sommelier_router
from app.api.twins.router import router as twins_router

router = APIRouter(prefix="/api")
router.include_router(health_router)
router.include_router(products_router)
router.include_router(imports_router)
router.include_router(search_v1_router)
router.include_router(search_v2_router)
router.include_router(search_v3_router)
router.include_router(twins_router)
router.include_router(sommelier_router)
