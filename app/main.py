import logging
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.router import router
from app.api.search.v1.router import router as search_v1_router
from app.api.search.v2.router import router as search_v2_router
from app.core.config import get_settings
from app.core.lifespan import lifespan
from app.core.logging import configure_logging
from app.web.detect_router import router as detect_router
from app.web.rectify_router import router as rectify_router
from app.web.router import router as web_router

BASE_DIR = Path(__file__).resolve().parent

settings = get_settings()
configure_logging(settings.log_level)
logger = logging.getLogger(__name__)

app = FastAPI(title=settings.app_name, version="1.0.0", docs_url="/api/docs", openapi_url="/api/openapi.json", lifespan=lifespan)
if settings.allowed_origins:
    app.add_middleware(CORSMiddleware, allow_origins=settings.allowed_origins, allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["Content-Type", "X-Request-ID"])

# Mount static and mobile client
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "web" / "static")), name="static")
app.mount("/mobi", StaticFiles(directory=str(BASE_DIR.parent / "web"), html=True), name="mobi")

app.include_router(router)
app.include_router(search_v1_router)
app.include_router(search_v2_router)
app.include_router(web_router)
app.include_router(detect_router)
app.include_router(rectify_router)


@app.middleware("http")
async def request_context(request: Request, call_next):
    started = time.perf_counter()
    raw_id = request.headers.get("X-Request-ID", "")
    request_id = raw_id if raw_id.isascii() and raw_id.replace("-", "").isalnum() and len(raw_id) <= 64 else uuid.uuid4().hex
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    logger.info("request id=%s method=%s path=%s status=%s duration_ms=%.2f", request_id, request.method, request.url.path, response.status_code, (time.perf_counter() - started) * 1000)
    return response
