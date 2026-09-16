from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from sqlalchemy import text

router = APIRouter(tags=["health"])


@router.get("/ping")
async def ping() -> dict[str, bool]:
    return {"ok": True}


@router.get("/ready")
async def ready(request: Request) -> JSONResponse:
    database_ready = False
    try:
        async with request.app.state.session_factory() as session:
            database_ready = bool(await session.scalar(text("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')")))
    except Exception:
        database_ready = False
    ml_ready = request.app.state.detector is not None and request.app.state.pipeline_v1 is not None
    ready_state = database_ready and ml_ready
    return JSONResponse({"ready": ready_state, "database": database_ready, "models": ml_ready}, status_code=200 if ready_state else 503)
