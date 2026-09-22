from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool

from app.schemas.sommelier import (
    SommelierAskRequest,
    SommelierAskResponse,
    SommelierResetRequest,
    SommelierWineInfoResponse,
)
from app.services.sommelier_service import SommelierService

router = APIRouter(prefix="/sommelier", tags=["sommelier"])


def _service(request: Request) -> SommelierService:
    service: SommelierService | None = request.app.state.sommelier
    if service is None or not service.available:
        raise HTTPException(status_code=503, detail="sommelier catalog is not loaded")
    return service


@router.post("/ask", response_model=SommelierAskResponse)
async def ask(payload: SommelierAskRequest, request: Request) -> SommelierAskResponse:
    """Диалоговая точка входа: свободный текст -> рекомендация/справка/уточнение/отказ (guardrails).

    Первый вызов без `session_id` открывает новую сессию (её id возвращается в ответе и должен передаваться
    в последующих сообщениях того же диалога — контекст блюда/предпочтений/обратной связи хранится по сессии).
    """
    service = _service(request)
    session_id, sommelier = service.session(payload.session_id)
    result = await run_in_threadpool(sommelier.ask, payload.message)
    return SommelierAskResponse(
        session_id=session_id,
        kind=result["kind"],
        text=result["text"],
        picks=result.get("picks", []),
        types=result.get("types", []),
        pool=result.get("pool"),
        context=result.get("context"),
    )


@router.post("/reset")
async def reset(payload: SommelierResetRequest, request: Request) -> dict[str, bool]:
    """Начинает диалог заново (тот же session_id, пустой контекст)."""
    service = _service(request)
    ok = await run_in_threadpool(service.reset, payload.session_id)
    if not ok:
        raise HTTPException(status_code=404, detail="unknown session_id")
    return {"ok": True}


@router.get("/wine/{slug}", response_model=SommelierWineInfoResponse)
async def wine_info(slug: str, request: Request, dish: str | None = Query(None, max_length=500)) -> SommelierWineInfoResponse:
    """Точка интеграции с распознаванием этикеток: slug товара из `/api/v1/search` -> справка сомелье по нему,
    и, если передан `dish` (текст блюда), оценка сочетания по правилам движка."""
    service = _service(request)
    sommelier = await run_in_threadpool(service.wine_lookup)
    dish_parsed = None
    if dish:
        from app.sommelier.dishes import parse_dish

        dish_parsed = await run_in_threadpool(parse_dish, dish)
    result = await run_in_threadpool(sommelier.about_wine, slug, dish_parsed)
    return SommelierWineInfoResponse(kind=result["kind"], id=result.get("id"), text=result["text"])
