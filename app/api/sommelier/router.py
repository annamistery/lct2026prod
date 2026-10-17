from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dependencies import get_session
from app.db.repositories.products import ProductRepository
from app.schemas.sommelier import (
    SommelierAlternative,
    SommelierAlternativesResponse,
    SommelierAskRequest,
    SommelierAskResponse,
    SommelierResetRequest,
    SommelierWineInfoResponse,
)
from app.services.sommelier_service import SommelierService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/sommelier", tags=["sommelier"])


def _service(request: Request) -> SommelierService:
    service: SommelierService | None = request.app.state.sommelier
    if service is None or not service.available:
        raise HTTPException(status_code=503, detail="sommelier catalog is not loaded")
    return service


@router.post("/ask", response_model=SommelierAskResponse)
async def ask(
    payload: SommelierAskRequest, request: Request, session: AsyncSession = Depends(get_session)
) -> SommelierAskResponse:
    """Диалоговая точка входа: свободный текст -> рекомендация/справка/уточнение/отказ (guardrails).

    Первый вызов без `session_id` открывает новую сессию (её id возвращается в ответе и должен передаваться
    в последующих сообщениях того же диалога — контекст блюда/предпочтений/обратной связи хранится по сессии).
    """
    service = _service(request)
    session_id, sommelier = service.session(payload.session_id)
    result = await run_in_threadpool(sommelier.ask, payload.message)
    picks = result.get("picks", [])
    # фото бутылки — из каталога распознавания (там есть не все вина сомелье); без базы чат работает без фото
    try:
        products = await ProductRepository(session).get_by_slugs([pick["id"] for pick in picks])
    except (SQLAlchemyError, OSError):
        logger.warning("sommelier picks: bottle photos unavailable", exc_info=True)
        products = {}
    picks = [
        {**pick, "image_url": f"/api/media/{products[pick['id']].source_image_path}" if pick["id"] in products else None}
        for pick in picks
    ]
    return SommelierAskResponse(
        session_id=session_id,
        kind=result["kind"],
        text=result["text"],
        picks=picks,
        types=result.get("types", []),
        pool=result.get("pool"),
        total=result.get("total"),
        context=result.get("context"),
        dish_traits=result.get("dish_traits", []),
        note=result.get("note"),
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
    result = await run_in_threadpool(sommelier.about_wine, slug, dish_parsed, dish)
    return SommelierWineInfoResponse(kind=result["kind"], id=result.get("id"), text=result["text"], card=result.get("card"),
                                     profile=result.get("profile"), pairing=result.get("pairing"),
                                     best_dishes=result.get("best_dishes", []), note=result.get("note"))


@router.get("/wine/{slug}/alternatives", response_model=SommelierAlternativesResponse)
async def wine_alternatives(
    slug: str,
    request: Request,
    limit: Annotated[int, Query(ge=1, le=20)] = 5,
    session: AsyncSession = Depends(get_session),
) -> SommelierAlternativesResponse:
    """Точка интеграции с распознаванием: похожие по стилю вина каталога к распознанному вину (slug).

    Цвет и игристость совпадают, сладость отличается не больше чем на ступень; дальше — сорт, тип, тело, танины,
    ароматика и регион из карточек. Не больше двух вин одной винодельни. Картинка — этикетка из каталога распознавания.
    """
    service = _service(request)
    sommelier = await run_in_threadpool(service.wine_lookup)
    result = await run_in_threadpool(sommelier.alternatives, slug, limit)
    products = await ProductRepository(session).get_by_slugs([item["id"] for item in result["items"]])
    items = [
        SommelierAlternative(
            **item,
            image_url=f"/api/media/{products[item['id']].label_image_path}" if item["id"] in products else None,
        )
        for item in result["items"]
    ]
    return SommelierAlternativesResponse(kind=result["kind"], id=result["id"], items=items)
