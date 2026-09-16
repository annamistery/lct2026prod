import asyncio
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dependencies import get_batch_import, get_session
from app.db.models import ImportItem, ImportJob
from app.schemas.imports import ImportItemResponse, ImportJobResponse, ImportStartRequest, ImportStartResponse
from app.services.batch_import import BatchImportService

router = APIRouter(prefix="/imports", tags=["imports"])


@router.post("", response_model=ImportStartResponse, status_code=202)
async def start_import(payload: ImportStartRequest, request: Request, session: AsyncSession = Depends(get_session), service: BatchImportService = Depends(get_batch_import)) -> ImportStartResponse:
    if any(not task.done() for task in request.app.state.import_tasks):
        raise HTTPException(status_code=409, detail="Another batch import is already running")
    try:
        await asyncio.to_thread(service.load_manifest, payload.batch_id, payload.manifest_name)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    job = ImportJob(batch_id=payload.batch_id, manifest_name=payload.manifest_name, status="pending")
    session.add(job)
    await session.commit()
    task = asyncio.create_task(service.run(job.id), name=f"import-{job.id}")
    request.app.state.import_tasks.add(task)
    task.add_done_callback(request.app.state.import_tasks.discard)
    return ImportStartResponse(job_id=job.id, status=job.status)


@router.get("/{job_id}", response_model=ImportJobResponse)
async def import_status(job_id: uuid.UUID, item_limit: int = Query(100, ge=0, le=1000), session: AsyncSession = Depends(get_session)) -> ImportJobResponse:
    job = await session.get(ImportJob, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Import job not found")
    items = []
    if item_limit:
        rows = await session.scalars(select(ImportItem).where(ImportItem.job_id == job_id).order_by(ImportItem.row_number.desc()).limit(item_limit))
        items = [ImportItemResponse(row_number=item.row_number, image_path=item.image_path, status=item.status, product_id=item.product_id, error=item.error) for item in rows]
    return ImportJobResponse(job_id=job.id, batch_id=job.batch_id, manifest_name=job.manifest_name, status=job.status, total_items=job.total_items, completed_items=job.completed_items, failed_items=job.failed_items, error=job.error, created_at=job.created_at, started_at=job.started_at, finished_at=job.finished_at, items=items)
