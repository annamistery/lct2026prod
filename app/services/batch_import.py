import asyncio
import csv
import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.models import ImportItem, ImportJob
from app.services.images import ImageService
from app.services.product_ingestion import ProductIngestionService


@dataclass(frozen=True)
class ImportRecord:
    title: str
    manufacturer: str
    description: str
    image_path: str


class BatchImportService:
    def __init__(self, staging_dir: Path, max_items: int, max_upload_bytes: int, images: ImageService, ingestion: ProductIngestionService, session_factory: async_sessionmaker[AsyncSession]):
        self.staging_dir = staging_dir.resolve()
        self.max_items = max_items
        self.max_upload_bytes = max_upload_bytes
        self.images = images
        self.ingestion = ingestion
        self.session_factory = session_factory

    async def recover_interrupted(self) -> None:
        async with self.session_factory() as session:
            await session.execute(update(ImportJob).where(ImportJob.status.in_(["pending", "running"])).values(status="failed", error="Import interrupted by application restart", finished_at=datetime.now(UTC)))
            await session.commit()

    def resolve_batch(self, batch_id: str) -> Path:
        batch = (self.staging_dir / batch_id).resolve()
        if self.staging_dir not in batch.parents or not batch.is_dir():
            raise FileNotFoundError("Import batch not found")
        return batch

    def load_manifest(self, batch_id: str, manifest_name: str) -> list[ImportRecord]:
        batch = self.resolve_batch(batch_id)
        manifest = (batch / manifest_name).resolve()
        if manifest.parent != batch or not manifest.is_file():
            raise FileNotFoundError("Import manifest not found")
        suffix = manifest.suffix.lower()
        if suffix == ".json":
            data = json.loads(manifest.read_text(encoding="utf-8"))
            rows = data.get("products") if isinstance(data, dict) else data
        elif suffix == ".csv":
            with manifest.open("r", encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
        else:
            raise ValueError("Manifest must be JSON or CSV")
        if not isinstance(rows, list) or not rows:
            raise ValueError("Manifest contains no products")
        if len(rows) > self.max_items:
            raise ValueError(f"Manifest exceeds {self.max_items} products")
        records = []
        for number, row in enumerate(rows, 1):
            if not isinstance(row, dict):
                raise ValueError(f"Row {number} must be an object")
            title = str(row.get("title") or row.get("name") or "").strip()
            manufacturer = str(row.get("manufacturer") or row.get("producer") or "").strip()
            image_path = str(row.get("image_path") or row.get("image") or row.get("filename") or "").strip()
            description = str(row.get("description") or "").strip()
            if not title or not manufacturer or not image_path:
                raise ValueError(f"Row {number} requires title, manufacturer, and image_path")
            if len(title) > 300 or len(manufacturer) > 300 or len(description) > 5000 or len(image_path) > 500:
                raise ValueError(f"Row {number} contains an oversized field")
            image = (batch / image_path).resolve()
            if batch not in image.parents or not image.is_file():
                raise ValueError(f"Row {number} image is missing or outside the batch: {image_path}")
            records.append(ImportRecord(title, manufacturer, description, image_path))
        return records

    async def run(self, job_id: uuid.UUID) -> None:
        try:
            async with self.session_factory() as session:
                job = await session.get(ImportJob, job_id)
                if job is None:
                    return
                job.status = "running"
                job.started_at = datetime.now(UTC)
                await session.commit()
                records = await asyncio.to_thread(self.load_manifest, job.batch_id, job.manifest_name)
                job = await session.get(ImportJob, job_id)
                job.total_items = len(records)
                await session.commit()

            for number, record in enumerate(records, 1):
                await self._process_item(job_id, number, record)

            async with self.session_factory() as session:
                job = await session.get(ImportJob, job_id)
                job.status = "completed"
                job.finished_at = datetime.now(UTC)
                await session.commit()
        except asyncio.CancelledError:
            await self._fail_job(job_id, "Import interrupted by application shutdown")
            raise
        except Exception as exc:
            await self._fail_job(job_id, str(exc))

    async def _process_item(self, job_id: uuid.UUID, number: int, record: ImportRecord) -> None:
        async with self.session_factory() as session:
            item = ImportItem(job_id=job_id, row_number=number, image_path=record.image_path, status="pending")
            session.add(item)
            await session.commit()
            item_id = item.id
            try:
                batch_id = await session.scalar(select(ImportJob.batch_id).where(ImportJob.id == job_id))
                image_path = self.resolve_batch(batch_id) / record.image_path
                contents = await asyncio.to_thread(self._read_limited, image_path)
                source = await asyncio.to_thread(self.images.decode, contents)
                product = await self.ingestion.create(session, record.title, record.manufacturer, record.description, source)
                item = await session.get(ImportItem, item_id)
                job = await session.get(ImportJob, job_id)
                item.status = "completed"
                item.product_id = product.id
                job.completed_items += 1
                await session.commit()
            except Exception as exc:
                await session.rollback()
                item = await session.get(ImportItem, item_id)
                job = await session.get(ImportJob, job_id)
                item.status = "failed"
                item.error = str(exc)[:2000]
                job.failed_items += 1
                await session.commit()

    def _read_limited(self, path: Path) -> bytes:
        with path.open("rb") as stream:
            return stream.read(self.max_upload_bytes + 1)

    async def _fail_job(self, job_id: uuid.UUID, error: str) -> None:
        async with self.session_factory() as session:
            job = await session.get(ImportJob, job_id)
            if job is not None:
                job.status = "failed"
                job.error = error[:4000]
                job.finished_at = datetime.now(UTC)
                await session.commit()
