import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class ImportStartRequest(BaseModel):
    batch_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,99}$")
    manifest_name: str = Field(default="manifest.json", pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,254}$")


class ImportStartResponse(BaseModel):
    job_id: uuid.UUID
    status: str


class ImportItemResponse(BaseModel):
    row_number: int
    image_path: str
    status: str
    product_id: uuid.UUID | None
    error: str | None


class ImportJobResponse(BaseModel):
    job_id: uuid.UUID
    batch_id: str
    manifest_name: str
    status: str
    total_items: int
    completed_items: int
    failed_items: int
    error: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    items: list[ImportItemResponse]
