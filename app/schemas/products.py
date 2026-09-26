import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict


class ProductResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    slug: str | None = None
    title: str
    manufacturer: str
    description: str
    image_url: str
    label_url: str
    created_at: datetime


class ProductListResponse(BaseModel):
    products: list[ProductResponse]
    total: int = 0
    limit: int = 50
    offset: int = 0
