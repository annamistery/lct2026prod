import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict


class ProductResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str
    manufacturer: str
    description: str
    image_url: str
    label_url: str
    created_at: datetime


class ProductListResponse(BaseModel):
    products: list[ProductResponse]
