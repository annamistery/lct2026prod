import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.product import Product


@dataclass(frozen=True)
class ProductCandidate:
    product: Product
    distance: float


class ProductRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def list(self, limit: int = 100, offset: int = 0) -> list[Product]:
        result = await self.session.scalars(select(Product).order_by(Product.created_at.desc()).limit(limit).offset(offset))
        return list(result)

    async def get(self, product_id: uuid.UUID) -> Product | None:
        return await self.session.get(Product, product_id)

    def add(self, product: Product) -> None:
        self.session.add(product)

    async def nearest(self, embedding: list[float], limit: int) -> list[ProductCandidate]:
        distance = Product.embedding.cosine_distance(embedding).label("distance")
        rows = (await self.session.execute(select(Product, distance).order_by(distance).limit(limit))).all()
        return [ProductCandidate(product=row[0], distance=float(row[1])) for row in rows]
