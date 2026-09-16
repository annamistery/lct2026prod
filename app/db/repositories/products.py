import uuid
from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.product import Product, ProductEmbedding


@dataclass(frozen=True)
class ProductCandidate:
    product: Product
    embedding_id: uuid.UUID
    image_path: str
    sample_type: str
    distance: float


class ProductRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def list(self, limit: int = 100, offset: int = 0) -> list[Product]:
        result = await self.session.scalars(select(Product).order_by(Product.created_at.desc()).limit(limit).offset(offset))
        return list(result)

    async def get(self, product_id: uuid.UUID) -> Product | None:
        return await self.session.get(Product, product_id)

    def add(self, product: Product, embedding: ProductEmbedding) -> None:
        product.embeddings.append(embedding)
        self.session.add(product)

    async def nearest(self, embedding: list[float], limit: int) -> list[ProductCandidate]:
        distance = ProductEmbedding.embedding.cosine_distance(embedding)
        ranked = select(
            ProductEmbedding.id.label("embedding_id"),
            ProductEmbedding.product_id,
            ProductEmbedding.image_path,
            ProductEmbedding.sample_type,
            distance.label("distance"),
            func.row_number().over(partition_by=ProductEmbedding.product_id, order_by=distance).label("product_rank"),
        ).subquery()
        statement = (
            select(Product, ranked.c.embedding_id, ranked.c.image_path, ranked.c.sample_type, ranked.c.distance)
            .join(ranked, ranked.c.product_id == Product.id)
            .where(ranked.c.product_rank == 1)
            .order_by(ranked.c.distance, Product.id)
            .limit(limit)
        )
        rows = (await self.session.execute(statement)).all()
        return [ProductCandidate(product=row[0], embedding_id=row[1], image_path=row[2], sample_type=row[3], distance=float(row[4])) for row in rows]
