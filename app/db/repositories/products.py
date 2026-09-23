from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.product import Product, ProductEmbedding, ProductEmbeddingV2, ProductEmbeddingV3, ProductEmbeddingV4


@dataclass(frozen=True)
class ProductCandidate:
    product: Product
    embedding_id: uuid.UUID
    image_path: str
    sample_type: str
    distance: float


@dataclass(frozen=True)
class ProductCandidateV3:
    product: Product
    embedding_id: uuid.UUID
    image_path: str
    sample_type: str
    distance: float
    aug_name: str
    aug_seed: int
    vote_count: int


@dataclass(frozen=True)
class ProductCandidateV4:
    product: Product
    embedding_id: uuid.UUID
    image_path: str
    sample_type: str
    distance: float
    aug_name: str
    aug_seed: int
    vote_count: int


class ProductRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def list_products(self, limit: int = 100, offset: int = 0) -> list[Product]:
        result = await self.session.scalars(select(Product).order_by(Product.created_at.desc()).limit(limit).offset(offset))
        return list(result)

    async def get(self, product_id: uuid.UUID) -> Product | None:
        return await self.session.get(Product, product_id)

    async def get_by_slug(self, slug: str) -> Product | None:
        result = await self.session.scalars(select(Product).where(Product.slug == slug).limit(1))
        return result.first()

    def add(self, product: Product, embedding: ProductEmbedding) -> None:
        product.embeddings.append(embedding)
        self.session.add(product)

    def add_v2(self, product: Product, embedding: ProductEmbeddingV2) -> None:
        product.embeddings_v2.append(embedding)
        self.session.add(product)

    def add_embedding_v2(self, embedding: ProductEmbeddingV2) -> None:
        self.session.add(embedding)

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

    async def nearest_v2(self, embedding: list[float], limit: int) -> list[ProductCandidate]:
        distance = ProductEmbeddingV2.embedding.cosine_distance(embedding)
        ranked = select(
            ProductEmbeddingV2.id.label("embedding_id"),
            ProductEmbeddingV2.product_id,
            ProductEmbeddingV2.image_path,
            ProductEmbeddingV2.sample_type,
            distance.label("distance"),
            func.row_number().over(partition_by=ProductEmbeddingV2.product_id, order_by=distance).label("product_rank"),
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

    async def nearest_v3(self, embedding: list[float], limit: int) -> list["ProductCandidateV3"]:
        distance = ProductEmbeddingV3.embedding.cosine_distance(embedding)
        ranked = select(
            ProductEmbeddingV3.id.label("embedding_id"),
            ProductEmbeddingV3.product_id,
            ProductEmbeddingV3.image_path,
            ProductEmbeddingV3.sample_type,
            ProductEmbeddingV3.aug_name,
            ProductEmbeddingV3.aug_seed,
            distance.label("distance"),
            func.row_number().over(partition_by=ProductEmbeddingV3.product_id, order_by=distance).label("product_rank"),
        ).subquery()
        statement = (
            select(
                Product,
                ranked.c.embedding_id,
                ranked.c.image_path,
                ranked.c.sample_type,
                ranked.c.distance,
                ranked.c.aug_name,
                ranked.c.aug_seed,
            )
            .join(ranked, ranked.c.product_id == Product.id)
            .where(ranked.c.product_rank == 1)
            .order_by(ranked.c.distance, Product.id)
            .limit(limit)
        )
        rows = (await self.session.execute(statement)).all()
        return [
            ProductCandidateV3(
                product=row[0], embedding_id=row[1], image_path=row[2],
                sample_type=row[3], distance=float(row[4]),
                aug_name=row[5], aug_seed=int(row[6]),
                vote_count=1,
            )
            for row in rows
        ]

    async def nearest_v3_with_votes(self, embedding: list[float], limit: int, vote_pool: int = 50) -> list["ProductCandidateV3"]:
        """Extended ranking: count how many of the top-N embeddings belong to each product."""
        distance = ProductEmbeddingV3.embedding.cosine_distance(embedding)
        top_embs = (
            select(
                ProductEmbeddingV3.product_id,
                ProductEmbeddingV3.id.label("embedding_id"),
                ProductEmbeddingV3.image_path,
                ProductEmbeddingV3.sample_type,
                ProductEmbeddingV3.aug_name,
                ProductEmbeddingV3.aug_seed,
                distance.label("distance"),
            )
            .order_by(distance)
            .limit(vote_pool)
        ).subquery()
        # Aggregate per product: vote count
        agg = (
            select(
                top_embs.c.product_id,
                func.count().label("vote_count"),
            )
            .group_by(top_embs.c.product_id)
            .subquery()
        )
        # Get the row with best distance per product to get aug_name/aug_seed
        best_per_product = (
            select(
                top_embs.c.product_id,
                top_embs.c.embedding_id,
                top_embs.c.image_path,
                top_embs.c.sample_type,
                top_embs.c.aug_name,
                top_embs.c.aug_seed,
                top_embs.c.distance,
                func.row_number().over(partition_by=top_embs.c.product_id, order_by=top_embs.c.distance).label("rn"),
            )
        ).subquery()
        statement = (
            select(
                Product,
                best_per_product.c.embedding_id,
                best_per_product.c.image_path,
                best_per_product.c.sample_type,
                best_per_product.c.distance,
                best_per_product.c.aug_name,
                best_per_product.c.aug_seed,
                agg.c.vote_count,
            )
            .join(best_per_product, best_per_product.c.product_id == Product.id)
            .join(agg, agg.c.product_id == Product.id)
            .where(best_per_product.c.rn == 1)
            .order_by(best_per_product.c.distance, Product.id)
            .limit(limit)
        )
        rows = (await self.session.execute(statement)).all()
        return [
            ProductCandidateV3(
                product=row[0], embedding_id=row[1], image_path=row[2],
                sample_type=row[3], distance=float(row[4]),
                aug_name=row[5], aug_seed=int(row[6]),
                vote_count=int(row[7]),
            )
            for row in rows
        ]

    async def nearest_v4_with_votes(self, embedding: list[float], limit: int, vote_pool: int = 50) -> list["ProductCandidateV4"]:
        """Vote-ranked nearest search on product_embeddings_v4 (SigLIP 2 768d).
        Guarantees `limit` unique products while counting votes in top `vote_pool` embeddings.
        """
        distance = ProductEmbeddingV4.embedding.cosine_distance(embedding)

        # 1. Best embedding per unique product
        all_ranked = (
            select(
                ProductEmbeddingV4.product_id,
                ProductEmbeddingV4.id.label("embedding_id"),
                ProductEmbeddingV4.image_path,
                ProductEmbeddingV4.sample_type,
                ProductEmbeddingV4.aug_name,
                ProductEmbeddingV4.aug_seed,
                distance.label("distance"),
                func.row_number().over(
                    partition_by=ProductEmbeddingV4.product_id, order_by=distance
                ).label("rn"),
            )
        ).subquery()

        top_products = (
            select(
                all_ranked.c.product_id,
                all_ranked.c.embedding_id,
                all_ranked.c.image_path,
                all_ranked.c.sample_type,
                all_ranked.c.aug_name,
                all_ranked.c.aug_seed,
                all_ranked.c.distance,
            )
            .where(all_ranked.c.rn == 1)
            .order_by(all_ranked.c.distance)
            .limit(limit)
        ).subquery()

        # 2. Count votes across top `vote_pool` embeddings in the whole table
        top_pool = (
            select(ProductEmbeddingV4.product_id)
            .order_by(distance)
            .limit(vote_pool)
        ).subquery()

        votes = (
            select(
                top_pool.c.product_id,
                func.count().label("vote_count"),
            )
            .group_by(top_pool.c.product_id)
            .subquery()
        )

        # 3. Join top products with Product and votes (outer join preserves all top candidates)
        statement = (
            select(
                Product,
                top_products.c.embedding_id,
                top_products.c.image_path,
                top_products.c.sample_type,
                top_products.c.distance,
                top_products.c.aug_name,
                top_products.c.aug_seed,
                func.coalesce(votes.c.vote_count, 1).label("vote_count"),
            )
            .join(top_products, top_products.c.product_id == Product.id)
            .outerjoin(votes, votes.c.product_id == Product.id)
            .order_by(top_products.c.distance, Product.id)
        )
        rows = (await self.session.execute(statement)).all()
        return [
            ProductCandidateV4(
                product=row[0], embedding_id=row[1], image_path=row[2],
                sample_type=row[3], distance=float(row[4]),
                aug_name=row[5], aug_seed=int(row[6]),
                vote_count=int(row[7]),
            )
            for row in rows
        ]
