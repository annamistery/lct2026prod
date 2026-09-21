import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Index, String, Text, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class Product(Base):
    __tablename__ = "products"
    __table_args__ = (
        Index("ix_products_title", "title"),
        Index("ix_products_manufacturer", "manufacturer"),
        Index("ix_products_created_at", "created_at"),
        Index("ix_products_slug", "slug", unique=True),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    slug: Mapped[str | None] = mapped_column(String(300), nullable=True)
    title: Mapped[str] = mapped_column(String(300), nullable=False)
    manufacturer: Mapped[str] = mapped_column(String(300), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="", nullable=False)
    source_image_path: Mapped[str] = mapped_column(String(500), nullable=False)
    label_image_path: Mapped[str] = mapped_column(String(500), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    embeddings: Mapped[list["ProductEmbedding"]] = relationship(back_populates="product", cascade="all, delete-orphan")
    embeddings_v2: Mapped[list["ProductEmbeddingV2"]] = relationship(back_populates="product", cascade="all, delete-orphan")


class ProductEmbedding(Base):
    __tablename__ = "product_embeddings"
    __table_args__ = (
        CheckConstraint("sample_type IN ('catalog', 'augmented', 'real', 'customer')", name="ck_product_embeddings_sample_type"),
        Index("ix_product_embeddings_product_id", "product_id"),
        Index("ix_product_embeddings_sample_type", "sample_type"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    product_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("products.id", ondelete="CASCADE"), nullable=False)
    image_path: Mapped[str] = mapped_column(String(500), nullable=False)
    sample_type: Mapped[str] = mapped_column(String(20), nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(384), nullable=False)
    embedding_model: Mapped[str] = mapped_column(String(200), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    product: Mapped[Product] = relationship(back_populates="embeddings")


class ProductEmbeddingV2(Base):
    __tablename__ = "product_embeddings_v2"
    __table_args__ = (
        CheckConstraint("sample_type IN ('catalog', 'augmented', 'real', 'customer')", name="ck_product_embeddings_v2_sample_type"),
        Index("ix_product_embeddings_v2_product_id", "product_id"),
        Index("ix_product_embeddings_v2_sample_type", "sample_type"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    product_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("products.id", ondelete="CASCADE"), nullable=False)
    image_path: Mapped[str] = mapped_column(String(500), nullable=False)
    sample_type: Mapped[str] = mapped_column(String(20), nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(384), nullable=False)
    embedding_model: Mapped[str] = mapped_column(String(200), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    product: Mapped[Product] = relationship(back_populates="embeddings_v2")
