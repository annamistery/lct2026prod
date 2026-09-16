"""Separate reusable product embeddings from product metadata."""

from alembic import op
import pgvector.sqlalchemy
import sqlalchemy as sa

revision = "20260916_0002"
down_revision = "20260916_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "product_embeddings",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("product_id", sa.Uuid(), nullable=False),
        sa.Column("image_path", sa.String(length=500), nullable=False),
        sa.Column("sample_type", sa.String(length=20), nullable=False),
        sa.Column("embedding", pgvector.sqlalchemy.Vector(dim=384), nullable=False),
        sa.Column("embedding_model", sa.String(length=200), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint("sample_type IN ('catalog', 'real', 'customer')", name="ck_product_embeddings_sample_type"),
        sa.ForeignKeyConstraint(["product_id"], ["products.id"], name="fk_product_embeddings_product_id_products", ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name="pk_product_embeddings"),
    )
    op.create_index("ix_product_embeddings_product_id", "product_embeddings", ["product_id"])
    op.create_index("ix_product_embeddings_sample_type", "product_embeddings", ["sample_type"])
    op.execute(
        """
        INSERT INTO product_embeddings (id, product_id, image_path, sample_type, embedding, embedding_model)
        SELECT gen_random_uuid(), id, label_image_path, 'catalog', embedding, embedding_model
        FROM products
        """
    )
    op.drop_column("products", "embedding")
    op.drop_column("products", "embedding_model")


def downgrade() -> None:
    op.add_column("products", sa.Column("embedding_model", sa.String(length=200), nullable=True))
    op.add_column("products", sa.Column("embedding", pgvector.sqlalchemy.Vector(dim=384), nullable=True))
    op.execute(
        """
        UPDATE products AS product
        SET embedding = source.embedding, embedding_model = source.embedding_model
        FROM product_embeddings AS source
        WHERE source.product_id = product.id
          AND source.id = (
              SELECT candidate.id
              FROM product_embeddings AS candidate
              WHERE candidate.product_id = product.id
              ORDER BY candidate.created_at, candidate.id
              LIMIT 1
          )
        """
    )
    op.alter_column("products", "embedding", nullable=False)
    op.alter_column("products", "embedding_model", nullable=False)
    op.drop_index("ix_product_embeddings_sample_type", table_name="product_embeddings")
    op.drop_index("ix_product_embeddings_product_id", table_name="product_embeddings")
    op.drop_table("product_embeddings")
