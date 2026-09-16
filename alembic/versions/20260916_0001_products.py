"""Create the initial production catalog and pgvector schema."""

from alembic import op
import pgvector.sqlalchemy
import sqlalchemy as sa

revision = "20260916_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "products",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("title", sa.String(length=300), nullable=False),
        sa.Column("manufacturer", sa.String(length=300), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("source_image_path", sa.String(length=500), nullable=False),
        sa.Column("label_image_path", sa.String(length=500), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_products"),
    )
    op.create_index("ix_products_title", "products", ["title"])
    op.create_index("ix_products_manufacturer", "products", ["manufacturer"])
    op.create_index("ix_products_created_at", "products", ["created_at"])
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


def downgrade() -> None:
    op.drop_index("ix_product_embeddings_sample_type", table_name="product_embeddings")
    op.drop_index("ix_product_embeddings_product_id", table_name="product_embeddings")
    op.drop_table("product_embeddings")
    op.drop_index("ix_products_created_at", table_name="products")
    op.drop_index("ix_products_manufacturer", table_name="products")
    op.drop_index("ix_products_title", table_name="products")
    op.drop_table("products")
