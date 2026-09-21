"""Create product_embeddings_v2 table.

Revision ID: 20260921_0005
Revises: 20260918_0004
Create Date: 2026-09-21 12:00:00.000000
"""

from alembic import op
import pgvector.sqlalchemy
import sqlalchemy as sa

revision = "20260921_0005"
down_revision = "20260918_0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "product_embeddings_v2",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("product_id", sa.Uuid(), nullable=False),
        sa.Column("image_path", sa.String(length=500), nullable=False),
        sa.Column("sample_type", sa.String(length=20), nullable=False),
        sa.Column("embedding", pgvector.sqlalchemy.Vector(dim=384), nullable=False),
        sa.Column("embedding_model", sa.String(length=200), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint("sample_type IN ('catalog', 'augmented', 'real', 'customer')", name="ck_product_embeddings_v2_sample_type"),
        sa.ForeignKeyConstraint(["product_id"], ["products.id"], name="fk_product_embeddings_v2_product_id_products", ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name="pk_product_embeddings_v2"),
    )
    op.create_index("ix_product_embeddings_v2_product_id", "product_embeddings_v2", ["product_id"])
    op.create_index("ix_product_embeddings_v2_sample_type", "product_embeddings_v2", ["sample_type"])


def downgrade() -> None:
    op.drop_index("ix_product_embeddings_v2_sample_type", table_name="product_embeddings_v2")
    op.drop_index("ix_product_embeddings_v2_product_id", table_name="product_embeddings_v2")
    op.drop_table("product_embeddings_v2")
