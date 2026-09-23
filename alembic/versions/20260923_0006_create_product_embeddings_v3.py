"""Create product_embeddings_v3 table for DINOv2-base 768d embeddings.

Revision ID: 20260923_0006
Revises: 20260921_0005
Create Date: 2026-09-23 12:00:00.000000
"""

from alembic import op
import pgvector.sqlalchemy
import sqlalchemy as sa

revision = "20260923_0006"
down_revision = "20260921_0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()
    conn.execute(sa.text("""
        CREATE TABLE IF NOT EXISTS product_embeddings_v3 (
            id UUID PRIMARY KEY,
            product_id UUID NOT NULL REFERENCES products(id) ON DELETE CASCADE,
            image_path VARCHAR(500) NOT NULL,
            sample_type VARCHAR(20) NOT NULL CHECK (sample_type IN ('catalog', 'augmented', 'real', 'customer')),
            aug_name VARCHAR(50) NOT NULL DEFAULT 'catalog',
            aug_seed INT NOT NULL DEFAULT 0,
            embedding vector(768) NOT NULL,
            embedding_model VARCHAR(200) NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        CREATE INDEX IF NOT EXISTS ix_product_embeddings_v3_product_id ON product_embeddings_v3(product_id);
        CREATE INDEX IF NOT EXISTS ix_product_embeddings_v3_sample_type ON product_embeddings_v3(sample_type);
    """))


def downgrade() -> None:
    op.drop_index("ix_product_embeddings_v3_sample_type", table_name="product_embeddings_v3")
    op.drop_index("ix_product_embeddings_v3_product_id", table_name="product_embeddings_v3")
    op.drop_table("product_embeddings_v3")
