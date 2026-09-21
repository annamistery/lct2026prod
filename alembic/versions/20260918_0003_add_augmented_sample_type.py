"""Add augmented to product_embeddings sample_type check constraint.

Revision ID: 20260918_0003
Revises: 20260916_0002
Create Date: 2026-09-18 10:20:00.000000
"""

from alembic import op
import sqlalchemy as sa

revision = "20260918_0003"
down_revision = "20260916_0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Drop existing check constraint if exists, then recreate with augmented
    conn = op.get_bind()
    conn.execute(sa.text("""
        ALTER TABLE product_embeddings DROP CONSTRAINT IF EXISTS ck_product_embeddings_sample_type;
        ALTER TABLE product_embeddings DROP CONSTRAINT IF EXISTS ck_product_embeddings_ck_product_embeddings_sample_type;
        ALTER TABLE product_embeddings ADD CONSTRAINT ck_product_embeddings_sample_type CHECK (sample_type IN ('catalog', 'augmented', 'real', 'customer'));
    """))


def downgrade() -> None:
    conn = op.get_bind()
    conn.execute(sa.text("""
        ALTER TABLE product_embeddings DROP CONSTRAINT IF EXISTS ck_product_embeddings_sample_type;
        ALTER TABLE product_embeddings DROP CONSTRAINT IF EXISTS ck_product_embeddings_ck_product_embeddings_sample_type;
        ALTER TABLE product_embeddings ADD CONSTRAINT ck_product_embeddings_sample_type CHECK (sample_type IN ('catalog', 'real', 'customer'));
    """))
