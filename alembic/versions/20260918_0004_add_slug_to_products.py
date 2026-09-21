"""Add slug to products table.

Revision ID: 20260918_0004
Revises: 20260918_0003
Create Date: 2026-09-18 17:30:00.000000
"""

from alembic import op
import sqlalchemy as sa

revision = "20260918_0004"
down_revision = "20260918_0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()
    conn.execute(sa.text("""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns 
                WHERE table_name='products' AND column_name='slug'
            ) THEN
                ALTER TABLE products ADD COLUMN slug VARCHAR(300);
            END IF;
        END $$;
        CREATE UNIQUE INDEX IF NOT EXISTS ix_products_slug ON products(slug);
    """))


def downgrade() -> None:
    op.drop_index("ix_products_slug", table_name="products")
    op.drop_column("products", "slug")
