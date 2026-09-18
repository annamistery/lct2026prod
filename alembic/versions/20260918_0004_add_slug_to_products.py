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
    op.add_column("products", sa.Column("slug", sa.String(length=300), nullable=True))
    op.create_index("ix_products_slug", "products", ["slug"], unique=True)


def downgrade() -> None:
    op.drop_index("ix_products_slug", table_name="products")
    op.drop_column("products", "slug")
