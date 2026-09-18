"""Add augmented to product_embeddings sample_type check constraint.

Revision ID: 20260918_0003
Revises: 20260916_0002
Create Date: 2026-09-18 10:20:00.000000
"""

from alembic import op

revision = "20260918_0003"
down_revision = "20260916_0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Drop existing check constraint and replace with one including 'augmented'
    op.drop_constraint("ck_product_embeddings_sample_type", "product_embeddings", type_="check")
    op.create_check_constraint(
        "ck_product_embeddings_sample_type",
        "product_embeddings",
        "sample_type IN ('catalog', 'augmented', 'real', 'customer')",
    )


def downgrade() -> None:
    op.drop_constraint("ck_product_embeddings_sample_type", "product_embeddings", type_="check")
    op.create_check_constraint(
        "ck_product_embeddings_sample_type",
        "product_embeddings",
        "sample_type IN ('catalog', 'real', 'customer')",
    )
