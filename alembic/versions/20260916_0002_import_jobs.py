"""Add durable background import jobs."""

from alembic import op
import sqlalchemy as sa

revision = "20260916_0002"
down_revision = "20260916_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "import_jobs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("batch_id", sa.String(length=100), nullable=False),
        sa.Column("manifest_name", sa.String(length=255), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("total_items", sa.Integer(), nullable=False),
        sa.Column("completed_items", sa.Integer(), nullable=False),
        sa.Column("failed_items", sa.Integer(), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint("status IN ('pending', 'running', 'completed', 'failed')", name="ck_import_jobs_status"),
        sa.PrimaryKeyConstraint("id", name="pk_import_jobs"),
    )
    op.create_index("ix_import_jobs_status", "import_jobs", ["status"])
    op.create_index("ix_import_jobs_created_at", "import_jobs", ["created_at"])
    op.create_table(
        "import_items",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("job_id", sa.Uuid(), nullable=False),
        sa.Column("row_number", sa.Integer(), nullable=False),
        sa.Column("image_path", sa.String(length=500), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("product_id", sa.Uuid(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.CheckConstraint("status IN ('pending', 'completed', 'failed')", name="ck_import_items_status"),
        sa.ForeignKeyConstraint(["job_id"], ["import_jobs.id"], name="fk_import_items_job_id_import_jobs", ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["product_id"], ["products.id"], name="fk_import_items_product_id_products", ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id", name="pk_import_items"),
    )
    op.create_index("ix_import_items_job_id", "import_items", ["job_id"])


def downgrade() -> None:
    op.drop_index("ix_import_items_job_id", table_name="import_items")
    op.drop_table("import_items")
    op.drop_index("ix_import_jobs_created_at", table_name="import_jobs")
    op.drop_index("ix_import_jobs_status", table_name="import_jobs")
    op.drop_table("import_jobs")
