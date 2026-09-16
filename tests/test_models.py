from pgvector.sqlalchemy import Vector

from app.db.models import ImportItem, ImportJob, Product, ProductEmbedding


def test_embedding_is_separate_from_product_metadata():
    assert "embedding" not in Product.__table__.c
    assert "embedding_model" not in Product.__table__.c


def test_product_embedding_dimension_and_foreign_key():
    column_type = ProductEmbedding.__table__.c.embedding.type
    assert isinstance(column_type, Vector)
    assert column_type.dim == 384
    foreign_key = next(iter(ProductEmbedding.__table__.c.product_id.foreign_keys))
    assert foreign_key.target_fullname == "products.id"


def test_product_supports_multiple_embeddings():
    relationship = Product.__mapper__.relationships["embeddings"]
    assert relationship.uselist is True
    assert relationship.cascade.delete_orphan is True


def test_import_jobs_have_durable_items():
    assert ImportJob.__tablename__ == "import_jobs"
    assert ImportItem.__tablename__ == "import_items"
    assert next(iter(ImportItem.__table__.c.job_id.foreign_keys)).target_fullname == "import_jobs.id"
