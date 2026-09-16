from pgvector.sqlalchemy import Vector

from app.db.models import Product


def test_product_embedding_dimension():
    column_type = Product.__table__.c.embedding.type
    assert isinstance(column_type, Vector)
    assert column_type.dim == 384
