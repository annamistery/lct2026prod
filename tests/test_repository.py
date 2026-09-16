from app.db.repositories.products import ProductRepository


def test_repository_annotations_resolve_without_builtin_shadowing():
    assert ProductRepository.list_products.__name__ == "list_products"
    assert ProductRepository.nearest.__annotations__["embedding"] == "list[float]"
