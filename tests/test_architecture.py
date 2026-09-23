from app.api.router import router


def test_only_search_is_versioned():
    paths = {route.path for route in router.routes}
    assert "/api/v1/search" in paths
    assert "/api/v1/search-from-crop" in paths
    assert "/api/v4/search" in paths
    assert "/api/v4/search-from-crop" in paths
    assert "/api/v4/eval/predict" in paths
    assert "/api/products" in paths
    assert "/api/imports" in paths
    assert "/api/imports/{job_id}" in paths
    assert "/api/ping" in paths
    assert "/api/sommelier/ask" in paths
    assert "/api/sommelier/wine/{slug}" in paths
    assert "/api/v1/products" not in paths
    assert "/api/search" not in paths
    assert "/api/v1/sommelier/ask" not in paths
