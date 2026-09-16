from PIL import Image

from app.pipelines.search.v1.reranking import SiftReranker


def test_sift_fallback_for_featureless_images():
    image = Image.new("RGB", (256, 256), "white")
    result = SiftReranker().compare(image, image, 0.8)
    assert result.inliers == 0
    assert result.inlier_ratio == 0
    assert result.score == 0.32
