import numpy as np
import pytest
from PIL import Image

from app.services.rectification import RectificationService


def test_order_points_rectangle():
    pts = np.array([[100, 100], [0, 0], [100, 0], [0, 100]], dtype=np.float32)
    ordered = RectificationService.order_points(pts)
    assert np.allclose(ordered[0], [0, 0])      # TL
    assert np.allclose(ordered[1], [100, 0])    # TR
    assert np.allclose(ordered[2], [100, 100])  # BR
    assert np.allclose(ordered[3], [0, 100])    # BL


def test_order_points_no_singularities():
    """Points with identical sums or diffs must never collapse into duplicate vertices."""
    pts = np.array([[113.0, 354.0], [725.0, 648.0], [320.0, 1680.0], [50.0, 1410.0]], dtype=np.float32)
    ordered = RectificationService.order_points(pts)
    assert len(ordered) == 4
    # Check that all 4 points are strictly unique
    for i in range(4):
        for j in range(i + 1, 4):
            assert not np.allclose(ordered[i], ordered[j])


def test_unroll_oriented_cylinder_synthetic_rectangle():
    """Synthetic rectangular cylinder mask should unroll cleanly without distortions."""
    rectifier = RectificationService(target_size=256)
    w, h = 300, 400
    img = Image.new("RGB", (w, h), (128, 128, 128))

    # A curved cylinder patch: top arch curves down, bottom curves down
    xs = np.linspace(20, 280, 50)
    top_y = 50 + 20 * np.sin(np.pi * (xs - 20) / 260)
    bot_y = 350 + 20 * np.sin(np.pi * (xs - 20) / 260)

    poly = []
    for x, y in zip(xs, top_y):
        poly.append((float(x), float(y)))
    for x, y in zip(reversed(xs), reversed(bot_y)):
        poly.append((float(x), float(y)))

    unrolled, quad = rectifier.unroll_oriented_cylinder(img, poly)
    assert unrolled is not None
    assert unrolled.size == (256, 256)
    assert quad is not None
    assert len(quad) == 4


def test_no_polygon_returns_fallback():
    rectifier = RectificationService(target_size=256)
    img = Image.new("RGB", (100, 100), (255, 255, 255))
    res = rectifier.rectify_crop(img)
    assert res.is_fallback is True
    assert res.method == "fallback_resize"
    assert res.rectified_image.size == (256, 256)
