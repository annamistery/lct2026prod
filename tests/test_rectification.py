import numpy as np
from PIL import Image

from app.services.rectification import RectificationService


def test_rectification_service_order_points():
    pts = np.array([[100, 100], [0, 0], [100, 0], [0, 100]], dtype=np.float32)
    ordered = RectificationService.order_points(pts)
    assert np.allclose(ordered[0], [0, 0])      # TL
    assert np.allclose(ordered[1], [100, 0])    # TR
    assert np.allclose(ordered[2], [100, 100])  # BR
    assert np.allclose(ordered[3], [0, 100])    # BL


def test_rectification_service_unroll_rectangular():
    rectifier = RectificationService(target_size=256)
    img = Image.new("RGB", (400, 600), color=(200, 50, 50))
    poly = [(50.0, 50.0), (350.0, 50.0), (350.0, 550.0), (50.0, 550.0)]
    warped = rectifier.unroll_cylinder_mesh(img, poly)
    assert isinstance(warped, Image.Image)
    assert warped.size == (256, 256)


def test_rectification_service_unroll_curved_with_spikes():
    rectifier = RectificationService(target_size=256)
    img = Image.new("RGB", (500, 800), color=(50, 150, 200))
    # Synthetic curved cylinder label with spike noise
    xs = np.linspace(50, 450, 40)
    top_y = 100 + 30 * np.sin(np.linspace(0, np.pi, 40))
    bot_y = 700 + 30 * np.sin(np.linspace(0, np.pi, 40))
    # Inject outward spike
    top_y[10] -= 40
    # Inject inward notch
    bot_y[25] -= 50

    top_pts = list(zip(xs, top_y))
    bot_pts = list(zip(xs[::-1], bot_y[::-1]))
    poly = top_pts + bot_pts

    warped = rectifier.unroll_cylinder_mesh(img, poly)
    assert isinstance(warped, Image.Image)
    assert warped.size == (256, 256)
