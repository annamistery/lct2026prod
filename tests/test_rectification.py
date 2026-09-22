import cv2
import numpy as np
import pytest
from PIL import Image

from app.services.rectification import RectificationService, _ASPECT_MIN, _ASPECT_MAX


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply_perspective(image: np.ndarray, pad: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """Apply a reproducible moderate perspective warp.

    Returns (warped_image, true_dst_corners_4x2).
    """
    h, w = image.shape[:2]
    rng = np.random.RandomState(42)
    max_shift = 0.12
    dst = np.float32(
        [
            [pad + rng.uniform(-max_shift, max_shift) * w, pad + rng.uniform(-max_shift, max_shift) * h],
            [pad + w + rng.uniform(-max_shift, max_shift) * w, pad + rng.uniform(-max_shift, max_shift) * h],
            [pad + w + rng.uniform(-max_shift, max_shift) * w, pad + h + rng.uniform(-max_shift, max_shift) * h],
            [pad + rng.uniform(-max_shift, max_shift) * w, pad + h + rng.uniform(-max_shift, max_shift) * h],
        ]
    )
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(
        image,
        matrix,
        (w + 2 * pad, h + 2 * pad),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return warped, dst


def _hull_polygon(corners: np.ndarray) -> list:
    hull = cv2.convexHull(corners.reshape(-1, 1, 2).astype(np.float32))[:, 0, :]
    return hull.tolist()


# ---------------------------------------------------------------------------
# order_points
# ---------------------------------------------------------------------------

def test_order_points_rectangle():
    pts = np.array([[100, 100], [0, 0], [100, 0], [0, 100]], dtype=np.float32)
    ordered = RectificationService.order_points(pts)
    assert np.allclose(ordered[0], [0, 0])    # TL
    assert np.allclose(ordered[1], [100, 0])  # TR
    assert np.allclose(ordered[2], [100, 100])# BR
    assert np.allclose(ordered[3], [0, 100])  # BL


# ---------------------------------------------------------------------------
# _extract_quad_from_hull
# ---------------------------------------------------------------------------

def test_extract_quad_rectangle_hull():
    """A simple rectangular hull should produce 4 corners via approxPolyDP."""
    rectifier = RectificationService()
    hull = np.float32([[0, 0], [200, 0], [200, 300], [0, 300]])
    quad, method = rectifier._extract_quad_from_hull(hull)
    assert quad is not None
    assert quad.shape == (4, 2)
    assert method in ("approxpoly", "minarearect")


def test_extract_quad_star_hull():
    """A star-shaped hull (many points) must still collapse to 4 corners."""
    rectifier = RectificationService()
    # Build a star polygon: 20 points alternating at radius 100 and 50
    n = 20
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    radii = np.where(np.arange(n) % 2 == 0, 100.0, 50.0)
    cx, cy = 150.0, 150.0
    star = np.column_stack(
        [cx + radii * np.cos(angles), cy + radii * np.sin(angles)]
    ).astype(np.float32)
    # Take convex hull to simulate what _clean_convex_hull returns
    hull = cv2.convexHull(star)[:, 0, :]
    quad, method = rectifier._extract_quad_from_hull(hull)
    assert quad is not None
    assert quad.shape == (4, 2), f"expected 4 corners, got {len(quad)} via {method}"


def test_extract_quad_too_few_points_returns_none():
    rectifier = RectificationService()
    hull = np.float32([[0, 0], [10, 0], [5, 5]])  # only 3 pts
    quad, method = rectifier._extract_quad_from_hull(hull)
    assert quad is None


# ---------------------------------------------------------------------------
# _orient_quad
# ---------------------------------------------------------------------------

def test_orient_quad_landscape_becomes_portrait():
    """A landscape frame (width > height) should be rotated to portrait."""
    rectifier = RectificationService()
    # Wide rectangle: 400 wide, 150 tall — TL(0,0) TR(400,0) BR(400,150) BL(0,150)
    quad = np.float32([[0, 0], [400, 0], [400, 150], [0, 150]])
    hull_pts = quad.copy()
    oriented = rectifier._orient_quad(quad, hull_pts)
    top_len = float(np.linalg.norm(oriented[1] - oriented[0]))
    left_len = float(np.linalg.norm(oriented[3] - oriented[0]))
    # After orientation the long side should be vertical
    assert left_len >= top_len, (
        f"Expected portrait orientation (left_len={left_len:.1f} >= top_len={top_len:.1f})"
    )


def test_orient_quad_portrait_unchanged():
    """A portrait frame (height > width) should not be rotated."""
    rectifier = RectificationService()
    quad = np.float32([[0, 0], [150, 0], [150, 400], [0, 400]])
    hull_pts = quad.copy()
    oriented = rectifier._orient_quad(quad, hull_pts)
    top_len = float(np.linalg.norm(oriented[1] - oriented[0]))
    left_len = float(np.linalg.norm(oriented[3] - oriented[0]))
    assert left_len >= top_len


# ---------------------------------------------------------------------------
# _rectify_perspective_letterbox — aspect ratio clamping
# ---------------------------------------------------------------------------

def test_letterbox_aspect_clamped_min():
    """An extremely narrow label (aspect < _ASPECT_MIN) must be clamped."""
    rectifier = RectificationService(target_size=256, fill_color=(0, 0, 0))
    # src_pts forming a quad with src_w=40, src_h=400 → aspect≈0.1 < 0.4
    src_pts = np.float32([[0, 0], [40, 0], [40, 400], [0, 400]])
    img = Image.new("RGB", (500, 500), (255, 255, 255))
    _, true_aspect, clamped = rectifier._rectify_perspective_letterbox(img, src_pts)
    assert abs(true_aspect - 40 / 400) < 0.02
    assert clamped is True


def test_letterbox_aspect_clamped_max():
    """An extremely wide label (aspect > _ASPECT_MAX) must be clamped."""
    rectifier = RectificationService(target_size=256, fill_color=(0, 0, 0))
    # src_pts forming a quad with src_w=800, src_h=100 → aspect=8.0 > 2.5
    src_pts = np.float32([[0, 0], [800, 0], [800, 100], [0, 100]])
    img = Image.new("RGB", (1000, 300), (128, 128, 128))
    _, true_aspect, clamped = rectifier._rectify_perspective_letterbox(img, src_pts)
    assert true_aspect > _ASPECT_MAX
    assert clamped is True


def test_letterbox_normal_aspect_not_clamped():
    """A normal aspect ratio (within bounds) should not be clamped."""
    rectifier = RectificationService(target_size=256, fill_color=(0, 0, 0))
    # src_pts: 150w × 300h → aspect=0.5 (within [0.4, 2.5])
    src_pts = np.float32([[0, 0], [150, 0], [150, 300], [0, 300]])
    img = Image.new("RGB", (300, 400), (200, 100, 50))
    _, true_aspect, clamped = rectifier._rectify_perspective_letterbox(img, src_pts)
    assert abs(true_aspect - 150 / 300) < 0.02
    assert clamped is False


# ---------------------------------------------------------------------------
# extract_perspective_frame (integration)
# ---------------------------------------------------------------------------

def test_extract_perspective_frame_recovers_warped_rectangle():
    """A perspective-warped rectangle's four corners should be recovered."""
    rectifier = RectificationService(target_size=256)
    h, w = 300, 180
    flat = np.zeros((h, w, 3), dtype=np.uint8)
    flat[:, :90] = (255, 0, 0)
    flat[:, 90:] = (0, 255, 0)

    warped, true_corners = _apply_perspective(flat)
    cw, ch = warped.shape[1], warped.shape[0]
    polygon = _hull_polygon(true_corners)

    frame, frame_method = rectifier.extract_perspective_frame(polygon, (ch, cw))
    assert frame is not None
    assert len(frame) == 4
    assert frame_method in ("approxpoly", "minarearect")

    # Recovered corners should be close to the true projected corners
    distances = np.linalg.norm(true_corners[:, None, :] - frame[None, :, :], axis=2)
    assert np.max(np.min(distances, axis=1)) < 8.0


def test_rectify_perspective_preserves_aspect_ratio_and_orient():
    """A tall label must remain portrait in the output canvas."""
    rectifier = RectificationService(target_size=256, fill_color=(20, 20, 20))
    h, w = 400, 150
    flat = np.zeros((h, w, 3), dtype=np.uint8)
    flat[:, 0::30] = (0, 200, 200)

    warped, true_corners = _apply_perspective(flat)
    cw, ch = warped.shape[1], warped.shape[0]
    polygon = _hull_polygon(true_corners)

    frame, _ = rectifier.extract_perspective_frame(polygon, (ch, cw))
    assert frame is not None
    result, true_aspect, _ = rectifier._rectify_perspective_letterbox(
        Image.fromarray(warped), frame
    )
    assert result.size == (256, 256)

    arr = np.array(result)
    bg = np.array(rectifier.fill_color)
    content_mask = np.any(arr != bg, axis=2)
    content_w = int(np.sum(np.sum(content_mask, axis=0) > 10))
    content_h = int(np.sum(np.sum(content_mask, axis=1) > 10))

    assert content_h > content_w, (
        f"Expected portrait layout but content_w={content_w}, content_h={content_h}"
    )
    out_aspect = content_w / max(1, content_h)
    # true_aspect ≈ 0.375, clamped to _ASPECT_MIN=0.4 → expected content aspect ≈ 0.4
    assert 0.2 < out_aspect < 0.8, f"Aspect {out_aspect:.2f} out of expected range"


def test_landscape_label_rotated_to_portrait():
    """A physically tall label shot sideways should be oriented portrait."""
    rectifier = RectificationService(target_size=256, fill_color=(20, 20, 20))
    h, w = 150, 400  # landscape crop of a portrait label
    flat = np.zeros((h, w, 3), dtype=np.uint8)
    flat[:, 0::40] = (200, 0, 200)

    warped, true_corners = _apply_perspective(flat)
    cw, ch = warped.shape[1], warped.shape[0]
    polygon = _hull_polygon(true_corners)

    frame, _ = rectifier.extract_perspective_frame(polygon, (ch, cw))
    assert frame is not None
    result, _, _ = rectifier._rectify_perspective_letterbox(
        Image.fromarray(warped), frame
    )
    assert result.size == (256, 256)

    arr = np.array(result)
    bg = np.array(rectifier.fill_color)
    content_mask = np.any(arr != bg, axis=2)
    content_w = int(np.sum(np.sum(content_mask, axis=0) > 10))
    content_h = int(np.sum(np.sum(content_mask, axis=1) > 10))
    assert content_h > content_w, (
        f"Expected portrait layout but content_w={content_w}, content_h={content_h}"
    )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_no_polygon_returns_none():
    rectifier = RectificationService(target_size=256)
    frame, method = rectifier.extract_perspective_frame([], (100, 100))
    assert frame is None
    assert method == "fallback_resize"


def test_rectification_result_fields():
    """RectificationResult must carry frame_method, aspect_ratio, aspect_clamped."""
    rectifier = RectificationService(target_size=64)
    img = Image.new("RGB", (200, 400), (100, 150, 200))
    result = rectifier.rectify_crop(img)
    # No segmenter → fallback resize
    assert result.method == "fallback_resize"
    assert isinstance(result.frame_method, str)
    assert isinstance(result.aspect_ratio, float)
    assert isinstance(result.aspect_clamped, bool)
    assert result.rectified_image.size == (64, 64)
