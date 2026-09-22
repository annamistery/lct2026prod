import numpy as np
import pytest
from PIL import Image

from app.services.rectification import RectificationService


def _apply_perspective(image: np.ndarray, src_rect: np.ndarray, pad: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """Apply a random perspective warp to a planar rectangle and return the warped
    image together with the four projected corner points in the warped image.
    """
    h, w = image.shape[:2]

    # random but moderate perspective distortion
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

    src = np.float32(
        [
            [0, 0],
            [w, 0],
            [w, h],
            [0, h],
        ]
    )

    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(
        image,
        matrix,
        (w + 2 * pad, h + 2 * pad),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return warped, dst


def test_order_points_rectangle():
    pts = np.array([[100, 100], [0, 0], [100, 0], [0, 100]], dtype=np.float32)
    ordered = RectificationService.order_points(pts)
    assert np.allclose(ordered[0], [0, 0])
    assert np.allclose(ordered[1], [100, 0])
    assert np.allclose(ordered[2], [100, 100])
    assert np.allclose(ordered[3], [0, 100])


def test_extract_perspective_frame_recovers_warped_rectangle():
    """A perspective-warped known rectangle should have its four corners recovered."""
    import cv2

    rectifier = RectificationService(target_size=256)
    h, w = 300, 180
    flat = np.zeros((h, w, 3), dtype=np.uint8)
    flat[:, :90] = (255, 0, 0)
    flat[:, 90:] = (0, 255, 0)

    warped, true_corners = _apply_perspective(flat)
    cw, ch = warped.shape[1], warped.shape[0]
    polygon = cv2.convexHull(true_corners.reshape(-1, 1, 2).astype(np.float32))[:, 0, :].tolist()

    frame = rectifier.extract_perspective_frame(polygon, (ch, cw))
    assert frame is not None
    assert len(frame) == 4

    # The recovered frame should be close to the true projected rectangle corners.
    # Order may differ, so compare as unordered sets.
    distances = np.linalg.norm(true_corners[:, None, :] - frame[None, :, :], axis=2)
    min_dist = np.min(distances, axis=1)
    assert np.max(min_dist) < 5.0


def test_rectify_perspective_preserves_aspect_ratio_and_orient():
    """A tall label should remain tall (portrait) in the output canvas."""
    import cv2

    rectifier = RectificationService(target_size=256, fill_color=(20, 20, 20))
    h, w = 400, 150
    flat = np.zeros((h, w, 3), dtype=np.uint8)
    flat[:, 0::30] = (0, 200, 200)

    warped, true_corners = _apply_perspective(flat)
    cw, ch = warped.shape[1], warped.shape[0]
    polygon = cv2.convexHull(true_corners.reshape(-1, 1, 2).astype(np.float32))[:, 0, :].tolist()

    frame = rectifier.extract_perspective_frame(polygon, (ch, cw))
    assert frame is not None
    result = rectifier._rectify_perspective_letterbox(Image.fromarray(warped), frame)
    assert result.size == (256, 256)

    arr = np.array(result)
    bg = np.array(rectifier.fill_color)
    content_mask = np.any(arr != bg, axis=2)

    label_columns = np.sum(content_mask, axis=0) > 10
    label_rows = np.sum(content_mask, axis=1) > 10
    content_w = int(np.sum(label_columns))
    content_h = int(np.sum(label_rows))

    # Aspect ratio: input label is 150/400 = 0.375. Output should be portrait.
    assert content_h > content_w
    # Rough aspect preservation: width/height should be within ~0.2 of 0.375
    out_aspect = content_w / max(1, content_h)
    assert 0.2 < out_aspect < 0.55


def test_landscape_label_rotated_to_portrait():
    """If the camera was held sideways, the physical label is still tall."""
    import cv2

    rectifier = RectificationService(target_size=256, fill_color=(20, 20, 20))
    # physical label 400x150, but rotated 90 degrees in source
    h, w = 150, 400
    flat = np.zeros((h, w, 3), dtype=np.uint8)
    flat[:, 0::40] = (200, 0, 200)

    warped, true_corners = _apply_perspective(flat)
    cw, ch = warped.shape[1], warped.shape[0]
    polygon = cv2.convexHull(true_corners.reshape(-1, 1, 2).astype(np.float32))[:, 0, :].tolist()

    frame = rectifier.extract_perspective_frame(polygon, (ch, cw))
    assert frame is not None
    result = rectifier._rectify_perspective_letterbox(Image.fromarray(warped), frame)
    assert result.size == (256, 256)

    arr = np.array(result)
    bg = np.array(rectifier.fill_color)
    content_mask = np.any(arr != bg, axis=2)
    content_w = int(np.sum(np.sum(content_mask, axis=0) > 10))
    content_h = int(np.sum(np.sum(content_mask, axis=1) > 10))

    assert content_h > content_w


def test_no_polygon_returns_none():
    rectifier = RectificationService(target_size=256)
    assert rectifier.extract_perspective_frame([], (100, 100)) is None
