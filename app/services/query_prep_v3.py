"""Query image preparation for v3 search pipeline.

Flow: full photo → YOLO detect → padded crop → YOLO seg → clean polygon →
      convex hull → bounding rect of segmentation → crop → proportional
      resize to 518×518 letterbox with zero-fill.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from app.services.detector import DetectorService
from app.services.segmenter import SegmenterService


@dataclass
class PrepResultV3:
    image: Image.Image
    bbox: tuple[int, int, int, int] | None
    seg_polygon: list[list[float]] | None
    is_fallback: bool
    method: str
    detect_ms: float
    seg_ms: float
    prep_ms: float
    total_ms: float


def letterbox_pil(img: Image.Image, target_size: int = 518) -> Image.Image:
    """Proportional resize to target_size×target_size with zero-fill (black padding)."""
    w, h = img.size
    if w == 0 or h == 0:
        return Image.new("RGB", (target_size, target_size), (0, 0, 0))
    scale = target_size / max(w, h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = img.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    paste_x = (target_size - new_w) // 2
    paste_y = (target_size - new_h) // 2
    canvas.paste(resized, (paste_x, paste_y))
    return canvas


def _clean_polygon(polygon: list[tuple[float, float]]) -> np.ndarray:
    """Clean segmentation polygon: morphological smoothing + convex hull."""
    pts = np.array(polygon, dtype=np.float32)
    if len(pts) < 3:
        return pts

    # Build a tight mask from the polygon, apply morphological closing to remove spikes
    x_min, y_min = pts.min(axis=0).astype(int)
    x_max, y_max = pts.max(axis=0).astype(int)
    pad = 10
    w = x_max - x_min + 2 * pad
    h = y_max - y_min + 2 * pad
    if w < 3 or h < 3:
        return pts

    shifted = (pts - [x_min - pad, y_min - pad]).astype(np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [shifted], 255)

    # Morphological closing to smooth spikes
    kernel_size = max(3, min(w, h) // 30) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return pts

    largest = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(largest)
    hull_pts = hull.reshape(-1, 2).astype(np.float32) + [x_min - pad, y_min - pad]
    return hull_pts


class QueryPrepV3:
    def __init__(
        self,
        detector: DetectorService | None = None,
        segmenter: SegmenterService | None = None,
        target_size: int = 518,
        padding_ratio: float = 0.05,
    ):
        self.detector = detector
        self.segmenter = segmenter
        self.target_size = target_size
        self.padding_ratio = padding_ratio

    def prepare(self, full_image: Image.Image) -> PrepResultV3:
        """Full pipeline: detect → seg → clean → crop → letterbox 518."""
        t0 = time.perf_counter()

        # 1. YOLO detect → bbox
        detect_t = time.perf_counter()
        bbox = None
        if self.detector:
            bbox = self.detector.best_box(full_image)
        detect_ms = (time.perf_counter() - detect_t) * 1000

        if bbox is None:
            # No detection → letterbox the whole image
            img_518 = letterbox_pil(full_image, self.target_size)
            total_ms = (time.perf_counter() - t0) * 1000
            return PrepResultV3(
                image=img_518, bbox=None, seg_polygon=None, is_fallback=True,
                method="letterbox_full", detect_ms=detect_ms, seg_ms=0, prep_ms=0, total_ms=total_ms,
            )

        # 2. Padded crop
        x1, y1, x2, y2 = bbox
        iw, ih = full_image.size
        pad_w = (x2 - x1) * self.padding_ratio
        pad_h = (y2 - y1) * self.padding_ratio
        cx1 = max(0, int(x1 - pad_w))
        cy1 = max(0, int(y1 - pad_h))
        cx2 = min(iw, int(x2 + pad_w))
        cy2 = min(ih, int(y2 + pad_h))
        crop = full_image.crop((cx1, cy1, cx2, cy2))
        int_bbox = (cx1, cy1, cx2, cy2)

        # 3. YOLO seg → polygon
        seg_t = time.perf_counter()
        polygon = None
        if self.segmenter:
            polygon = self.segmenter.best_polygon(crop)
        seg_ms = (time.perf_counter() - seg_t) * 1000

        return self._crop_and_letterbox(crop, polygon, int_bbox, detect_ms, seg_ms, t0)

    def prepare_crop(self, crop_image: Image.Image) -> PrepResultV3:
        """From an already-cropped label image (no detection step)."""
        t0 = time.perf_counter()

        # Segmentation on the crop directly
        seg_t = time.perf_counter()
        polygon = None
        if self.segmenter:
            polygon = self.segmenter.best_polygon(crop_image)
        seg_ms = (time.perf_counter() - seg_t) * 1000

        return self._crop_and_letterbox(crop_image, polygon, None, 0, seg_ms, t0)

    def _crop_and_letterbox(
        self,
        crop: Image.Image,
        polygon: list[tuple[float, float]] | None,
        bbox: tuple[int, int, int, int] | None,
        detect_ms: float,
        seg_ms: float,
        t0: float,
    ) -> PrepResultV3:
        prep_t = time.perf_counter()

        if polygon is None or len(polygon) < 3:
            # No segmentation → letterbox the crop
            img_518 = letterbox_pil(crop, self.target_size)
            total_ms = (time.perf_counter() - t0) * 1000
            prep_ms = (time.perf_counter() - prep_t) * 1000
            return PrepResultV3(
                image=img_518, bbox=bbox, seg_polygon=None, is_fallback=True,
                method="letterbox_crop", detect_ms=detect_ms, seg_ms=seg_ms,
                prep_ms=prep_ms, total_ms=total_ms,
            )

        # 4. Clean polygon → convex hull
        hull = _clean_polygon(polygon)
        seg_polygon_list = hull.tolist()

        # 5. Bounding rect of segmentation area
        hull_int = hull.astype(np.int32)
        sx, sy, sw, sh = cv2.boundingRect(hull_int)
        crop_arr = np.array(crop.convert("RGB"))

        # 6. Mask out everything outside the hull
        mask = np.zeros(crop_arr.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [hull_int], 255)
        masked = cv2.bitwise_and(crop_arr, crop_arr, mask=mask)

        # 7. Crop to bounding rect of segmentation
        ch, cw = crop_arr.shape[:2]
        sx = max(0, sx)
        sy = max(0, sy)
        ex = min(cw, sx + sw)
        ey = min(ch, sy + sh)
        seg_crop = masked[sy:ey, sx:ex]

        if seg_crop.size == 0:
            img_518 = letterbox_pil(crop, self.target_size)
            total_ms = (time.perf_counter() - t0) * 1000
            prep_ms = (time.perf_counter() - prep_t) * 1000
            return PrepResultV3(
                image=img_518, bbox=bbox, seg_polygon=seg_polygon_list, is_fallback=True,
                method="letterbox_crop_empty_seg", detect_ms=detect_ms, seg_ms=seg_ms,
                prep_ms=prep_ms, total_ms=total_ms,
            )

        # 8. Proportional resize to target_size×target_size letterbox (zero-fill)
        seg_pil = Image.fromarray(seg_crop)
        img_518 = letterbox_pil(seg_pil, self.target_size)

        prep_ms = (time.perf_counter() - prep_t) * 1000
        total_ms = (time.perf_counter() - t0) * 1000
        return PrepResultV3(
            image=img_518, bbox=bbox, seg_polygon=seg_polygon_list, is_fallback=False,
            method="seg_letterbox", detect_ms=detect_ms, seg_ms=seg_ms,
            prep_ms=prep_ms, total_ms=total_ms,
        )
