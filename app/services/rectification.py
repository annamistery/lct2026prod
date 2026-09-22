import itertools
import time
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from app.services.detector import DetectorService
from app.services.segmenter import SegmenterService


@dataclass
class RectificationResult:
    rectified_image: Image.Image
    bbox: tuple[float, float, float, float] | None
    crop_bbox: tuple[int, int, int, int] | None
    seg_polygon_orig: list[list[float]] | None
    quad_corners_orig: list[list[float]] | None
    is_fallback: bool
    method: str
    bbox_ms: float
    seg_ms: float
    warp_ms: float
    total_ms: float


class RectificationService:
    def __init__(
        self,
        detector: DetectorService | None = None,
        segmenter: SegmenterService | None = None,
        target_size: int = 256,
        padding_ratio: float = 0.05,
        fill_color: tuple[int, int, int] = (20, 20, 20),
    ):
        self.detector = detector
        self.segmenter = segmenter
        self.target_size = target_size
        self.padding_ratio = padding_ratio
        self.fill_color = fill_color

    @staticmethod
    def order_points(pts: np.ndarray) -> np.ndarray:
        """Orders 4 points into canonical order: TL, TR, BR, BL."""
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def _clean_convex_hull(
        self,
        polygon: list[tuple[float, float]],
        crop_shape: tuple[int, int],
    ) -> np.ndarray | None:
        """Return a cleaned convex hull of the segmentation polygon."""
        pts = np.array(polygon, dtype=np.float32)
        if len(pts) < 4:
            return None

        h, w = crop_shape
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [np.int32(pts)], 255)

        k = max(3, int(min(w, h) * 0.01) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        cnt = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(cnt)[:, 0, :]
        if len(hull) < 4:
            return None
        return hull.astype(np.float32)

    def _max_area_inscribed_quadrilateral(self, hull_pts: np.ndarray) -> np.ndarray | None:
        """Find the four hull vertices that form the largest-area quadrilateral.

        This is used as the perspective frame of the label. We simplify the hull
        first, then brute-force all 4-point combinations; the vertex count is small
        after simplification.
        """
        pts = hull_pts.astype(np.float32)
        if len(pts) < 4:
            return None

        closed = pts if np.allclose(pts[0], pts[-1]) else np.vstack([pts, pts[0]])
        perimeter = cv2.arcLength(closed, True)
        eps = 0.005 * perimeter
        approx = cv2.approxPolyDP(closed, eps, True).reshape(-1, 2)
        n = len(approx)
        if n < 4:
            return None

        if n > 60:
            # Fallback to minAreaRect if the hull is too noisy to brute-force.
            rect = cv2.minAreaRect(pts)
            return cv2.boxPoints(rect)

        best_area = -1.0
        best_quad = None
        for combo in itertools.combinations(range(n), 4):
            quad = approx[np.array(combo)]
            area = abs(float(cv2.contourArea(quad)))
            if area > best_area:
                best_area = area
                best_quad = quad

        if best_quad is None:
            return None
        return best_quad.astype(np.float32)

    def _orient_frame_portrait(self, quad: np.ndarray) -> np.ndarray:
        """Reorder the 4 corner frame so the label's long side is vertical."""
        ordered = self.order_points(quad)
        top_len = float(np.linalg.norm(ordered[1] - ordered[0]))
        bot_len = float(np.linalg.norm(ordered[2] - ordered[3]))
        left_len = float(np.linalg.norm(ordered[3] - ordered[0]))
        right_len = float(np.linalg.norm(ordered[2] - ordered[1]))
        width = (top_len + bot_len) / 2.0
        height = (left_len + right_len) / 2.0

        if width > height:
            # Rotate clockwise: old top edge becomes the new left edge (vertical)
            return np.array(
                [ordered[1], ordered[2], ordered[3], ordered[0]], dtype=np.float32
            )
        return ordered

    def extract_perspective_frame(
        self,
        polygon: list[tuple[float, float]],
        crop_shape: tuple[int, int],
    ) -> np.ndarray | None:
        """Extract a robust 4-point perspective frame from the segmentation polygon."""
        hull = self._clean_convex_hull(polygon, crop_shape)
        if hull is None:
            return None

        quad = self._max_area_inscribed_quadrilateral(hull)
        if quad is None:
            return None

        ordered = self._orient_frame_portrait(quad)

        h, w = crop_shape
        ordered[:, 0] = np.clip(ordered[:, 0], 0, w - 1)
        ordered[:, 1] = np.clip(ordered[:, 1], 0, h - 1)

        area = abs(float(cv2.contourArea(ordered)))
        if area < (w * h * 0.01):
            return None

        return ordered

    def _rectify_perspective_letterbox(
        self,
        crop_image: Image.Image,
        src_pts: np.ndarray,
    ) -> Image.Image:
        """Map the perspective frame to a 256x256 canvas preserving aspect ratio."""
        crop_np = np.array(crop_image.convert("RGB"))

        top_len = float(np.linalg.norm(src_pts[1] - src_pts[0]))
        bot_len = float(np.linalg.norm(src_pts[2] - src_pts[3]))
        left_len = float(np.linalg.norm(src_pts[3] - src_pts[0]))
        right_len = float(np.linalg.norm(src_pts[2] - src_pts[1]))

        src_w = (top_len + bot_len) / 2.0
        src_h = (left_len + right_len) / 2.0
        aspect = src_w / max(1e-6, src_h)

        target = float(self.target_size)
        if aspect >= 1.0:
            dst_w = target
            dst_h = max(1.0, target / aspect)
        else:
            dst_h = target
            dst_w = max(1.0, target * aspect)

        cx = target / 2.0
        cy = target / 2.0
        dst_pts = np.float32(
            [
                [cx - dst_w / 2, cy - dst_h / 2],
                [cx + dst_w / 2, cy - dst_h / 2],
                [cx + dst_w / 2, cy + dst_h / 2],
                [cx - dst_w / 2, cy + dst_h / 2],
            ]
        )

        matrix = cv2.getPerspectiveTransform(src_pts.astype(np.float32), dst_pts)
        warped = cv2.warpPerspective(
            crop_np,
            matrix,
            (self.target_size, self.target_size),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=self.fill_color,
        )
        return Image.fromarray(warped)

    def rectify_crop(
        self,
        crop_image: Image.Image,
        conf_seg: float | None = None,
    ) -> RectificationResult:
        started = time.perf_counter()
        crop_w, crop_h = crop_image.size
        seg_started = time.perf_counter()
        polygon = (
            self.segmenter.best_polygon(crop_image, conf=conf_seg)
            if self.segmenter is not None
            else None
        )
        seg_ms = round((time.perf_counter() - seg_started) * 1000, 2)

        warp_started = time.perf_counter()
        if polygon is not None:
            frame = self.extract_perspective_frame(polygon, (crop_h, crop_w))
            if frame is not None:
                rectified = self._rectify_perspective_letterbox(crop_image, frame)
                method = "perspective_letterbox"
                quad = frame
            else:
                rectified = crop_image.convert("RGB").resize(
                    (self.target_size, self.target_size), Image.Resampling.LANCZOS
                )
                method = "fallback_resize"
                quad = None
        else:
            rectified = crop_image.convert("RGB").resize(
                (self.target_size, self.target_size), Image.Resampling.LANCZOS
            )
            method = "fallback_resize"
            quad = None

        warp_ms = round((time.perf_counter() - warp_started) * 1000, 2)
        total_ms = round((time.perf_counter() - started) * 1000, 2)

        poly_orig = (
            [[float(round(p[0], 1)), float(round(p[1], 1))] for p in polygon]
            if polygon
            else None
        )
        quad_orig = (
            [[float(round(p[0], 1)), float(round(p[1], 1))] for p in quad]
            if quad is not None
            else None
        )

        return RectificationResult(
            rectified_image=rectified,
            bbox=(0.0, 0.0, float(crop_w), float(crop_h)),
            crop_bbox=(0, 0, int(crop_w), int(crop_h)),
            seg_polygon_orig=poly_orig,
            quad_corners_orig=quad_orig,
            is_fallback=method == "fallback_resize",
            method=method,
            bbox_ms=0.0,
            seg_ms=float(seg_ms),
            warp_ms=float(warp_ms),
            total_ms=float(total_ms),
        )

    def rectify(
        self,
        full_image: Image.Image,
        conf_detect: float | None = None,
        conf_seg: float | None = None,
    ) -> RectificationResult:
        started = time.perf_counter()
        orig_w, orig_h = full_image.size

        bbox_started = time.perf_counter()
        box = self.detector.best_box(full_image, conf=conf_detect) if self.detector is not None else None
        bbox_ms = round((time.perf_counter() - bbox_started) * 1000, 2)

        if box is None:
            result = self.rectify_crop(full_image, conf_seg=conf_seg)
            result.bbox_ms = float(bbox_ms)
            result.total_ms = round((time.perf_counter() - started) * 1000, 2)
            return result

        xtl, ytl, xbr, ybr = box
        pad_x = int((xbr - xtl) * self.padding_ratio)
        pad_y = int((ybr - ytl) * self.padding_ratio)
        crop_xtl = max(0, int(xtl) - pad_x)
        crop_ytl = max(0, int(ytl) - pad_y)
        crop_xbr = min(orig_w, int(xbr) + pad_x)
        crop_ybr = min(orig_h, int(ybr) + pad_y)

        crop_img = full_image.crop((crop_xtl, crop_ytl, crop_xbr, crop_ybr))
        crop_w, crop_h = crop_img.size

        seg_started = time.perf_counter()
        polygon = (
            self.segmenter.best_polygon(crop_img, conf=conf_seg)
            if self.segmenter is not None
            else None
        )
        seg_ms = round((time.perf_counter() - seg_started) * 1000, 2)

        warp_started = time.perf_counter()
        method = "fallback_resize"
        rectified = crop_img.convert("RGB").resize(
            (self.target_size, self.target_size), Image.Resampling.LANCZOS
        )
        quad = None

        if polygon is not None:
            frame = self.extract_perspective_frame(polygon, (crop_h, crop_w))
            if frame is not None:
                rectified = self._rectify_perspective_letterbox(crop_img, frame)
                method = "perspective_letterbox"
                quad = frame

        warp_ms = round((time.perf_counter() - warp_started) * 1000, 2)
        total_ms = round((time.perf_counter() - started) * 1000, 2)

        poly_orig = (
            [[float(round(p[0] + crop_xtl, 1)), float(round(p[1] + crop_ytl, 1))] for p in polygon]
            if polygon
            else None
        )
        quad_orig = (
            [[float(round(p[0] + crop_xtl, 1)), float(round(p[1] + crop_ytl, 1))] for p in quad]
            if quad is not None
            else None
        )

        return RectificationResult(
            rectified_image=rectified,
            bbox=(
                float(round(float(xtl), 1)),
                float(round(float(ytl), 1)),
                float(round(float(xbr), 1)),
                float(round(float(ybr), 1)),
            ),
            crop_bbox=(int(crop_xtl), int(crop_ytl), int(crop_xbr), int(crop_ybr)),
            seg_polygon_orig=poly_orig,
            quad_corners_orig=quad_orig,
            is_fallback=method == "fallback_resize",
            method=method,
            bbox_ms=float(bbox_ms),
            seg_ms=float(seg_ms),
            warp_ms=float(warp_ms),
            total_ms=float(total_ms),
        )
