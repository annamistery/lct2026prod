import time
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from app.services.detector import DetectorService
from app.services.segmenter import SegmenterService

# Aspect ratio clamp bounds: prevents extreme letterboxing on DINOv2 input
_ASPECT_MIN = 0.4
_ASPECT_MAX = 2.5


@dataclass
class RectificationResult:
    rectified_image: Image.Image
    bbox: tuple[float, float, float, float] | None
    crop_bbox: tuple[int, int, int, int] | None
    seg_polygon_orig: list[list[float]] | None
    quad_corners_orig: list[list[float]] | None
    is_fallback: bool
    method: str
    frame_method: str
    aspect_ratio: float
    aspect_clamped: bool
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
        """Orders 4 points into canonical clockwise order: [TL, TR, BR, BL].

        Guarantees that all 4 points are unique and form a non-self-intersecting
        quadrilateral by sorting points by polar angle around their centroid.
        """
        pts = np.asarray(pts, dtype=np.float32)
        if len(pts) != 4:
            return pts

        center = pts.mean(axis=0)
        # Polar angles: -pi to +pi
        angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
        # Clockwise sort in image coordinates (Y down): angle increases clockwise
        order = np.argsort(angles)
        pts_sorted = pts[order]

        # Top-Left direction vector in image space is (-1, -1)
        target_tl = np.array([-1.0, -1.0], dtype=np.float32)
        vecs = pts_sorted - center
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms < 1e-6] = 1.0
        vecs_norm = vecs / norms
        tl_idx = int(np.argmax(np.dot(vecs_norm, target_tl)))

        # Roll so TL is at index 0
        return np.roll(pts_sorted, -tl_idx, axis=0)

    def _clean_convex_hull(
        self,
        polygon: list[tuple[float, float]],
        crop_shape: tuple[int, int],
    ) -> np.ndarray | None:
        """Return a morphologically cleaned convex hull of the segmentation polygon."""
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

    def _extract_quad_from_hull(self, hull_pts: np.ndarray) -> tuple[np.ndarray | None, str]:
        """Reduce the convex hull to exactly 4 corners via adaptive approxPolyDP."""
        pts = hull_pts.astype(np.float32)
        if len(pts) < 4:
            return None, "fallback_resize"

        closed = pts if np.allclose(pts[0], pts[-1]) else np.vstack([pts, pts[0]])
        perimeter = cv2.arcLength(closed, True)
        if perimeter < 1e-3:
            return None, "fallback_resize"

        eps_step = 0.005 * perimeter
        eps_max = 0.08 * perimeter
        eps = eps_step
        while eps <= eps_max + 1e-9:
            approx = cv2.approxPolyDP(closed, eps, True).reshape(-1, 2)
            if len(approx) == 4:
                return approx.astype(np.float32), "approxpoly"
            if len(approx) < 4:
                break
            eps += eps_step

        # Reliable geometric fallback: oriented bounding box of the hull
        rect = cv2.minAreaRect(pts)
        box = cv2.boxPoints(rect)
        return box.astype(np.float32), "minarearect"

    def _orient_quad(
        self,
        quad: np.ndarray,
        hull_pts: np.ndarray,
    ) -> np.ndarray:
        """Ensure the 4-corner frame is oriented so TL/TR form the top edge."""
        ordered = self.order_points(quad)
        top_len = float(np.linalg.norm(ordered[1] - ordered[0]))
        bot_len = float(np.linalg.norm(ordered[2] - ordered[3]))
        left_len = float(np.linalg.norm(ordered[3] - ordered[0]))
        right_len = float(np.linalg.norm(ordered[2] - ordered[1]))
        width = (top_len + bot_len) / 2.0
        height = (left_len + right_len) / 2.0

        if width <= height:
            return ordered

        cy_hull = float(hull_pts[:, 1].mean())
        top_a = (ordered[0] + ordered[3]) / 2.0
        top_b = (ordered[1] + ordered[2]) / 2.0

        if abs(top_a[1] - cy_hull) <= abs(top_b[1] - cy_hull):
            return np.array(
                [ordered[3], ordered[0], ordered[1], ordered[2]], dtype=np.float32
            )
        else:
            return np.array(
                [ordered[1], ordered[2], ordered[3], ordered[0]], dtype=np.float32
            )

    def extract_perspective_frame(
        self,
        polygon: list[tuple[float, float]],
        crop_shape: tuple[int, int],
    ) -> tuple[np.ndarray | None, str]:
        """Extract a robust 4-point perspective frame from the segmentation polygon."""
        hull = self._clean_convex_hull(polygon, crop_shape)
        if hull is None:
            return None, "fallback_resize"

        quad, frame_method = self._extract_quad_from_hull(hull)
        if quad is None:
            return None, "fallback_resize"

        ordered = self._orient_quad(quad, hull)

        h, w = crop_shape
        ordered[:, 0] = np.clip(ordered[:, 0], 0, w - 1)
        ordered[:, 1] = np.clip(ordered[:, 1], 0, h - 1)

        area = abs(float(cv2.contourArea(ordered)))
        if area < (w * h * 0.01):
            return None, "fallback_resize"

        return ordered, frame_method

    def _rectify_perspective_letterbox(
        self,
        crop_image: Image.Image,
        src_pts: np.ndarray,
    ) -> tuple[Image.Image, float, bool]:
        """Map the perspective frame to a 256×256 canvas preserving aspect ratio."""
        crop_np = np.array(crop_image.convert("RGB"))

        top_len = float(np.linalg.norm(src_pts[1] - src_pts[0]))
        bot_len = float(np.linalg.norm(src_pts[2] - src_pts[3]))
        left_len = float(np.linalg.norm(src_pts[3] - src_pts[0]))
        right_len = float(np.linalg.norm(src_pts[2] - src_pts[1]))

        src_w = (top_len + bot_len) / 2.0
        src_h = (left_len + right_len) / 2.0
        true_aspect = src_w / max(1e-6, src_h)

        aspect = true_aspect
        aspect_clamped = False
        if aspect < _ASPECT_MIN:
            aspect = _ASPECT_MIN
            aspect_clamped = True
        elif aspect > _ASPECT_MAX:
            aspect = _ASPECT_MAX
            aspect_clamped = True

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
        return Image.fromarray(warped), true_aspect, aspect_clamped

    def _make_result(
        self,
        *,
        rectified: Image.Image,
        bbox: tuple,
        crop_bbox: tuple,
        polygon: list | None,
        quad: np.ndarray | None,
        method: str,
        frame_method: str,
        aspect_ratio: float,
        aspect_clamped: bool,
        bbox_ms: float,
        seg_ms: float,
        warp_ms: float,
        total_ms: float,
        offset_x: float = 0.0,
        offset_y: float = 0.0,
    ) -> RectificationResult:
        poly_orig = (
            [[float(round(p[0] + offset_x, 1)), float(round(p[1] + offset_y, 1))] for p in polygon]
            if polygon
            else None
        )
        quad_orig = (
            [[float(round(p[0] + offset_x, 1)), float(round(p[1] + offset_y, 1))] for p in quad]
            if quad is not None
            else None
        )
        return RectificationResult(
            rectified_image=rectified,
            bbox=bbox,
            crop_bbox=crop_bbox,
            seg_polygon_orig=poly_orig,
            quad_corners_orig=quad_orig,
            is_fallback=method == "fallback_resize",
            method=method,
            frame_method=frame_method,
            aspect_ratio=aspect_ratio,
            aspect_clamped=aspect_clamped,
            bbox_ms=float(bbox_ms),
            seg_ms=float(seg_ms),
            warp_ms=float(warp_ms),
            total_ms=float(total_ms),
        )

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
        method = "fallback_resize"
        frame_method = "fallback_resize"
        aspect_ratio = 1.0
        aspect_clamped = False
        quad: np.ndarray | None = None

        if polygon is not None:
            frame, frame_method = self.extract_perspective_frame(polygon, (crop_h, crop_w))
            if frame is not None:
                rectified, aspect_ratio, aspect_clamped = self._rectify_perspective_letterbox(
                    crop_image, frame
                )
                method = "perspective_letterbox"
                quad = frame
            else:
                rectified = crop_image.convert("RGB").resize(
                    (self.target_size, self.target_size), Image.Resampling.LANCZOS
                )
        else:
            rectified = crop_image.convert("RGB").resize(
                (self.target_size, self.target_size), Image.Resampling.LANCZOS
            )

        warp_ms = round((time.perf_counter() - warp_started) * 1000, 2)
        total_ms = round((time.perf_counter() - started) * 1000, 2)

        return self._make_result(
            rectified=rectified,
            bbox=(0.0, 0.0, float(crop_w), float(crop_h)),
            crop_bbox=(0, 0, int(crop_w), int(crop_h)),
            polygon=polygon,
            quad=quad,
            method=method,
            frame_method=frame_method,
            aspect_ratio=aspect_ratio,
            aspect_clamped=aspect_clamped,
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
        box = (
            self.detector.best_box(full_image, conf=conf_detect)
            if self.detector is not None
            else None
        )
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
        frame_method = "fallback_resize"
        aspect_ratio = 1.0
        aspect_clamped = False
        quad: np.ndarray | None = None
        rectified = crop_img.convert("RGB").resize(
            (self.target_size, self.target_size), Image.Resampling.LANCZOS
        )

        if polygon is not None:
            frame, frame_method = self.extract_perspective_frame(polygon, (crop_h, crop_w))
            if frame is not None:
                rectified, aspect_ratio, aspect_clamped = self._rectify_perspective_letterbox(
                    crop_img, frame
                )
                method = "perspective_letterbox"
                quad = frame

        warp_ms = round((time.perf_counter() - warp_started) * 1000, 2)
        total_ms = round((time.perf_counter() - started) * 1000, 2)

        return self._make_result(
            rectified=rectified,
            bbox=(
                float(round(float(xtl), 1)),
                float(round(float(ytl), 1)),
                float(round(float(xbr), 1)),
                float(round(float(ybr), 1)),
            ),
            crop_bbox=(int(crop_xtl), int(crop_ytl), int(crop_xbr), int(crop_ybr)),
            polygon=polygon,
            quad=quad,
            method=method,
            frame_method=frame_method,
            aspect_ratio=aspect_ratio,
            aspect_clamped=aspect_clamped,
            bbox_ms=float(bbox_ms),
            seg_ms=float(seg_ms),
            warp_ms=float(warp_ms),
            total_ms=float(total_ms),
            offset_x=float(crop_xtl),
            offset_y=float(crop_ytl),
        )
