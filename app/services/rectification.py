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
    ):
        self.detector = detector
        self.segmenter = segmenter
        self.target_size = target_size
        self.padding_ratio = padding_ratio

    @staticmethod
    def order_points(pts: np.ndarray) -> np.ndarray:
        """Orders 4 points into canonical order: [Top-Left, Top-Right, Bottom-Right, Bottom-Left]."""
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def extract_quadrilateral(
        self,
        polygon: list[tuple[float, float]],
        crop_w: int,
        crop_h: int,
    ) -> np.ndarray | None:
        """Extracts 4 ordered corner points from a segmentation polygon contour."""
        pts = np.array(polygon, dtype=np.float32).reshape(-1, 1, 2)
        if len(pts) < 4:
            return None

        peri = cv2.arcLength(pts, True)
        quad = None

        # 1. Try approxPolyDP with progressive epsilon to find 4 corners
        for factor in (0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.015, 0.01):
            approx = cv2.approxPolyDP(pts, factor * peri, True)
            if len(approx) == 4:
                quad = approx.reshape(4, 2)
                break

        # 2. Fallback to minimal rotated bounding rectangle
        if quad is None:
            rect = cv2.minAreaRect(pts)
            quad = cv2.boxPoints(rect)

        quad = self.order_points(np.array(quad, dtype=np.float32))

        # Clamp points to crop boundaries
        quad[:, 0] = np.clip(quad[:, 0], 0, crop_w - 1)
        quad[:, 1] = np.clip(quad[:, 1], 0, crop_h - 1)

        # Sanity check: quadrilateral area must be at least 2% of crop
        area = abs(float(cv2.contourArea(quad)))
        if area < (crop_w * crop_h * 0.02):
            return None

        return quad

    def rectify_crop(self, crop_image: Image.Image, conf_seg: float | None = None) -> RectificationResult:
        """Rectifies an already cropped label image (e.g. from camera view or manual crop)."""
        started = time.perf_counter()
        crop_w, crop_h = crop_image.size
        seg_started = time.perf_counter()

        polygon = self.segmenter.best_polygon(crop_image, conf=conf_seg) if self.segmenter is not None else None
        seg_ms = round((time.perf_counter() - seg_started) * 1000, 2)

        warp_started = time.perf_counter()
        if polygon is not None:
            quad = self.extract_quadrilateral(polygon, crop_w, crop_h)
            if quad is not None:
                rectified = self.unroll_cylinder_mesh(crop_image, polygon, quad)
                warp_ms = round((time.perf_counter() - warp_started) * 1000, 2)
                return RectificationResult(
                    rectified_image=rectified,
                    bbox=(0.0, 0.0, float(crop_w), float(crop_h)),
                    crop_bbox=(0, 0, int(crop_w), int(crop_h)),
                    seg_polygon_orig=[[float(round(float(p[0]), 1)), float(round(float(p[1]), 1))] for p in polygon],
                    quad_corners_orig=[[float(round(float(p[0]), 1)), float(round(float(p[1]), 1))] for p in quad],
                    is_fallback=False,
                    bbox_ms=0.0,
                    seg_ms=float(seg_ms),
                    warp_ms=float(warp_ms),
                    total_ms=float(round((time.perf_counter() - started) * 1000, 2)),
                )

        # Fallback to direct resize
        rectified = crop_image.convert("RGB").resize((self.target_size, self.target_size), Image.Resampling.LANCZOS)
        warp_ms = round((time.perf_counter() - warp_started) * 1000, 2)
        return RectificationResult(
            rectified_image=rectified,
            bbox=(0.0, 0.0, float(crop_w), float(crop_h)),
            crop_bbox=(0, 0, int(crop_w), int(crop_h)),
            seg_polygon_orig=[[float(round(float(p[0]), 1)), float(round(float(p[1]), 1))] for p in polygon] if polygon else None,
            quad_corners_orig=None,
            is_fallback=True,
            bbox_ms=0.0,
            seg_ms=float(seg_ms),
            warp_ms=float(warp_ms),
            total_ms=float(round((time.perf_counter() - started) * 1000, 2)),
        )

    def rectify(
        self,
        full_image: Image.Image,
        conf_detect: float | None = None,
        conf_seg: float | None = None,
    ) -> RectificationResult:
        """Runs full cascade: BBox detect -> Crop with padding -> Seg on Crop -> 4-point Warp."""
        started = time.perf_counter()
        orig_w, orig_h = full_image.size

        # 1. BBox Detection
        bbox_started = time.perf_counter()
        box = self.detector.best_box(full_image, conf=conf_detect) if self.detector is not None else None
        bbox_ms = round((time.perf_counter() - bbox_started) * 1000, 2)

        if box is None:
            # Fallback on whole image
            return self.rectify_crop(full_image, conf_seg=conf_seg)

        xtl, ytl, xbr, ybr = box
        pad_x = int((xbr - xtl) * self.padding_ratio)
        pad_y = int((ybr - ytl) * self.padding_ratio)
        crop_xtl = max(0, int(xtl) - pad_x)
        crop_ytl = max(0, int(ytl) - pad_y)
        crop_xbr = min(orig_w, int(xbr) + pad_x)
        crop_ybr = min(orig_h, int(ybr) + pad_y)

        crop_img = full_image.crop((crop_xtl, crop_ytl, crop_xbr, crop_ybr))
        crop_w, crop_h = crop_img.size

        # 2. Segmentation on the clean crop
        seg_started = time.perf_counter()
        polygon = self.segmenter.best_polygon(crop_img, conf=conf_seg) if self.segmenter is not None else None
        seg_ms = round((time.perf_counter() - seg_started) * 1000, 2)

        # 3. 4-Corner Extraction & Homography / Mesh Warp
        warp_started = time.perf_counter()
        if polygon is not None:
            quad = self.extract_quadrilateral(polygon, crop_w, crop_h)
            if quad is not None:
                rectified = self.unroll_cylinder_mesh(crop_img, polygon, quad)
                warp_ms = round((time.perf_counter() - warp_started) * 1000, 2)

                # Map polygon and corners back to full original image coordinates
                poly_orig = [[float(round(float(p[0]) + crop_xtl, 1)), float(round(float(p[1]) + crop_ytl, 1))] for p in polygon]
                quad_orig = [[float(round(float(p[0]) + crop_xtl, 1)), float(round(float(p[1]) + crop_ytl, 1))] for p in quad]

                return RectificationResult(
                    rectified_image=rectified,
                    bbox=(float(round(float(xtl), 1)), float(round(float(ytl), 1)), float(round(float(xbr), 1)), float(round(float(ybr), 1))),
                    crop_bbox=(int(crop_xtl), int(crop_ytl), int(crop_xbr), int(crop_ybr)),
                    seg_polygon_orig=poly_orig,
                    quad_corners_orig=quad_orig,
                    is_fallback=False,
                    bbox_ms=float(bbox_ms),
                    seg_ms=float(seg_ms),
                    warp_ms=float(warp_ms),
                    total_ms=float(round((time.perf_counter() - started) * 1000, 2)),
                )

        # Fallback to direct BBox crop without segmentation warp
        fallback_crop = full_image.crop((int(xtl), int(ytl), int(xbr), int(ybr)))
        rectified = fallback_crop.convert("RGB").resize((self.target_size, self.target_size), Image.Resampling.LANCZOS)
        warp_ms = round((time.perf_counter() - warp_started) * 1000, 2)

        return RectificationResult(
            rectified_image=rectified,
            bbox=(float(round(float(xtl), 1)), float(round(float(ytl), 1)), float(round(float(xbr), 1)), float(round(float(ybr), 1))),
            crop_bbox=(int(crop_xtl), int(crop_ytl), int(crop_xbr), int(crop_ybr)),
            seg_polygon_orig=[[float(round(float(p[0]) + crop_xtl, 1)), float(round(float(p[1]) + crop_ytl, 1))] for p in polygon] if polygon else None,
            quad_corners_orig=None,
            is_fallback=True,
            bbox_ms=float(bbox_ms),
            seg_ms=float(seg_ms),
            warp_ms=float(warp_ms),
            total_ms=float(round((time.perf_counter() - started) * 1000, 2)),
        )

    def unroll_cylinder_mesh(
        self,
        crop_image: Image.Image,
        polygon: list[tuple[float, float]],
        quad: np.ndarray,
    ) -> Image.Image:
        """Unrolls curved cylindrical bottle label into a flat canonical square matrix using cv2.remap.

        Straightens top and bottom arc curves and stretches horizontally compressed text on cylinder flanks.
        """
        try:
            crop_np = np.array(crop_image.convert("RGB"))
            crop_h, crop_w = crop_np.shape[:2]

            tl, tr, br, bl = quad[0], quad[1], quad[2], quad[3]

            pts = np.array(polygon, dtype=np.float32)

            # Mid-line Y between top edge (TL-TR) and bottom edge (BL-BR)
            def mid_y(x: float) -> float:
                t_ratio = np.clip((x - tl[0]) / max(1e-5, (tr[0] - tl[0])), 0.0, 1.0)
                b_ratio = np.clip((x - bl[0]) / max(1e-5, (br[0] - bl[0])), 0.0, 1.0)
                y_top = (1.0 - t_ratio) * tl[1] + t_ratio * tr[1]
                y_bot = (1.0 - b_ratio) * bl[1] + b_ratio * br[1]
                return float((y_top + y_bot) * 0.5)

            # Fit 2nd-degree polynomial (arc of cylinder) to top and bottom curves
            # Top boundary points strictly between TL and TR (ignoring outer 10% flanks)
            x_left = tl[0] + 0.05 * (tr[0] - tl[0])
            x_right = tr[0] - 0.05 * (tr[0] - tl[0])

            valid_top = [p for p in pts if x_left <= p[0] <= x_right and p[1] < mid_y(p[0])]
            valid_bot = [p for p in pts if x_left <= p[0] <= x_right and p[1] >= mid_y(p[0])]

            # Always anchor with corner points
            top_pts = [tl, tr] + valid_top
            bot_pts = [bl, br] + valid_bot

            top_arr = np.array(top_pts, dtype=np.float32)
            bot_arr = np.array(bot_pts, dtype=np.float32)

            # Linear fallbacks (exact straight homography edges)
            def lin_top(x: float | np.ndarray) -> np.ndarray:
                t = np.clip((x - tl[0]) / max(1e-5, tr[0] - tl[0]), 0.0, 1.0)
                return (1.0 - t) * tl[1] + t * tr[1]

            def lin_bot(x: float | np.ndarray) -> np.ndarray:
                t = np.clip((x - bl[0]) / max(1e-5, br[0] - bl[0]), 0.0, 1.0)
                return (1.0 - t) * bl[1] + t * br[1]

            poly_top = lin_top
            if len(valid_top) >= 6:
                try:
                    c_top = np.polyfit(top_arr[:, 0], top_arr[:, 1], 2)
                    mid_x = (tl[0] + tr[0]) * 0.5
                    lin_y = (tl[1] + tr[1]) * 0.5
                    # Check that arc curvature is mild (less than 15% of crop height)
                    if abs(np.poly1d(c_top)(mid_x) - lin_y) < (crop_h * 0.15):
                        poly_top = np.poly1d(c_top)
                except Exception:
                    poly_top = lin_top

            poly_bot = lin_bot
            if len(valid_bot) >= 6:
                try:
                    c_bot = np.polyfit(bot_arr[:, 0], bot_arr[:, 1], 2)
                    mid_x = (bl[0] + br[0]) * 0.5
                    lin_y = (bl[1] + br[1]) * 0.5
                    if abs(np.poly1d(c_bot)(mid_x) - lin_y) < (crop_h * 0.15):
                        poly_bot = np.poly1d(c_bot)
                except Exception:
                    poly_bot = lin_bot

            # Build remap grid of size (target_size, target_size)
            W, H = self.target_size, self.target_size
            theta_0 = 0.82  # ~47 degrees cylindrical view angle

            cols = np.linspace(0.0, 1.0, W, dtype=np.float32)
            s_norm = 2.0 * cols - 1.0
            theta = s_norm * theta_0
            factor = np.sin(theta) / np.sin(theta_0)
            u = np.clip((factor + 1.0) * 0.5, 0.0, 1.0)

            x_top = (1.0 - u) * tl[0] + u * tr[0]
            x_bot = (1.0 - u) * bl[0] + u * br[0]

            y_top = np.array([float(poly_top(x)) for x in x_top], dtype=np.float32)
            y_bot = np.array([float(poly_bot(x)) for x in x_bot], dtype=np.float32)

            y_bot = np.maximum(y_bot, y_top + 10.0)

            rows = np.linspace(0.0, 1.0, H, dtype=np.float32)[:, None]

            map_x = ((1.0 - rows) * x_top[None, :] + rows * x_bot[None, :]).astype(np.float32)
            map_y = ((1.0 - rows) * y_top[None, :] + rows * y_bot[None, :]).astype(np.float32)

            map_x = np.clip(map_x, 0, crop_w - 1)
            map_y = np.clip(map_y, 0, crop_h - 1)

            unrolled = cv2.remap(
                crop_np,
                map_x,
                map_y,
                interpolation=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_REPLICATE,
            )
            return Image.fromarray(unrolled)
        except Exception:
            return self._warp_perspective(crop_image, quad)

    def _warp_perspective(self, crop_image: Image.Image, quad: np.ndarray) -> Image.Image:
        """Applies 4-point perspective warp into canonical square matrix."""
        try:
            crop_np = np.array(crop_image.convert("RGB"))
            dst_pts = np.float32([
                [0, 0],
                [self.target_size - 1, 0],
                [self.target_size - 1, self.target_size - 1],
                [0, self.target_size - 1],
            ])
            matrix = cv2.getPerspectiveTransform(np.float32(quad), dst_pts)
            warped = cv2.warpPerspective(
                crop_np,
                matrix,
                (self.target_size, self.target_size),
                flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_REPLICATE,
            )
            return Image.fromarray(warped)
        except Exception:
            return crop_image.convert("RGB").resize((self.target_size, self.target_size), Image.Resampling.LANCZOS)
