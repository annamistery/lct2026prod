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
        """Extracts 4 boundary corner anchors from the segmentation polygon for display."""
        pts = np.array(polygon, dtype=np.float32)
        if len(pts) < 4:
            return None

        rect = cv2.minAreaRect(pts)
        box = cv2.boxPoints(rect)
        quad = self.order_points(box)

        quad[:, 0] = np.clip(quad[:, 0], 0, crop_w - 1)
        quad[:, 1] = np.clip(quad[:, 1], 0, crop_h - 1)

        area = abs(float(cv2.contourArea(quad)))
        if area < (crop_w * crop_h * 0.02):
            return None

        return quad

    def rectify_crop(self, crop_image: Image.Image, conf_seg: float | None = None) -> RectificationResult:
        """Rectifies an already cropped label image."""
        started = time.perf_counter()
        crop_w, crop_h = crop_image.size
        seg_started = time.perf_counter()

        polygon = self.segmenter.best_polygon(crop_image, conf=conf_seg) if self.segmenter is not None else None
        seg_ms = round((time.perf_counter() - seg_started) * 1000, 2)

        warp_started = time.perf_counter()
        if polygon is not None:
            quad = self.extract_quadrilateral(polygon, crop_w, crop_h)
            rectified = self.unroll_cylinder_mesh(crop_image, polygon, quad)
            warp_ms = round((time.perf_counter() - warp_started) * 1000, 2)
            return RectificationResult(
                rectified_image=rectified,
                bbox=(0.0, 0.0, float(crop_w), float(crop_h)),
                crop_bbox=(0, 0, int(crop_w), int(crop_h)),
                seg_polygon_orig=[[float(round(float(p[0]), 1)), float(round(float(p[1]), 1))] for p in polygon],
                quad_corners_orig=[[float(round(float(p[0]), 1)), float(round(float(p[1]), 1))] for p in quad] if quad is not None else None,
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
        """Runs full cascade: BBox detect -> Crop with padding -> Seg on Crop -> Cylinder Unroll."""
        started = time.perf_counter()
        orig_w, orig_h = full_image.size

        # 1. BBox Detection
        bbox_started = time.perf_counter()
        box = self.detector.best_box(full_image, conf=conf_detect) if self.detector is not None else None
        bbox_ms = round((time.perf_counter() - bbox_started) * 1000, 2)

        if box is None:
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

        # 2. Segmentation on clean crop
        seg_started = time.perf_counter()
        polygon = self.segmenter.best_polygon(crop_img, conf=conf_seg) if self.segmenter is not None else None
        seg_ms = round((time.perf_counter() - seg_started) * 1000, 2)

        # 3. True Cylindrical Unrolling into 256x256 canonical matrix
        warp_started = time.perf_counter()
        if polygon is not None:
            quad = self.extract_quadrilateral(polygon, crop_w, crop_h)
            rectified = self.unroll_cylinder_mesh(crop_img, polygon, quad)
            warp_ms = round((time.perf_counter() - warp_started) * 1000, 2)

            poly_orig = [[float(round(float(p[0]) + crop_xtl, 1)), float(round(float(p[1]) + crop_ytl, 1))] for p in polygon]
            quad_orig = [[float(round(float(p[0]) + crop_xtl, 1)), float(round(float(p[1]) + crop_ytl, 1))] for p in quad] if quad is not None else None

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
            seg_polygon_orig=None,
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
        quad: np.ndarray | None = None,
    ) -> Image.Image:
        """Truly unrolls the cylindrical label into a flat canonical 256x256 matrix.

        Straightens the top/bottom cylinder curves into horizontal lines,
        unbends the cylindrical perspective along the horizontal axis,
        and isolates ONLY the label surface without the bottle background.
        """
        try:
            crop_np = np.array(crop_image.convert("RGB"))
            ch, cw = crop_np.shape[:2]
            pts = np.array(polygon, dtype=np.float32)

            if len(pts) < 4:
                return crop_image.convert("RGB").resize((self.target_size, self.target_size), Image.Resampling.LANCZOS)

            # 1. Orientation via minAreaRect (align upright cylinder axis)
            rect = cv2.minAreaRect(pts)
            center, (bw, bh), angle = rect
            if bw < bh:
                tilt = angle
            else:
                tilt = angle + 90.0 if angle < 0 else angle - 90.0

            if abs(tilt) > 1.0 and abs(tilt) < 45.0:
                M_rot = cv2.getRotationMatrix2D(center, tilt, 1.0)
                crop_rot = cv2.warpAffine(
                    crop_np, M_rot, (cw, ch), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE
                )
                ones = np.ones((len(pts), 1), dtype=np.float32)
                pts_rot = np.dot(M_rot, np.hstack([pts, ones]).T).T
            else:
                crop_rot = crop_np
                pts_rot = pts

            # 2. Raster Mask with spike & notch suppression
            mask = np.zeros((ch, cw), dtype=np.uint8)
            cv2.fillPoly(mask, [np.int32(pts_rot)], 255)

            k_w = max(5, int(cw * 0.015) | 1)
            k_h = max(5, int(ch * 0.015) | 1)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_w, k_h))
            mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel)

            # 3. Detect valid solid columns
            col_sums = np.sum(mask_clean > 0, axis=0)
            max_col = np.max(col_sums)
            valid_cols = np.where(col_sums >= max(20, int(0.15 * max_col)))[0]
            if len(valid_cols) < 20:
                if quad is not None:
                    return self._warp_perspective(crop_image, quad)
                return crop_image.convert("RGB").resize((self.target_size, self.target_size), Image.Resampling.LANCZOS)

            xmin = int(valid_cols[0])
            xmax = int(valid_cols[-1])
            w_label = xmax - xmin

            # 4. Extract clean top and bottom boundary curves per column
            xs_prof = []
            y_tops = []
            y_bots = []
            for x in range(xmin, xmax + 1):
                rows = np.where(mask_clean[:, x] > 0)[0]
                if len(rows) >= 10:
                    xs_prof.append(x)
                    y_tops.append(rows[0])
                    y_bots.append(rows[-1])

            if len(xs_prof) < 15:
                if quad is not None:
                    return self._warp_perspective(crop_image, quad)
                return crop_image.convert("RGB").resize((self.target_size, self.target_size), Image.Resampling.LANCZOS)

            xs_prof = np.array(xs_prof, dtype=np.float32)
            y_tops = np.array(y_tops, dtype=np.float32)
            y_bots = np.array(y_bots, dtype=np.float32)

            # 5. Parabolic fitting of top and bottom cylinder arcs
            x_mid = (xmin + xmax) * 0.5
            X_rel = xs_prof - x_mid

            poly_top = np.polyfit(X_rel, y_tops, 2)
            poly_bot = np.polyfit(X_rel, y_bots, 2)

            # 6. Build Remap Grid:
            # - Straightens top curve into line Y=0
            # - Straightens bottom curve into line Y=255
            # - Unbends cylinder compression via sin(theta)/sin(theta_0)
            cols = np.linspace(0.0, 1.0, self.target_size, dtype=np.float32)
            theta_0 = 0.82
            s_norm = 2.0 * cols - 1.0
            u = np.sin(s_norm * theta_0) / np.sin(theta_0)
            grid_x_rel = u * (w_label * 0.5)
            grid_x = x_mid + grid_x_rel

            c_top = np.polyval(poly_top, grid_x_rel)
            c_bot = np.polyval(poly_bot, grid_x_rel)

            mean_h = float(np.mean(y_bots - y_tops))
            min_clearance = max(20.0, 0.70 * mean_h)
            c_bot = np.maximum(c_bot, c_top + min_clearance)

            V = np.linspace(0.0, 1.0, self.target_size, dtype=np.float32)[:, None]
            map_x = np.tile(grid_x[None, :], (self.target_size, 1)).astype(np.float32)
            map_y = ((1.0 - V) * c_top[None, :] + V * c_bot[None, :]).astype(np.float32)

            map_x = np.clip(map_x, 0, cw - 1)
            map_y = np.clip(map_y, 0, ch - 1)

            unrolled = cv2.remap(
                crop_rot,
                map_x,
                map_y,
                interpolation=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_REPLICATE,
            )
            return Image.fromarray(unrolled)

        except Exception:
            if quad is not None:
                return self._warp_perspective(crop_image, quad)
            return crop_image.convert("RGB").resize((self.target_size, self.target_size), Image.Resampling.LANCZOS)

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
