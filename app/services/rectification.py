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
        """Extracts 4 stable boundary corner anchors from the segmentation polygon.

        Uses oriented bounding box (minAreaRect) or convex hull extrema to guarantee
        that corners never collapse into a single point on vertical/straight boundaries.
        """
        pts = np.array(polygon, dtype=np.float32)
        if len(pts) < 4:
            return None

        # Calculate oriented bounding box
        rect = cv2.minAreaRect(pts)
        box = cv2.boxPoints(rect)  # 4 points
        box = self.order_points(box)

        # Refine corners by snapping to nearest actual polygon points along direction rays
        refined_corners = []
        for c in box:
            dists = np.linalg.norm(pts - c, axis=1)
            nearest_idx = int(np.argmin(dists))
            refined_corners.append(pts[nearest_idx])

        quad = self.order_points(np.array(refined_corners, dtype=np.float32))

        # Clamp points to crop boundaries
        quad[:, 0] = np.clip(quad[:, 0], 0, crop_w - 1)
        quad[:, 1] = np.clip(quad[:, 1], 0, crop_h - 1)

        # Sanity check: area must be at least 2% of crop
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
        quad: np.ndarray | None = None,
    ) -> Image.Image:
        """Unrolls the label into a canonical square matrix (target_size x target_size).

        Implements a robust hybrid rectification pipeline:
        1. Spike / Notch suppression via adaptive morphological opening/closing.
        2. Solidity assessment (handles complex figurative crowns vs cylindrical labels).
        3. Column-wise cylindrical remap with quadratic curvature smoothing and guaranteed
           positive Jacobian (eliminates all fold/wrinkle defects).
        4. Fallback to robust perspective homography if non-cylindrical or irregular.
        """
        try:
            crop_np = np.array(crop_image.convert("RGB"))
            crop_h, crop_w = crop_np.shape[:2]
            pts = np.array(polygon, dtype=np.float32)

            if len(pts) < 4:
                return crop_image.convert("RGB").resize((self.target_size, self.target_size), Image.Resampling.LANCZOS)

            # Check solidity
            hull = cv2.convexHull(pts)
            area = float(cv2.contourArea(pts))
            hull_area = float(cv2.contourArea(hull))
            solidity = area / max(1.0, hull_area)

            # If label has highly irregular decorative contours (e.g. domes/spires, solidity < 0.78),
            # use robust perspective homography on the main body
            if solidity < 0.78 and quad is not None:
                return self._warp_perspective(crop_image, quad)

            # 1. Orientation via minAreaRect
            rect = cv2.minAreaRect(pts)
            center, (bw, bh), angle = rect
            if bw < bh:
                tilt = angle
            else:
                tilt = angle + 90.0 if angle < 0 else angle - 90.0

            # Correct tilt if bottle is angled > 1 deg
            if abs(tilt) > 1.0 and abs(tilt) < 45.0:
                M_rot = cv2.getRotationMatrix2D(center, tilt, 1.0)
                crop_rot = cv2.warpAffine(
                    crop_np, M_rot, (crop_w, crop_h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE
                )
                ones = np.ones((len(pts), 1), dtype=np.float32)
                pts_rot = np.dot(M_rot, np.hstack([pts, ones]).T).T
            else:
                crop_rot = crop_np
                pts_rot = pts

            # 2. Raster Mask from polygon with Spike and Notch suppression
            mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
            cv2.fillPoly(mask, [np.int32(pts_rot)], 255)

            # Adaptive kernel size (1.5% of crop dimensions)
            k_w = max(5, int(crop_w * 0.015) | 1)
            k_h = max(5, int(crop_h * 0.015) | 1)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_w, k_h))
            mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel)

            # 3. Solid column detection (filter out empty fringe borders)
            col_sums = np.sum(mask_clean > 0, axis=0)
            valid_cols = np.where(col_sums >= max(15, int(0.12 * np.max(col_sums))))[0]
            if len(valid_cols) < 20:
                if quad is not None:
                    return self._warp_perspective(crop_image, quad)
                return crop_image.convert("RGB").resize((self.target_size, self.target_size), Image.Resampling.LANCZOS)

            xmin = int(valid_cols[0])
            xmax = int(valid_cols[-1])
            w_label = xmax - xmin

            # 4. Extract Top and Bottom profiles per column
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

            # 5. Robust Quadratic Curvature Smoothing
            x_mid = (xmin + xmax) * 0.5
            X_rel = xs_prof - x_mid

            poly_top = np.polyfit(X_rel, y_tops, 2)
            poly_bot = np.polyfit(X_rel, y_bots, 2)

            # 6. Build Cylindrical Remap Grid (target_size x target_size)
            cols = np.linspace(0.0, 1.0, self.target_size, dtype=np.float32)
            theta_0 = 0.82  # ~47 deg cylindrical view
            s_norm = 2.0 * cols - 1.0
            u = np.sin(s_norm * theta_0) / np.sin(theta_0)
            grid_x_rel = u * (w_label * 0.5)
            grid_x = x_mid + grid_x_rel

            c_top = np.polyval(poly_top, grid_x_rel)
            c_bot = np.polyval(poly_bot, grid_x_rel)

            # Strict clearance guarantee: bottom must exceed top by at least 65% of mean height
            mean_h = float(np.mean(y_bots - y_tops))
            min_clearance = max(20.0, 0.65 * mean_h)
            c_bot = np.maximum(c_bot, c_top + min_clearance)

            V = np.linspace(0.0, 1.0, self.target_size, dtype=np.float32)[:, None]
            map_x = np.tile(grid_x[None, :], (self.target_size, 1)).astype(np.float32)
            map_y = ((1.0 - V) * c_top[None, :] + V * c_bot[None, :]).astype(np.float32)

            map_x = np.clip(map_x, 0, crop_w - 1)
            map_y = np.clip(map_y, 0, crop_h - 1)

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
