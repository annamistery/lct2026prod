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
        angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
        order = np.argsort(angles)
        pts_sorted = pts[order]

        target_tl = np.array([-1.0, -1.0], dtype=np.float32)
        vecs = pts_sorted - center
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms < 1e-6] = 1.0
        vecs_norm = vecs / norms
        tl_idx = int(np.argmax(np.dot(vecs_norm, target_tl)))

        return np.roll(pts_sorted, -tl_idx, axis=0)

    def unroll_oriented_cylinder(
        self,
        crop_image: Image.Image,
        polygon: list[tuple[float, float]],
    ) -> tuple[Image.Image | None, np.ndarray | None]:
        """Truly unrolls the cylindrical label into a flat canonical 256x256 matrix.

        Aligns the bottle's vertical axis using minAreaRect to prevent false perspective skew.
        Extracts column-wise top and bottom cylinder arcs from the raster mask.
        Straightens top and bottom curves into horizontal lines and compensates for
        lateral cylindrical compression.
        """
        try:
            crop_np = np.array(crop_image.convert("RGB"))
            ch, cw = crop_np.shape[:2]
            pts = np.array(polygon, dtype=np.float32)

            if len(pts) < 4:
                return None, None

            # 1. Orientation via minAreaRect (bottle roll/tilt angle)
            rect = cv2.minAreaRect(pts)
            center, (bw, bh), angle = rect

            if bw < bh:
                tilt = angle
            else:
                tilt = angle + 90.0 if angle < 0 else angle - 90.0

            if abs(tilt) > 0.5 and abs(tilt) < 45.0:
                M_rot = cv2.getRotationMatrix2D(center, tilt, 1.0)
                crop_rot = cv2.warpAffine(
                    crop_np, M_rot, (cw, ch), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE
                )
                ones = np.ones((len(pts), 1), dtype=np.float32)
                pts_rot = np.dot(M_rot, np.hstack([pts, ones]).T).T
            else:
                crop_rot = crop_np
                pts_rot = pts
                tilt = 0.0

            # 2. Raster mask in upright coordinates with spike & notch suppression
            mask = np.zeros((ch, cw), dtype=np.uint8)
            cv2.fillPoly(mask, [np.int32(pts_rot)], 255)

            k_w = max(5, int(cw * 0.015) | 1)
            k_h = max(5, int(ch * 0.015) | 1)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_w, k_h))
            mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel)

            # 3. Detect solid horizontal label column span
            col_sums = np.sum(mask_clean > 0, axis=0)
            max_c = np.max(col_sums)
            valid_cols = np.where(col_sums >= max(20, int(0.15 * max_c)))[0]
            if len(valid_cols) < 20:
                return None, None

            xmin = int(valid_cols[0])
            xmax = int(valid_cols[-1])
            w_label = xmax - xmin
            x_mid = (xmin + xmax) * 0.5

            # 4. Extract continuous top and bottom boundary curves per column
            xs_prof, y_tops, y_bots = [], [], []
            for x in range(xmin, xmax + 1):
                rows = np.where(mask_clean[:, x] > 0)[0]
                if len(rows) >= 10:
                    xs_prof.append(x)
                    y_tops.append(rows[0])
                    y_bots.append(rows[-1])

            if len(xs_prof) < 15:
                return None, None

            xs_prof = np.array(xs_prof, dtype=np.float32)
            y_tops = np.array(y_tops, dtype=np.float32)
            y_bots = np.array(y_bots, dtype=np.float32)

            X_rel = xs_prof - x_mid
            poly_top = np.polyfit(X_rel, y_tops, 2)
            poly_bot = np.polyfit(X_rel, y_bots, 2)

            # 5. Build Remap Grid:
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

            # Reconstruct true 4 corner anchors in unrotated crop coordinates
            box = cv2.boxPoints(rect)
            quad = self.order_points(box)

            return Image.fromarray(unrolled), quad

        except Exception:
            return None, None

    def rectify_crop(
        self,
        crop_image: Image.Image,
        conf_seg: float | None = None,
    ) -> RectificationResult:
        """Rectify an already-cropped label image."""
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
        rectified: Image.Image | None = None

        if polygon is not None:
            rectified, quad = self.unroll_oriented_cylinder(crop_image, polygon)
            if rectified is not None:
                method = "cylinder_unroll"
                frame_method = "minarearect_raster"

        if rectified is None:
            rectified = crop_image.convert("RGB").resize(
                (self.target_size, self.target_size), Image.Resampling.LANCZOS
            )

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
        """Full pipeline: BBox detect -> padded crop -> segmentation -> cylinder unroll."""
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
        rectified: Image.Image | None = None

        if polygon is not None:
            rectified, quad = self.unroll_oriented_cylinder(crop_img, polygon)
            if rectified is not None:
                method = "cylinder_unroll"
                frame_method = "minarearect_raster"

        if rectified is None:
            rectified = crop_img.convert("RGB").resize(
                (self.target_size, self.target_size), Image.Resampling.LANCZOS
            )

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
            frame_method=frame_method,
            aspect_ratio=aspect_ratio,
            aspect_clamped=aspect_clamped,
            bbox_ms=float(bbox_ms),
            seg_ms=float(seg_ms),
            warp_ms=float(warp_ms),
            total_ms=float(total_ms),
        )
