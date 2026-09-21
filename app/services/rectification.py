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
        """Extracts 4 boundary corner anchors from the full segmentation polygon."""
        pts = np.array(polygon, dtype=np.float32)
        if len(pts) < 4:
            return None

        # Ensure clockwise orientation
        sa = 0.5 * np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - pts[:, 1] * np.roll(pts[:, 0], -1))
        if sa < 0:
            pts = pts[::-1]

        # Natural image orientation:
        # In the crop image, the bottle is already roughly upright (Y is vertical, X is horizontal).
        # Top-Left: min(x + y)
        # Top-Right: max(x - y)
        # Bottom-Right: max(x + y)
        # Bottom-Left: min(x - y)
        idx_tl = int(np.argmin(pts[:, 0] + pts[:, 1]))
        idx_tr = int(np.argmax(pts[:, 0] - pts[:, 1]))
        idx_br = int(np.argmax(pts[:, 0] + pts[:, 1]))
        idx_bl = int(np.argmin(pts[:, 0] - pts[:, 1]))

        quad = np.array([pts[idx_tl], pts[idx_tr], pts[idx_br], pts[idx_bl]], dtype=np.float32)
        quad = self.order_points(quad)

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
        """Unrolls the entire segmentation boundary contour into a canonical square matrix (target_size x target_size).

        Preserves 100% of the label surface without cutting edges, straightening top/bottom curves
        and compensating for cylindrical perspective via transfinite interpolation (Coons patch).
        """
        try:
            crop_np = np.array(crop_image.convert("RGB"))
            crop_h, crop_w = crop_np.shape[:2]

            pts = np.array(polygon, dtype=np.float32)
            N = len(pts)
            if N < 4:
                return crop_image.convert("RGB").resize((self.target_size, self.target_size), Image.Resampling.LANCZOS)

            # Ensure Clockwise orientation
            sa = 0.5 * np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - pts[:, 1] * np.roll(pts[:, 0], -1))
            if sa < 0:
                pts = pts[::-1]

            # Natural image orientation:
            # In the crop image, the bottle is upright (Y is vertical downwards, X is horizontal).
            # Top-Left: min(x + y)
            # Top-Right: max(x - y)
            # Bottom-Right: max(x + y)
            # Bottom-Left: min(x - y)
            idx_tl = int(np.argmin(pts[:, 0] + pts[:, 1]))
            idx_tr = int(np.argmax(pts[:, 0] - pts[:, 1]))
            idx_br = int(np.argmax(pts[:, 0] + pts[:, 1]))
            idx_bl = int(np.argmin(pts[:, 0] - pts[:, 1]))

            def get_arc(i_from: int, i_to: int, step_dir: int = 1) -> np.ndarray:
                steps = ((i_to - i_from) * step_dir) % N
                return np.array([pts[(i_from + step_dir * s) % N] for s in range(steps + 1)], dtype=np.float32)

            c_top_raw = get_arc(idx_tl, idx_tr, 1)      # TL -> TR along top boundary
            c_right_raw = get_arc(idx_tr, idx_br, 1)    # TR -> BR along right flank
            c_bot_raw = get_arc(idx_bl, idx_br, -1)    # BL -> BR along bottom boundary
            c_left_raw = get_arc(idx_tl, idx_bl, -1)   # TL -> BL along left flank

            def resample_curve(curve_pts: np.ndarray, num_samples: int, cylindrical: bool = False) -> np.ndarray:
                dists = np.linalg.norm(np.diff(curve_pts, axis=0), axis=1)
                cum_dist = np.insert(np.cumsum(dists), 0, 0.0)
                total_len = cum_dist[-1]
                if total_len < 1e-5:
                    return np.repeat(curve_pts[0:1], num_samples, axis=0)

                t_norm = cum_dist / total_len
                cols = np.linspace(0.0, 1.0, num_samples, dtype=np.float32)

                if cylindrical:
                    theta_0 = 0.82  # ~47 degrees cylindrical view angle
                    s_norm = 2.0 * cols - 1.0
                    theta = s_norm * theta_0
                    factor = np.sin(theta) / np.sin(theta_0)
                    u = np.clip((factor + 1.0) * 0.5, 0.0, 1.0)
                else:
                    u = cols

                x_s = np.interp(u, t_norm, curve_pts[:, 0])
                y_s = np.interp(u, t_norm, curve_pts[:, 1])
                return np.column_stack([x_s, y_s])

            W, H = self.target_size, self.target_size
            c_top = resample_curve(c_top_raw, W, cylindrical=True)
            c_bot = resample_curve(c_bot_raw, W, cylindrical=True)
            c_left = resample_curve(c_left_raw, H, cylindrical=False)
            c_right = resample_curve(c_right_raw, H, cylindrical=False)

            p_tl = c_top[0]
            p_tr = c_top[-1]
            p_bl = c_bot[0]
            p_br = c_bot[-1]

            u = np.linspace(0.0, 1.0, W, dtype=np.float32)[None, :]
            v = np.linspace(0.0, 1.0, H, dtype=np.float32)[:, None]

            # Coons Patch Transfinite Interpolation
            blend_tb = (1.0 - v) * c_top[None, :, :] + v * c_bot[None, :, :]
            blend_lr = (1.0 - u[:, :, None]) * c_left[:, None, :] + u[:, :, None] * c_right[:, None, :]
            blend_corners = (
                (1.0 - u[:, :, None]) * (1.0 - v) * p_tl +
                u[:, :, None] * (1.0 - v) * p_tr +
                (1.0 - u[:, :, None]) * v * p_bl +
                u[:, :, None] * v * p_br
            )

            map_grid = blend_tb + blend_lr - blend_corners
            map_x = np.clip(map_grid[:, :, 0].astype(np.float32), 0, crop_w - 1)
            map_y = np.clip(map_grid[:, :, 1].astype(np.float32), 0, crop_h - 1)

            unrolled = cv2.remap(
                crop_np,
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
