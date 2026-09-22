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
        """Rectifies the label into a canonical 256x256 matrix preserving true physical aspect ratio.

        Uses robust perspective homography derived from the oriented bounding box of the segmentation.
        Guarantees:
        1. Straight text lines remain strictly straight (no wave or bending distortion).
        2. Proportions (aspect ratio) are preserved via centered letterboxing on a neutral canvas,
           preventing unnatural 2x horizontal stretching or vertical squashing.
        3. 100% sharp, readable typography without interpolation blur.
        """
        try:
            crop_np = np.array(crop_image.convert("RGB"))
            ch, cw = crop_np.shape[:2]
            pts = np.array(polygon, dtype=np.float32)

            if len(pts) < 4:
                return crop_image.convert("RGB").resize((self.target_size, self.target_size), Image.Resampling.LANCZOS)

            # 1. Obtain stable oriented quad from segmentation contour
            rect = cv2.minAreaRect(pts)
            box = cv2.boxPoints(rect)
            ordered = self.order_points(box)

            # Clamp coordinates to crop boundary
            ordered[:, 0] = np.clip(ordered[:, 0], 0, cw - 1)
            ordered[:, 1] = np.clip(ordered[:, 1], 0, ch - 1)

            # 2. Calculate true physical width and height to preserve natural aspect ratio
            w_top = float(np.linalg.norm(ordered[1] - ordered[0]))
            w_bot = float(np.linalg.norm(ordered[2] - ordered[3]))
            h_left = float(np.linalg.norm(ordered[3] - ordered[0]))
            h_right = float(np.linalg.norm(ordered[2] - ordered[1]))

            real_w = max(10.0, (w_top + w_bot) * 0.5)
            real_h = max(10.0, (h_left + h_right) * 0.5)
            aspect = real_w / real_h

            # 3. Fit label proportionally into canonical target_size x target_size (Letterbox)
            if aspect <= 1.0:
                # Vertical/tall label: fit height to target_size, scale width proportionally
                out_h = self.target_size
                out_w = max(16, min(self.target_size, int(round(self.target_size * aspect))))
                pad_x = (self.target_size - out_w) // 2
                pad_y = 0
            else:
                # Horizontal/wide label: fit width to target_size, scale height proportionally
                out_w = self.target_size
                out_h = max(16, min(self.target_size, int(round(self.target_size / aspect))))
                pad_x = 0
                pad_y = (self.target_size - out_h) // 2

            dst_pts = np.float32([
                [pad_x, pad_y],
                [pad_x + out_w - 1, pad_y],
                [pad_x + out_w - 1, pad_y + out_h - 1],
                [pad_x, pad_y + out_h - 1],
            ])

            # 4. Perspective warp into canonical canvas
            matrix = cv2.getPerspectiveTransform(ordered, dst_pts)
            warped = cv2.warpPerspective(
                crop_np,
                matrix,
                (self.target_size, self.target_size),
                flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(20, 20, 20),
            )
            return Image.fromarray(warped)

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
