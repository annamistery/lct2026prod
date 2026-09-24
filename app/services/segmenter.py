from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO


class SegmenterService:
    def __init__(self, model_path: Path, confidence: float = 0.25):
        if not model_path.is_file():
            raise FileNotFoundError(f"YOLO segmentation model not found: {model_path}")
        self.model = YOLO(str(model_path))
        self.confidence = confidence

    def best_polygon(self, image: Image.Image, conf: float | None = None) -> list[tuple[float, float]] | None:
        """Runs segmentation on the image and returns the largest polygon as list of (x, y) coordinates."""
        threshold = self.confidence if conf is None else conf
        results = self.model.predict(source=image, conf=threshold, verbose=False)
        polygons = []
        for res in results:
            if res.masks is not None:
                for idx, mask_xy in enumerate(res.masks.xy):
                    if len(mask_xy) >= 4:
                        conf = float(res.boxes.conf[idx]) if (res.boxes and len(res.boxes) > idx) else 0.0
                        polygons.append((mask_xy.tolist(), conf))
        if not polygons:
            return None

        def poly_area(poly: list[list[float]]) -> float:
            pts = np.array(poly, dtype=np.float32)
            return float(cv2.contourArea(pts))

        best = max(polygons, key=lambda item: poly_area(item[0]))
        return [(float(p[0]), float(p[1])) for p in best[0]]
