from pathlib import Path

from PIL import Image
from ultralytics import YOLO


class DetectorService:
    # A label whose nearest edge is farther than this share of the width from the centre line is a side bottle
    max_side_offset = 0.15

    def __init__(self, model_path: Path, confidence: float):
        if not model_path.is_file():
            raise FileNotFoundError(f"YOLO model not found: {model_path}")
        self.model = YOLO(str(model_path))
        self.confidence = confidence

    def boxes(self, image: Image.Image, conf: float | None = None) -> list[tuple[float, float, float, float, float]]:
        """All detected label boxes as (x1, y1, x2, y2, confidence)."""
        threshold = self.confidence if conf is None else conf
        results = self.model.predict(source=image, conf=threshold, verbose=False)
        detections = []
        for result in results:
            for box in result.boxes:
                coordinates = box.xyxy[0].tolist()
                detections.append((*coordinates, float(box.conf[0])))
        return detections

    def best_box(self, image: Image.Image, conf: float | None = None) -> tuple[float, float, float, float] | None:
        """The label of the bottle the user aims at: the one on the vertical line through the image centre.

        A bottle stands upright, so its label may be above or below the image centre (a bottle held in
        the hand, a close-up of the neck) but it crosses the central vertical line. Among such boxes the
        one vertically closest to the centre wins, then the most confident. When no box crosses the line
        (the centre falls between two bottles) the box with the nearest edge is taken — unless it is
        farther than `max_side_offset` of the width: then the central label was not detected and a side
        label would be the wrong wine, so None is returned and the caller uses the whole frame.
        """
        detections = self.boxes(image, conf)
        if not detections:
            return None
        center_x, center_y = image.width / 2, image.height / 2

        covering = [box for box in detections if box[0] <= center_x <= box[2]]
        if covering:
            # 0 when the box also covers the centre vertically, otherwise the distance to its nearer edge
            best = min(covering, key=lambda box: (max(box[1] - center_y, center_y - box[3], 0.0), -box[4]))
        else:
            best = min(detections, key=lambda box: (min(abs(box[0] - center_x), abs(box[2] - center_x)), -box[4]))
            if min(abs(best[0] - center_x), abs(best[2] - center_x)) > self.max_side_offset * image.width:
                return None
        return best[0], best[1], best[2], best[3]
