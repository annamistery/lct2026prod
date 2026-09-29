from pathlib import Path

from PIL import Image
from ultralytics import YOLO


class DetectorService:
    def __init__(self, model_path: Path, confidence: float):
        if not model_path.is_file():
            raise FileNotFoundError(f"YOLO model not found: {model_path}")
        self.model = YOLO(str(model_path))
        self.confidence = confidence

    def best_box(self, image: Image.Image, conf: float | None = None) -> tuple[float, float, float, float] | None:
        threshold = self.confidence if conf is None else conf
        results = self.model.predict(source=image, conf=threshold, verbose=False)
        detections = []
        for result in results:
            for box in result.boxes:
                coordinates = box.xyxy[0].tolist()
                detections.append((*coordinates, float(box.conf[0])))
        if not detections:
            return None
        # The label the user aims at is the one closest to the image centre, not the most confident one.
        center_x, center_y = image.width / 2, image.height / 2

        def centrality(item: tuple[float, float, float, float, float]) -> tuple[float, float]:
            box_x = (item[0] + item[2]) / 2
            box_y = (item[1] + item[3]) / 2
            return (box_x - center_x) ** 2 + (box_y - center_y) ** 2, -item[4]

        best = min(detections, key=centrality)
        return best[0], best[1], best[2], best[3]
