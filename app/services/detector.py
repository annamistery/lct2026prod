from pathlib import Path

from PIL import Image
from ultralytics import YOLO


class DetectorService:
    def __init__(self, model_path: Path, confidence: float):
        if not model_path.is_file():
            raise FileNotFoundError(f"YOLO model not found: {model_path}")
        self.model = YOLO(str(model_path))
        self.confidence = confidence

    def best_box(self, image: Image.Image) -> tuple[float, float, float, float] | None:
        results = self.model.predict(source=image, conf=self.confidence, verbose=False)
        detections = []
        for result in results:
            for box in result.boxes:
                coordinates = box.xyxy[0].tolist()
                detections.append((*coordinates, float(box.conf[0])))
        if not detections:
            return None
        best = max(detections, key=lambda item: item[4])
        return best[0], best[1], best[2], best[3]
