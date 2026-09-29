"""The detector must crop the label closest to the image centre, not the most confident one."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
from PIL import Image

from app.services.detector import DetectorService


def make_detector(boxes: list[tuple[float, float, float, float, float]]) -> DetectorService:
    detector = DetectorService.__new__(DetectorService)
    detector.confidence = 0.25
    result = SimpleNamespace(boxes=[SimpleNamespace(xyxy=[np.array(box[:4])], conf=[box[4]]) for box in boxes])
    detector.model = MagicMock()
    detector.model.predict.return_value = [result]
    return detector


def test_best_box_prefers_central_box_over_confident_one():
    detector = make_detector([(0, 0, 100, 100, 0.95), (350, 250, 450, 350, 0.40)])

    assert detector.best_box(Image.new("RGB", (800, 600))) == (350, 250, 450, 350)


def test_best_box_breaks_ties_by_confidence():
    detector = make_detector([(300, 200, 500, 400, 0.50), (350, 250, 450, 350, 0.90)])

    assert detector.best_box(Image.new("RGB", (800, 600))) == (350, 250, 450, 350)


def test_best_box_returns_none_without_detections():
    assert make_detector([]).best_box(Image.new("RGB", (800, 600))) is None
