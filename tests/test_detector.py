"""The detector must crop the label of the bottle on the vertical line through the image centre."""

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


def test_label_below_the_centre_on_the_central_bottle_beats_a_side_label_nearer_by_distance():
    # a bottle held in the hand: its label is low under the centre; a background label is diagonally closer
    held = (380, 420, 470, 580, 0.6)
    background = (250, 200, 360, 290, 0.9)
    detector = make_detector([background, held])

    assert detector.best_box(Image.new("RGB", (800, 600))) == held[:4]


def test_among_boxes_on_the_centre_line_the_vertically_nearest_wins():
    top = (350, 0, 450, 100, 0.9)
    middle = (350, 250, 450, 350, 0.5)
    detector = make_detector([top, middle])

    assert detector.best_box(Image.new("RGB", (800, 600))) == middle[:4]


def test_centre_between_two_bottles_takes_the_nearest_edge():
    left = (250, 200, 395, 400, 0.9)
    right = (410, 200, 560, 400, 0.9)
    detector = make_detector([left, right])

    assert detector.best_box(Image.new("RGB", (800, 600))) == left[:4]


def test_only_side_labels_found_means_no_box():
    # the central label was not detected: a side bottle would be a wrong wine, the caller uses the whole frame
    detector = make_detector([(0, 200, 150, 400, 0.9), (650, 200, 800, 400, 0.9)])

    assert detector.best_box(Image.new("RGB", (800, 600))) is None
