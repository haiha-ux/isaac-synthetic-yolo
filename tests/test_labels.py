import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from yolo_labels import box_to_yolo, label_line, yolo_to_box  # noqa: E402


def test_round_trip():
    box = (100.0, 50.0, 300.0, 250.0)
    back = yolo_to_box(*box_to_yolo(*box, 640, 480), 640, 480)
    assert back == pytest.approx(box)


def test_known_values():
    assert box_to_yolo(0, 0, 640, 480, 640, 480) == (0.5, 0.5, 1.0, 1.0)
    assert label_line(3, (64, 48, 128, 96), 640, 480) == "3 0.150000 0.150000 0.100000 0.100000"


def test_clips_to_image():
    cx, cy, bw, bh = box_to_yolo(-20, 400, 100, 500, 640, 480)
    assert yolo_to_box(cx, cy, bw, bh, 640, 480) == pytest.approx((0, 400, 100, 480))


def test_empty_box_rejected():
    with pytest.raises(ValueError):
        box_to_yolo(700, 10, 800, 50, 640, 480)
    with pytest.raises(ValueError):
        box_to_yolo(10, 10, 10, 50, 640, 480)
