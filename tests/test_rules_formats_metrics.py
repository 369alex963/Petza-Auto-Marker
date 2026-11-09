import cv2
import numpy as np

from petza.formats import extract_regions
from petza.metrics import boundary_f, dice, iou
from petza.rules import load_rules, parse_rules

TRI = [[10, 10], [50, 10], [30, 40]]
TRI_F = [[10.0, 10.0], [50.0, 10.0], [30.0, 40.0]]


def test_rules_front_matter():
    r = parse_rules("---\nlabels: [wound, slough]\nmin_area_px: 400\n---\nTrace the outer edge.\n")
    assert r.labels == ["wound", "slough"] and r.min_area_px == 400
    assert r.text == "Trace the outer edge."


def test_rules_plain_text_and_missing(tmp_path):
    assert parse_rules("Just text").text == "Just text"
    assert load_rules(tmp_path / "nope.md").labels == []


def test_ours():
    assert extract_regions({"regions": [{"label": "w", "points": TRI}]}) == [("w", TRI_F)]


def test_labelme():
    d = {"shapes": [{"label": "wound", "points": TRI, "shape_type": "polygon"},
                    {"label": "c", "points": [[1, 1], [2, 2]], "shape_type": "circle"}]}
    assert extract_regions(d) == [("wound", TRI_F)]


def test_coco_multi_image():
    d = {
        "images": [{"id": 1, "file_name": "a.jpg"}, {"id": 2, "file_name": "b.jpg"}],
        "categories": [{"id": 7, "name": "ulcer"}],
        "annotations": [
            {"image_id": 1, "category_id": 7, "segmentation": [[10, 10, 50, 10, 30, 40]]},
            {"image_id": 2, "category_id": 7, "segmentation": [[1, 1, 5, 1, 3, 4]]},
        ],
    }
    assert extract_regions(d, "b.jpg") == [("ulcer", [[1.0, 1.0], [5.0, 1.0], [3.0, 4.0]])]


def test_generic_dict_points_and_normalized():
    d = {"case": {"marks": [{"type": "wound", "outline": [{"x": 0.1, "y": 0.1}, {"x": 0.5, "y": 0.1}, {"x": 0.3, "y": 0.4}]}]}}
    assert extract_regions(d, width=100, height=100) == [("wound", TRI_F)]


def test_unknown_is_none():
    assert extract_regions({"foo": 1, "bar": "x"}) is None


def test_metrics():
    a = np.zeros((200, 200), np.uint8)
    cv2.circle(a, (100, 100), 50, 1, -1)
    a = a.astype(bool)
    b = np.zeros_like(a)
    b[:, 1:] = a[:, :-1]
    assert iou(a, a) == 1 and dice(a, a) == 1
    assert iou(a, ~a) == 0
    assert boundary_f(b, a) > 0.9
