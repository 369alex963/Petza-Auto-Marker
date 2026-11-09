import pytest
from pydantic import ValidationError

from petza.schema import Marking, Region, marking_path_for, read_marking, write_marking


def make(points=((10, 10), (50, 10), (30, 40))):
    return Marking(image="a.jpg", width=100, height=80, regions=[Region(id="r1", points=[list(p) for p in points])])


def test_roundtrip(tmp_path):
    m = make()
    p = tmp_path / "a.json"
    write_marking(m, p)
    assert read_marking(p) == m


def test_points_outside_image_rejected():
    with pytest.raises(ValidationError):
        make(((10, 10), (500, 10), (30, 40)))


def test_too_few_points_rejected():
    with pytest.raises(ValidationError):
        make(((10, 10), (50, 10)))


def test_marking_path(tmp_path):
    img = tmp_path / "x" / "w1.jpg"
    assert marking_path_for(img) == tmp_path / "x" / "w1.json"
    assert marking_path_for(img, tmp_path / "out") == tmp_path / "out" / "w1.json"
