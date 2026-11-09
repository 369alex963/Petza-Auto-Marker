import cv2
import numpy as np

from petza.geometry import letterbox, mask_to_polygons, polygons_to_mask, resample_polygon, unletterbox


def circle_mask(h=400, w=600, c=(300, 200), r=90):
    m = np.zeros((h, w), np.uint8)
    cv2.circle(m, c, r, 1, -1)
    return m.astype(bool)


def test_letterbox_roundtrip_keeps_shape_and_content():
    img = np.random.randint(0, 255, (300, 500, 3), np.uint8)
    boxed, lb = letterbox(img, 256)
    assert boxed.shape == (256, 256, 3)
    back = unletterbox(boxed[..., 0].astype(np.float32), lb)
    assert back.shape == (300, 500)


def test_circle_becomes_one_accurate_polygon():
    m = circle_mask()
    polys = mask_to_polygons(m, min_area_px=50, simplify_tolerance_px=1.0)
    assert len(polys) == 1
    redrawn = polygons_to_mask([polys[0]], *m.shape)
    iou = (redrawn & m).sum() / (redrawn | m).sum()
    assert iou > 0.97


def test_two_blobs_two_polygons_and_small_dropped():
    m = circle_mask(c=(120, 200), r=60) | circle_mask(c=(450, 200), r=60) | circle_mask(c=(300, 30), r=3)
    polys = mask_to_polygons(m, min_area_px=200, simplify_tolerance_px=1.0)
    assert len(polys) == 2


def test_holes_are_filled():
    m = circle_mask(r=100)
    m &= ~circle_mask(r=30)
    polys = mask_to_polygons(m, min_area_px=50, simplify_tolerance_px=1.0)
    redrawn = polygons_to_mask(polys, *m.shape)
    assert redrawn[200, 300]  # hole centre is inside the outer outline


def test_resample_spacing():
    square = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], float)
    pts = resample_polygon(square, 10)
    d = np.linalg.norm(np.diff(np.vstack([pts, pts[:1]]), axis=0), axis=1)
    assert len(pts) == 40
    assert np.all((d > 5) & (d < 15))


def test_max_regions_keeps_largest():
    m = circle_mask(c=(120, 200), r=80) | circle_mask(c=(450, 200), r=40)
    polys = mask_to_polygons(m, min_area_px=50, simplify_tolerance_px=1.0, max_regions=1)
    assert len(polys) == 1
    assert np.asarray(polys[0])[:, 0].mean() < 300
