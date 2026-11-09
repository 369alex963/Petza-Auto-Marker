"""Pure geometry: letterboxing, mask <-> polygon conversion.

All coordinates are pixels with the origin at the top-left of the
EXIF-corrected image. Points are (x, y).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Letterbox:
    scale: float
    orig_h: int
    orig_w: int
    size: int


def letterbox(img: np.ndarray, size: int) -> tuple[np.ndarray, Letterbox]:
    """Resize so the long side is `size`, pad bottom/right to a square."""
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (nw, nh), interpolation=interp)
    out = np.zeros((size, size) + img.shape[2:], img.dtype)
    out[:nh, :nw] = resized
    return out, Letterbox(scale, h, w, size)


def unletterbox(arr: np.ndarray, lb: Letterbox) -> np.ndarray:
    """Map a (size, size[, C]) map back to the original (h, w[, C])."""
    nh, nw = max(1, round(lb.orig_h * lb.scale)), max(1, round(lb.orig_w * lb.scale))
    return cv2.resize(arr[:nh, :nw], (lb.orig_w, lb.orig_h), interpolation=cv2.INTER_LINEAR)


def resample_polygon(points: np.ndarray, spacing: float) -> np.ndarray:
    """Evenly resample a closed polygon so points are ~`spacing` apart."""
    pts = np.asarray(points, float)
    closed = np.vstack([pts, pts[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    perimeter = seg.sum()
    n = max(8, int(round(perimeter / spacing)))
    cum = np.concatenate([[0], np.cumsum(seg)])
    targets = np.linspace(0, perimeter, n, endpoint=False)
    idx = np.clip(np.searchsorted(cum, targets, side="right") - 1, 0, len(seg) - 1)
    t = (targets - cum[idx]) / np.maximum(seg[idx], 1e-9)
    return closed[idx] + (closed[idx + 1] - closed[idx]) * t[:, None]


def mask_to_polygons(
    mask: np.ndarray,
    min_area_px: float = 100,
    simplify_tolerance_px: float = 1.0,
    point_spacing_px: float | None = None,
    max_regions: int | None = None,
) -> list[list[list[float]]]:
    """Outer outline of each connected blob, largest first. Holes are filled."""
    m = np.ascontiguousarray(mask.astype(np.uint8))
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    items = [(cv2.contourArea(c), c) for c in contours]
    items = [it for it in items if it[0] >= min_area_px]
    items.sort(key=lambda it: it[0], reverse=True)
    if max_regions:
        items = items[:max_regions]
    h, w = m.shape
    polys = []
    for _, c in items:
        pts = cv2.approxPolyDP(c, simplify_tolerance_px, True).reshape(-1, 2).astype(float)
        if point_spacing_px:
            pts = resample_polygon(pts, point_spacing_px)
        if len(pts) < 3:
            continue
        pts[:, 0] = np.clip(pts[:, 0], 0, w)
        pts[:, 1] = np.clip(pts[:, 1], 0, h)
        polys.append(np.round(pts, 1).tolist())
    return polys


def polygons_to_mask(polys, h: int, w: int) -> np.ndarray:
    """Rasterize polygons (filled) into a bool mask of shape (h, w)."""
    m = np.zeros((h, w), np.uint8)
    for p in polys:
        arr = np.round(np.asarray(p, float)).astype(np.int32).reshape(-1, 1, 2)
        if len(arr) >= 3:
            cv2.fillPoly(m, [arr], 1)
    return m.astype(bool)


def median_point_spacing(polys) -> float | None:
    """Typical distance between consecutive points, to mimic a marking style."""
    d = []
    for p in polys:
        a = np.asarray(p, float)
        if len(a) >= 3:
            d.extend(np.linalg.norm(np.diff(np.vstack([a, a[:1]]), axis=0), axis=1).tolist())
    return float(np.median(d)) if d else None
