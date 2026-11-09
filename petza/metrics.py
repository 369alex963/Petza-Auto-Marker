"""Segmentation quality metrics on boolean masks."""
from __future__ import annotations

import cv2
import numpy as np


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = pred.astype(bool), gt.astype(bool)
    union = (p | g).sum()
    return 1.0 if union == 0 else float((p & g).sum() / union)


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = pred.astype(bool), gt.astype(bool)
    total = p.sum() + g.sum()
    return 1.0 if total == 0 else float(2 * (p & g).sum() / total)


def _boundary(m: np.ndarray) -> np.ndarray:
    m = m.astype(np.uint8)
    return (m - cv2.erode(m, np.ones((3, 3), np.uint8))).astype(bool)


def boundary_f(pred: np.ndarray, gt: np.ndarray, tolerance_frac: float = 0.005) -> float:
    """Boundary F-score: share of edge pixels within a tolerance of the other edge."""
    pb, gb = _boundary(pred), _boundary(gt)
    if not pb.any() and not gb.any():
        return 1.0
    if not pb.any() or not gb.any():
        return 0.0
    h, w = gt.shape
    tol = max(2, int(round(tolerance_frac * (h * h + w * w) ** 0.5)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol + 1, 2 * tol + 1))
    gd = cv2.dilate(gb.astype(np.uint8), k).astype(bool)
    pd = cv2.dilate(pb.astype(np.uint8), k).astype(bool)
    precision = (pb & gd).sum() / pb.sum()
    recall = (gb & pd).sum() / gb.sum()
    return 0.0 if precision + recall == 0 else float(2 * precision * recall / (precision + recall))


def score_pair(pred: np.ndarray, gt: np.ndarray) -> dict:
    return {"iou": iou(pred, gt), "dice": dice(pred, gt), "boundary_f": boundary_f(pred, gt)}
