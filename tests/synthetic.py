"""Synthetic 'wound' photos with LabelMe-style JSON, for tests and demos.

python tests/synthetic.py <folder> [count]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np


def blob_polygon(rng, cx, cy, r, n=90):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    radius = np.ones(n) * r
    for k in range(2, 6):
        radius += r * rng.uniform(0.03, 0.14) * np.sin(k * t + rng.uniform(0, 6.28))
    return np.stack([cx + radius * np.cos(t), cy + radius * np.sin(t) * rng.uniform(0.6, 1.0)], 1)


def make_image(rng, w=900, h=700):
    skin = np.array([rng.uniform(150, 230), rng.uniform(110, 180), rng.uniform(90, 150)])
    yy, xx = np.mgrid[0:h, 0:w]
    grad = (xx / w * rng.uniform(-40, 40) + yy / h * rng.uniform(-40, 40))[..., None]
    noise = cv2.GaussianBlur(rng.normal(0, 18, (h, w, 1)).astype(np.float32), (0, 0), 3)[..., None]
    img = np.clip(skin + grad + noise, 0, 255).astype(np.uint8)
    regions = []
    for _ in range(rng.integers(1, 3)):
        r = rng.uniform(50, 140)
        poly = blob_polygon(rng, rng.uniform(r + 20, w - r - 20), rng.uniform(r + 20, h - r - 20), r)
        mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask, [poly.round().astype(np.int32)], 1)
        wound = np.array([rng.uniform(140, 200), rng.uniform(30, 70), rng.uniform(40, 80)])
        tex = cv2.GaussianBlur(rng.normal(0, 25, (h, w)).astype(np.float32), (0, 0), 2)[..., None]
        soft = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 1.5)[..., None]
        img = np.clip(img * (1 - soft) + np.clip(wound + tex, 0, 255) * soft, 0, 255).astype(np.uint8)
        regions.append(poly)
    if rng.random() < 0.4:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.5, 2.0))
    return img, regions


def write_dataset(folder: Path, count: int, seed: int = 0) -> list[Path]:
    rng = np.random.default_rng(seed)
    folder.mkdir(parents=True, exist_ok=True)
    out = []
    for i in range(count):
        img, regions = make_image(rng)
        p = folder / f"wound_{i:03d}.jpg"
        cv2.imwrite(str(p), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(60, 95))])
        shapes = [{"label": "wound", "points": poly.round(1).tolist(), "shape_type": "polygon"} for poly in regions]
        (folder / f"wound_{i:03d}.json").write_text(json.dumps({"shapes": shapes, "imageWidth": img.shape[1], "imageHeight": img.shape[0]}))
        out.append(p)
    return out


if __name__ == "__main__":
    write_dataset(Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 40)
