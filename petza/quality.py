"""Photo quality: is the photo dark, flat or blurry?

The result is saved in the JSON and shown on the photo. Only for a dark or flat
photo, the copy sent to the online AI gets a gentle brightness, white balance
and contrast correction, so the wound colours read correctly. The original
file and what the local model sees are never changed.
"""
from __future__ import annotations

import cv2
import numpy as np

DARK = 60           # mean brightness (0..255) below this is dark
FLAT = 20           # brightness spread below this is low contrast (very flat only; most photos are fine)
BLURRY = 20.0       # sharpness (Laplacian variance at 1024 px) below this is blurry (clearly soft only)


def measure(img: np.ndarray) -> dict:
    h, w = img.shape[:2]
    scale = min(1.0, 1024 / max(h, w))
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else img
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    return {
        "brightness": round(float(gray.mean()), 1),
        "contrast": round(float(gray.std()), 1),
        "sharpness": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 1),
    }


def check(img: np.ndarray) -> str:
    m = measure(img)
    if m["brightness"] < DARK:
        return "dark"
    if m["contrast"] < FLAT:
        return "low_contrast"
    if m["sharpness"] < BLURRY:
        return "blurry"
    return "good"


def needs_correction(status: str) -> bool:
    return status in ("dark", "low_contrast")


def correct(img: np.ndarray) -> np.ndarray:
    """Even out lightness and contrast; fix a colour cast only when the photo has
    white or gray things to judge it by (a sheet, a ruler, a dressing). Skin is
    never pushed toward gray, so wound colours stay true. RGB in, RGB out."""
    out = img
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    neutral = (hsv[..., 1] < 40) & (hsv[..., 2] > 120)
    if neutral.mean() > 0.02:
        means = img[neutral].astype(np.float32).mean(0)
        gains = np.clip(means.mean() / np.maximum(means, 1), 0.8, 1.25)
        out = np.clip(img.astype(np.float32) * gains, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(out, cv2.COLOR_RGB2LAB)
    if lab[..., 0].mean() < DARK * 1.1:  # dark: lift the shadows gently first
        lut = np.clip(((np.arange(256) / 255.0) ** 0.7) * 255, 0, 255).astype(np.uint8)
        lab[..., 0] = cv2.LUT(lab[..., 0], lut)
    lab[..., 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
