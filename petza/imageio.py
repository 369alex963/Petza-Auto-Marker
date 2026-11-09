"""Image loading. Coordinates everywhere refer to the EXIF-corrected image."""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
Image.MAX_IMAGE_PIXELS = 120_000_000
warnings.simplefilter("error", Image.DecompressionBombWarning)


def open_image(path: str | Path) -> Image.Image:
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        return im.convert("RGB")


def load_image(path: str | Path) -> np.ndarray:
    """RGB uint8 array (h, w, 3), EXIF rotation applied."""
    return np.asarray(open_image(path))


def list_images(folder: str | Path, skip: Path | None = None) -> list[Path]:
    """All images under `folder` (recursive), sorted, skipping hidden folders."""
    folder = Path(folder)
    if not folder.is_dir():
        return []
    out = []
    for p in folder.rglob("*"):
        rel = p.relative_to(folder).parts
        if any(part.startswith(".") for part in rel[:-1]):
            continue
        if skip and skip in p.parents:
            continue
        if p.suffix.lower() in IMAGE_EXTS and p.is_file():
            out.append(p)
    return sorted(out, key=lambda p: str(p).lower())
