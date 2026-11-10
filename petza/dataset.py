"""The training pool: imported examples plus the reviewed corrections.

Layout (inside data/pool):
    <id>.<ext>   copy of the original image file
    <id>.json    its marking in our format (coordinates in EXIF-corrected pixels)
    index.json   {id: {"source": path, "kind": "example"|"reviewed"|"ai", "added": iso}}
"ai" items are outlines the online AI picked; they are trained on but never
used as held-out test photos, so scores always compare against people.
data/labels.json keeps the ordered class list; class index = position + 1.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np

from . import config
from .formats import extract_regions
from .geometry import median_point_spacing, polygons_to_mask
from .imageio import list_images, open_image
from .schema import Marking, Region, now_iso, read_marking, write_marking

HOLDOUT_PERCENT = 15


def item_id(source: Path) -> str:
    return hashlib.sha1(str(Path(source).resolve()).lower().encode()).hexdigest()[:12]


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def load_index() -> dict:
    return _read_json(config.POOL / "index.json")


def labels() -> list[str]:
    return _read_json(config.DATA / "labels.json").get("labels", []) or ["wound"]


def _add_labels(new: list[str]) -> None:
    current = _read_json(config.DATA / "labels.json").get("labels", [])
    for lab in new:
        if lab not in current:
            current.append(lab)
    _write_json(config.DATA / "labels.json", {"labels": current})


def add_to_pool(image_path: Path, marking: Marking, kind: str) -> str:
    """Copy image + marking into the pool (replacing an older copy)."""
    image_path = Path(image_path)
    iid = item_id(image_path)
    config.POOL.mkdir(parents=True, exist_ok=True)
    for old in config.POOL.glob(f"{iid}.*"):
        if old.suffix != ".json":
            old.unlink()
    shutil.copy2(image_path, config.POOL / f"{iid}{image_path.suffix.lower()}")
    write_marking(marking, config.POOL / f"{iid}.json")
    index = load_index()
    index[iid] = {"source": str(image_path), "kind": kind, "added": now_iso()}
    _write_json(config.POOL / "index.json", index)
    _add_labels([r.label for r in marking.regions])
    return iid


def _find_json_for(image: Path, folder_jsons: list[Path]) -> list[Path]:
    """Per-image JSON first (same stem, or name.ext.json), then folder-level files."""
    own = [image.with_suffix(".json"), image.with_name(image.name + ".json")]
    return [p for p in own if p.is_file()] + folder_jsons


def import_examples(folder: str | Path, progress=lambda f, m: None, should_stop=lambda: False) -> dict:
    folder = Path(folder)
    images = list_images(folder)
    folder_jsons = [p for p in folder.rglob("*.json") if not any(p.with_suffix(e).exists() for e in (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"))]
    cache: dict[Path, object] = {}
    result = {"imported": 0, "no_marking": [], "bad": [], "total": len(images)}
    for i, img in enumerate(images):
        if should_stop():
            break
        progress(i / max(1, len(images)), f"Importing {img.name}")
        try:
            with open_image(img) as im:
                w, h = im.size
            regions = None
            for jp in _find_json_for(img, folder_jsons):
                if jp not in cache:
                    try:
                        cache[jp] = json.loads(jp.read_text(encoding="utf-8"))
                    except ValueError:
                        cache[jp] = None
                if cache[jp] is not None:
                    regions = extract_regions(cache[jp], img.name, w, h)
                    if regions:
                        break
            if not regions:
                result["no_marking"].append(img.name)
                continue
            clean = []
            for k, (lab, pts) in enumerate(regions):
                pts = [[min(max(x, 0.0), w), min(max(y, 0.0), h)] for x, y in pts]
                clean.append(Region(id=f"r{k + 1}", label=lab, points=pts, source="manual"))
            add_to_pool(img, Marking(image=img.name, width=w, height=h, regions=clean, status="reviewed"), "example")
            result["imported"] += 1
        except Exception as e:  # one bad file must not stop the import
            result["bad"].append(f"{img.name}: {e}")
    progress(1.0, "Import finished")
    return result


def pool_items() -> list[tuple[str, Path, Path]]:
    """[(id, image_path, json_path)] for every complete pool item."""
    out = []
    for jp in sorted(config.POOL.glob("*.json")):
        if jp.name == "index.json":
            continue
        imgs = [p for p in config.POOL.glob(f"{jp.stem}.*") if p.suffix != ".json"]
        if imgs:
            out.append((jp.stem, imgs[0], jp))
    return out


def is_holdout(iid: str) -> bool:
    return int(hashlib.sha1(("holdout:" + iid).encode()).hexdigest(), 16) % 100 < HOLDOUT_PERCENT


def split(ids: list[str]) -> tuple[list[str], list[str]]:
    """Stable train/holdout split by hash; at least one holdout once there are 5+ items.
    AI-picked items are always trained on, never held out."""
    index = load_index()
    human = [i for i in ids if index.get(i, {}).get("kind") != "ai"]
    hold = [i for i in human if is_holdout(i)]
    if not hold and len(human) >= 5:
        hold = [min(human, key=lambda i: hashlib.sha1(("holdout:" + i).encode()).hexdigest())]
    train = [i for i in ids if i not in hold]
    return train, hold


def class_mask(marking: Marking, classes: list[str], h: int, w: int, scale: float = 1.0) -> np.ndarray:
    """uint8 mask: 0 background, k = classes[k-1]. Later regions draw on top."""
    m = np.zeros((h, w), np.uint8)
    for r in marking.regions:
        k = classes.index(r.label) + 1 if r.label in classes else 1
        pts = [[x * scale, y * scale] for x, y in r.points]
        m[polygons_to_mask([pts], h, w)] = k
    return m


def spacing_fraction() -> float | None:
    """Median point spacing in the pool, as a fraction of the image diagonal."""
    fracs = []
    for _, _, jp in pool_items():
        mk = read_marking(jp)
        s = median_point_spacing([r.points for r in mk.regions])
        if s:
            fracs.append(s / (mk.width**2 + mk.height**2) ** 0.5)
    return float(np.median(fracs)) if fracs else None


def pool_stats() -> dict:
    items = pool_items()
    index = load_index()
    ids = [i for i, _, _ in items]
    train, hold = split(ids)
    return {
        "total": len(items),
        "examples": sum(1 for i in ids if index.get(i, {}).get("kind") == "example"),
        "reviewed": sum(1 for i in ids if index.get(i, {}).get("kind") == "reviewed"),
        "ai": sum(1 for i in ids if index.get(i, {}).get("kind") == "ai"),
        "train": len(train),
        "holdout": len(hold),
        "labels": labels(),
    }
