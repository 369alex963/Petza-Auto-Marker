"""Reading markings made by other tools.

`extract_regions` understands, in order: our own format, LabelMe, COCO
(single or multi-image), and a generic fallback that finds any list of
>= 3 points ([x, y] pairs, {x, y} dicts or flat [x1, y1, x2, y2, ...])
under a points-like key anywhere in the JSON. Normalized coordinates
(all within 0..1) are scaled to pixels when the image size is known.
"""
from __future__ import annotations

from typing import Any

LABEL_KEYS = ("label", "name", "class", "category", "type", "tag")
POINT_KEYS = ("points", "polygon", "coordinates", "coords", "vertices", "path", "contour", "outline", "segmentation")


def _num(n) -> bool:
    return isinstance(n, (int, float)) and not isinstance(n, bool)


def _as_points(v: Any) -> list[list[float]] | None:
    if not isinstance(v, list) or len(v) < 3:
        return None
    if all(isinstance(p, (list, tuple)) and len(p) >= 2 and _num(p[0]) and _num(p[1]) for p in v):
        return [[float(p[0]), float(p[1])] for p in v]
    if all(isinstance(p, dict) and _num(p.get("x")) and _num(p.get("y")) for p in v):
        return [[float(p["x"]), float(p["y"])] for p in v]
    if len(v) >= 6 and len(v) % 2 == 0 and all(_num(n) for n in v):
        return [[float(v[i]), float(v[i + 1])] for i in range(0, len(v), 2)]
    return None


def _label_of(d: dict, default: str = "wound") -> str:
    for k in LABEL_KEYS:
        if isinstance(d.get(k), str) and d[k].strip():
            return d[k].strip()
    return default


def _scale(regions, w, h):
    if regions and w and h and all(0 <= x <= 1.0 and 0 <= y <= 1.0 for _, pts in regions for x, y in pts):
        return [(lab, [[x * w, y * h] for x, y in pts]) for lab, pts in regions]
    return regions


def _ours(d: dict):
    regions = d.get("regions")
    if not (isinstance(regions, list) and all(isinstance(r, dict) and "points" in r for r in regions)):
        return None
    return [(_label_of(r), p) for r in regions if (p := _as_points(r["points"]))]


def _labelme(d: dict):
    if not isinstance(d.get("shapes"), list):
        return None
    out = []
    for s in d["shapes"]:
        pts = s.get("points") or []
        kind = s.get("shape_type") or "polygon"
        if kind == "rectangle" and len(pts) == 2:
            (x1, y1), (x2, y2) = pts
            pts = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        elif kind != "polygon":
            continue
        if p := _as_points(pts):
            out.append((_label_of(s), p))
    return out


def _coco(d: dict, image_name: str | None):
    if not isinstance(d.get("annotations"), list):
        return None
    cats = {c.get("id"): c.get("name", "wound") for c in d.get("categories", []) if isinstance(c, dict)}
    images = d.get("images") or []
    image_id = None
    if image_name and images:
        stem = image_name.rsplit(".", 1)[0].lower()
        for im in images:
            fn = str(im.get("file_name", "")).replace("\\", "/").rsplit("/", 1)[-1].lower()
            if fn == image_name.lower() or fn.rsplit(".", 1)[0] == stem:
                image_id = im.get("id")
        if image_id is None and len(images) > 1:
            return []
    out = []
    for a in d["annotations"]:
        if image_id is not None and a.get("image_id") != image_id:
            continue
        seg = a.get("segmentation")
        if isinstance(seg, list):
            for poly in seg:
                if p := _as_points(poly):
                    out.append((cats.get(a.get("category_id"), "wound"), p))
    return out


def _generic(node, found: list, label: str = "wound", depth: int = 0) -> None:
    if depth > 12:
        return
    if isinstance(node, dict):
        lab = _label_of(node, label)
        for k, v in node.items():
            pts = _as_points(v) if k.lower() in POINT_KEYS else None
            if pts:
                found.append((lab, pts))
            elif k.lower() in POINT_KEYS and isinstance(v, list) and v and all(_as_points(x) for x in v):
                found.extend((lab, _as_points(x)) for x in v)  # list of polygons
            else:
                _generic(v, found, lab, depth + 1)
    elif isinstance(node, list):
        for v in node:
            _generic(v, found, label, depth + 1)


def extract_regions(data: Any, image_name: str | None = None, width: int | None = None, height: int | None = None):
    """Return [(label, [[x, y], ...]), ...] or None if nothing recognizable."""
    if isinstance(data, dict):
        for r in (_ours(data), _labelme(data), _coco(data, image_name)):
            if r is not None:
                return _scale(r, width, height)
    found: list = []
    _generic(data, found)
    return _scale(found, width, height) if found else None
