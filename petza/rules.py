"""Marking rules file: free text for people (and the AI reviewer), plus an
optional YAML front-matter block with the measurable rules.

    ---
    labels: [wound]
    min_area_px: 400
    simplify_tolerance_px: 1.5
    point_spacing_px: 12
    max_regions: 3
    review_below_confidence: 0.75
    ---
    Free text rules...
"""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel


class Rules(BaseModel):
    labels: list[str] = []
    min_area_px: float | None = None            # None -> 0.02% of image area
    simplify_tolerance_px: float | None = None  # None -> 0.05% of image diagonal (min 1px)
    point_spacing_px: float | None = None       # None -> learned from examples
    max_regions: int | None = None
    review_below_confidence: float = 0.75
    text: str = ""

    def min_area_for(self, h: int, w: int) -> float:
        return self.min_area_px if self.min_area_px is not None else max(30.0, 0.0002 * h * w)

    def tolerance_for(self, h: int, w: int) -> float:
        if self.simplify_tolerance_px is not None:
            return self.simplify_tolerance_px
        return max(1.0, 0.0005 * (h * h + w * w) ** 0.5)


def parse_rules(raw: str) -> Rules:
    data: dict = {}
    body = raw.lstrip("﻿")
    if body.startswith("---"):
        parts = body.split("---", 2)
        if len(parts) == 3:
            loaded = yaml.safe_load(parts[1])
            data = loaded if isinstance(loaded, dict) else {}
            body = parts[2]
    known = {k: v for k, v in data.items() if k in Rules.model_fields and k != "text"}
    return Rules(**known, text=body.strip())


def load_rules(path: str | Path | None) -> Rules:
    if not path or not Path(path).is_file():
        return Rules()
    return parse_rules(Path(path).read_text(encoding="utf-8", errors="replace"))
