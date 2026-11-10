"""The marking JSON written next to each image."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Region(BaseModel):
    id: str
    label: str = "wound"
    points: list[list[float]] = Field(min_length=3)
    confidence: float | None = None
    source: Literal["auto", "manual"] = "auto"

    @model_validator(mode="after")
    def _pairs(self):
        if any(len(p) != 2 for p in self.points):
            raise ValueError("each point must be [x, y]")
        return self


class Marking(BaseModel):
    image: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    regions: list[Region] = []
    model_version: str | None = None
    created_at: str = Field(default_factory=now_iso)
    status: Literal["auto", "reviewed"] = "auto"
    approved_by: Literal["person", "auto"] | None = None  # "auto": approved in one go because the app was sure
    confidence: float | None = None
    review: dict | None = None   # older AI second opinion {ok, problems, summary}
    ai: dict | None = None       # what the online AI chose {via, runs, choices, missed, notes, ...}
    quality: str | None = None   # photo quality: good | dark | low_contrast | blurry

    @model_validator(mode="after")
    def _inside(self):
        for r in self.regions:
            for x, y in r.points:
                if not (0 <= x <= self.width and 0 <= y <= self.height):
                    raise ValueError(f"point ({x}, {y}) of {r.id} is outside the {self.width}x{self.height} image")
        return self


def marking_path_for(image_path: Path, output_dir: Path | None = None) -> Path:
    image_path = Path(image_path)
    folder = Path(output_dir) if output_dir else image_path.parent
    return folder / f"{image_path.stem}.json"


def read_marking(path: Path) -> Marking:
    return Marking.model_validate_json(Path(path).read_text(encoding="utf-8"))


def write_marking(marking: Marking, path: Path) -> None:
    """Atomic write so a crash never leaves a half-written JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(marking.model_dump(), indent=2), encoding="utf-8")
    os.replace(tmp, path)
