"""Paths and user settings. Everything lives inside this project folder."""
from __future__ import annotations

import json
import os
from pathlib import Path

from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("PETZA_DATA", ROOT / "data"))
MODELS = Path(os.environ.get("PETZA_MODELS", ROOT / "models"))  # bundled pretrained weights
POOL = DATA / "pool"            # training examples (images + our JSON)
RUNS = DATA / "models"          # trained models, one folder per run
CACHE = DATA / "cache"
SETTINGS_FILE = DATA / "settings.json"

# Keep any library cache inside the project, never in the user profile.
os.environ.setdefault("TORCH_HOME", str(MODELS / "torch"))
os.environ.setdefault("HF_HOME", str(MODELS / "hf"))


class Settings(BaseModel):
    images_dir: str = ""
    rules_path: str = ""
    examples_dir: str = ""
    output_dir: str = ""               # empty -> JSON next to each image
    reviewer_enabled: bool = False
    reviewer_provider: str = "claude"  # claude | openai
    anthropic_api_key: str = ""
    reviewer_model: str = "claude-opus-5-5"
    openai_api_key: str = ""
    openai_model: str = "gpt-6-astra"


def load_settings() -> Settings:
    if SETTINGS_FILE.is_file():
        try:
            return Settings.model_validate_json(SETTINGS_FILE.read_text(encoding="utf-8"))
        except ValueError:
            pass
    return Settings()


def save_settings(s: Settings) -> None:
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = SETTINGS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(s.model_dump(), indent=2), encoding="utf-8")
    os.replace(tmp, SETTINGS_FILE)
