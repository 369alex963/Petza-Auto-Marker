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
TOOLS = ROOT / "tools"          # official AI command-line apps the app downloaded itself
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
    reviewer_provider: str = "openai"  # openai (default) | claude | groq
    anthropic_api_key: str = ""
    reviewer_model: str = "claude-haiku-5-5"  # Claude: fast and affordable, hundreds of photos per account
    claude_effort: str = "low"
    openai_api_key: str = ""
    openai_model: str = "gpt-6-luna"     # fast and affordable, reads images: hundreds of photos per account
    openai_effort: str = "low"            # reasoning effort: low is enough to pick from ready outlines
    groq_api_key: str = ""
    groq_model: str = "qwen/qwen3.8-27b"  # Groq's vision model (free plan)
    ai_runs: int = 1                      # 1, 2 or 3 questions per photo (majority vote)
    ai_scope: str = "unsure"              # "unsure": ask only when the photo is not clear-cut; "all": every photo
    ai_accounts_off: list[str] = []       # providers whose account this app must not use ("Disconnect")
    settings_version: int = 3
    ai_edges: bool = False                # let the AI move the edge (tight/wide); measured less exact, so off


def load_settings() -> Settings:
    if SETTINGS_FILE.is_file():
        try:
            s = Settings.model_validate_json(SETTINGS_FILE.read_text(encoding="utf-8"))
            raw = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
            version = raw.get("settings_version", 1)
            if version < 2 and s.openai_model == "gpt-6-astra":
                s.openai_model = "gpt-6-luna"  # the old default spent the account's quota in a few dozen photos
            if version < 3 and s.reviewer_model == "claude-opus-5-5":
                s.reviewer_model = "claude-haiku-5-5"  # same for Claude: the old default was the most expensive model
            return s
        except ValueError:
            pass
    return Settings()


def save_settings(s: Settings) -> None:
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = SETTINGS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(s.model_dump(), indent=2), encoding="utf-8")
    os.replace(tmp, SETTINGS_FILE)
