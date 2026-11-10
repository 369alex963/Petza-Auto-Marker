"""Demo data shipped in `demo/`: set up on the very first start, removed in one go.

The demo model learned from the demo examples only (no wound data before them),
so it shows how the app works; the user's own examples make it far more exact.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from . import config, dataset, train


def folders(root: Path | None = None) -> dict:
    root = Path(root or config.DEMO)
    return {"examples_dir": root / "Examples", "images_dir": root / "Photos", "output_dir": root / "Marked JSON"}


def _inside_demo(path: str) -> bool:
    if not path:
        return False
    demo = config.DEMO.resolve()
    p = Path(path).resolve()
    return p == demo or demo in p.parents


def active(s: config.Settings) -> bool:
    return config.DEMO.is_dir() and any(_inside_demo(getattr(s, k)) for k in ("images_dir", "examples_dir", "output_dir"))


def seed() -> bool:
    """First start only (no settings yet): point the app at the demo and install its model."""
    if config.SETTINGS_FILE.is_file() or not config.DEMO.is_dir():
        return False
    f = folders()
    s = config.Settings(**{k: str(v) for k, v in f.items()})
    dataset.import_examples(f["examples_dir"])
    model = config.DEMO / "model"
    meta_file = model / "meta.json"
    if meta_file.is_file():
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        # The ids are path hashes, so the shipped meta stores file names; map them to this install.
        ids = {n: dataset.item_id(f["examples_dir"] / n) for n in [*meta.get("holdout_ids", []), *[p["id"] for p in (meta.get("score") or {}).get("per_image", [])]]}
        meta["holdout_ids"] = [ids[n] for n in meta.get("holdout_ids", [])]
        for p in (meta.get("score") or {}).get("per_image", []):
            p["id"] = ids[p["id"]]
        meta.pop("ids_are_names", None)
        run = config.RUNS / meta["run_id"]
        run.mkdir(parents=True, exist_ok=True)
        for name in ("model.pt", "model.onnx"):
            if (model / name).is_file():
                shutil.copy2(model / name, run / name)
        (run / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        score = meta.get("score") or {}
        line = {"run_id": meta["run_id"], "created_at": meta["created_at"], "train_count": meta.get("train_count"),
                "holdout_count": len(meta["holdout_ids"]), "iou": score.get("iou"), "dice": score.get("dice"),
                "boundary_f": score.get("boundary_f"), "previous_iou": None, "promoted": True,
                "device": meta.get("device"), "minutes": None, "finetuned": False, "demo": True}
        with open(config.RUNS / "runs.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(line) + "\n")
        train.set_current(meta["run_id"])
    config.save_settings(s)
    return True


def remove() -> dict:
    """Delete the demo folder, the demo examples in the pool, every trained model
    (all of them learned from the demo examples) and the caches. Prompts, AI
    settings and the user's own folders and examples stay."""
    index = dataset.load_index()
    gone = [i for i, v in index.items() if _inside_demo(v.get("source", ""))]
    for iid in gone:
        for f in config.POOL.glob(f"{iid}.*"):
            f.unlink()
        index.pop(iid)
    if config.POOL.is_dir():
        dataset._write_json(config.POOL / "index.json", index)
    s = config.load_settings()
    for k in ("images_dir", "examples_dir", "output_dir"):
        if _inside_demo(getattr(s, k)):
            setattr(s, k, "")
    shutil.rmtree(config.RUNS, ignore_errors=True)
    shutil.rmtree(config.CACHE, ignore_errors=True)
    shutil.rmtree(config.DEMO, ignore_errors=True)
    config.save_settings(s)
    return {"pool_removed": len(gone)}
