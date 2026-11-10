"""Mark every image in a folder. Reviewed markings are never overwritten."""
from __future__ import annotations

from pathlib import Path

from . import dataset
from .imageio import list_images, load_image
from .infer import Predictor
from .refine import Result, mark
from .rules import Rules
from .schema import Marking, marking_path_for, read_marking, write_marking


def existing_status(json_path: Path) -> str | None:
    try:
        return read_marking(json_path).status
    except Exception:
        return None  # missing or not ours: safe to write


def to_marking(img_path: Path, img, r: Result, run_id: str) -> Marking:
    h, w = img.shape[:2]
    return Marking(image=Path(img_path).name, width=w, height=h, regions=r.regions, model_version=run_id,
                   status="auto", confidence=r.confidence, ai=r.ai, quality=r.quality)


def save_result(img_path: Path, img, r: Result, run_id: str, out_dir: Path | None) -> Marking:
    """Write the JSON; an outline the AI picked also becomes a training example."""
    m = to_marking(img_path, img, r, run_id)
    write_marking(m, marking_path_for(img_path, out_dir))
    known = dataset.load_index().get(dataset.item_id(Path(img_path)), {}).get("kind")
    if r.ai and r.ai.get("ok") and m.regions and known in (None, "ai"):  # never replaces a person's marking
        dataset.add_to_pool(Path(img_path), m, "ai")
    return m


def mark_folder(images_dir, output_dir, predictor: Predictor, rules: Rules, progress=lambda f, m: None,
                should_stop=lambda: False, only: list[str] | None = None, s=None, use_ai: bool = False) -> dict:
    out_dir = Path(output_dir) if output_dir else None
    images = [Path(p) for p in only] if only else list_images(images_dir, skip=out_dir)
    done = {"marked": 0, "skipped_reviewed": 0, "needs_review": 0, "ai_improved": 0, "ai_failed": 0, "ai_not_needed": 0,
            "poor_quality": 0, "failed": [], "total": len(images)}
    for i, img_path in enumerate(images):
        if should_stop():
            break
        progress(i / max(1, len(images)), f"{'Marking with AI' if use_ai else 'Marking'} {img_path.name} ({i + 1}/{len(images)})")
        jp = marking_path_for(img_path, out_dir)
        if existing_status(jp) == "reviewed":
            done["skipped_reviewed"] += 1
            continue
        try:
            img = load_image(img_path)
            r = mark(predictor, img, rules, s, use_ai)
            save_result(img_path, img, r, predictor.run_id, out_dir)
            done["marked"] += 1
            if r.ai is not None:
                done["ai_not_needed" if r.ai.get("skipped") else "ai_improved" if r.ai.get("ok") else "ai_failed"] += 1
            if r.quality != "good":
                done["poor_quality"] += 1
            if not r.regions or r.confidence < rules.review_below_confidence or r.quality == "blurry":
                done["needs_review"] += 1
        except Exception as e:
            done["failed"].append(f"{img_path.name}: {e}")
    progress(1.0, "Done")
    return done
