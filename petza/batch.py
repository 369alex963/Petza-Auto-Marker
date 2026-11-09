"""Mark every image in a folder. Reviewed markings are never overwritten."""
from __future__ import annotations

from pathlib import Path

from .imageio import list_images, load_image
from .infer import Predictor
from .rules import Rules
from .schema import Marking, marking_path_for, read_marking, write_marking


def existing_status(json_path: Path) -> str | None:
    try:
        return read_marking(json_path).status
    except Exception:
        return None  # missing or not ours: safe to write


def mark_folder(images_dir, output_dir, predictor: Predictor, rules: Rules, progress=lambda f, m: None,
                should_stop=lambda: False, only: list[str] | None = None) -> dict:
    out_dir = Path(output_dir) if output_dir else None
    images = [Path(p) for p in only] if only else list_images(images_dir, skip=out_dir)
    done = {"marked": 0, "skipped_reviewed": 0, "needs_review": 0, "failed": [], "total": len(images)}
    for i, img_path in enumerate(images):
        if should_stop():
            break
        progress(i / max(1, len(images)), f"Marking {img_path.name} ({i + 1}/{len(images)})")
        jp = marking_path_for(img_path, out_dir)
        if existing_status(jp) == "reviewed":
            done["skipped_reviewed"] += 1
            continue
        try:
            img = load_image(img_path)
            regions, conf = predictor.predict(img, rules)
            h, w = img.shape[:2]
            m = Marking(image=img_path.name, width=w, height=h, regions=regions, model_version=predictor.run_id,
                        status="auto", confidence=conf)
            write_marking(m, jp)
            done["marked"] += 1
            if not regions or conf < rules.review_below_confidence:
                done["needs_review"] += 1
        except Exception as e:
            done["failed"].append(f"{img_path.name}: {e}")
    progress(1.0, "Done")
    return done
