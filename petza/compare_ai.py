"""Does the online AI make the marking more exact? Measured, not assumed.

    .venv\\Scripts\\python -m petza.compare_ai

Marks every held-out photo (your own markings, never trained on) twice, with the
local model alone and with the AI's choices, and prints the overlap (IoU) of
each with your marking. Uses the saved Settings and the active marking prompt.
"""
from __future__ import annotations

import numpy as np

from . import config, dataset, prompts, train
from .imageio import load_image
from .infer import Predictor
from .metrics import score_pair
from .refine import mark
from .schema import Marking, read_marking


def _iou(regions, truth: Marking, classes, h, w) -> float:
    pred = dataset.class_mask(Marking(image="x", width=w, height=h, regions=regions), classes, h, w)
    gt = dataset.class_mask(truth, classes, h, w)
    scores = [score_pair(pred == c, gt == c)["iou"] for c in range(1, len(classes) + 1) if (gt == c).any() or (pred == c).any()]
    return float(np.mean(scores)) if scores else 1.0


def main() -> None:
    run = train.current_run_dir()
    if run is None:
        raise SystemExit("No trained model yet.")
    s = config.load_settings()
    pred = Predictor(run)
    rules = prompts.active_rules()
    items = dataset.pool_items()
    _, hold = dataset.split([i for i, _, _ in items])
    rows = []
    for iid, img_path, json_path in (it for it in items if it[0] in hold):
        img = load_image(img_path)
        h, w = img.shape[:2]
        truth = read_marking(json_path)
        r = mark(pred, img, rules, s, use_ai=True)
        local, ai = _iou(r.local_regions, truth, pred.classes, h, w), _iou(r.regions, truth, pred.classes, h, w)
        ok = bool(r.ai and r.ai.get("ok"))
        rows.append((local, ai, ok))
        print(f"{img_path.name}: local {local:.3f}  with AI {ai:.3f}{'' if ok else '  (AI did not answer)'}", flush=True)
    if rows:
        print(f"\n{len(rows)} held-out photos. Mean IoU: local {np.mean([a for a, _, _ in rows]):.3f}, with AI {np.mean([b for _, b, _ in rows]):.3f}"
              f" ({s.reviewer_provider}, {s.ai_runs} {'run' if s.ai_runs == 1 else 'runs'})")


if __name__ == "__main__":
    main()
