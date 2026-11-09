"""End to end: import examples -> teach -> mark a folder -> score. Uses real training."""
import importlib
import json

import pytest

torch = pytest.importorskip("torch")


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("PETZA_DATA", str(tmp_path / "data"))
    import petza.config as config

    importlib.reload(config)
    import petza.dataset as dataset
    import petza.train as train

    importlib.reload(dataset)
    importlib.reload(train)
    return tmp_path, config, dataset, train


def test_import_teach_mark(env):
    from synthetic import write_dataset

    tmp, config, dataset, train = env
    from petza.batch import mark_folder
    from petza.infer import Predictor
    from petza.metrics import iou
    from petza.model import ModelConfig
    from petza.rules import Rules
    from petza.schema import read_marking

    write_dataset(tmp / "examples", 40, seed=1)
    res = dataset.import_examples(tmp / "examples")
    assert res["imported"] == 40 and not res["bad"]

    cfg = ModelConfig(arch="Unet", encoder="resnet34", size=320, batch=8, lr=1e-3)
    summary = train.train(epochs=8, cfg=cfg)
    assert summary["promoted"] and summary["holdout_count"] >= 1
    assert summary["iou"] > 0.85, summary

    # Mark a fresh folder; JSON must line up with the ground truth.
    imgs = write_dataset(tmp / "new", 6, seed=99)
    gt = {p.name: json.loads(p.with_suffix(".json").read_text()) for p in imgs}
    for p in imgs:
        p.with_suffix(".json").unlink()
    out = tmp / "out"
    done = mark_folder(tmp / "new", out, Predictor(train.current_run_dir()), Rules())
    assert done["marked"] == 6 and not done["failed"]

    import numpy as np
    from petza.geometry import polygons_to_mask

    scores = []
    for name, g in gt.items():
        m = read_marking(out / name.replace(".jpg", ".json"))
        h, w = m.height, m.width
        scores.append(iou(polygons_to_mask([r.points for r in m.regions], h, w),
                          polygons_to_mask([s["points"] for s in g["shapes"]], h, w)))
    assert np.mean(scores) > 0.85, scores
