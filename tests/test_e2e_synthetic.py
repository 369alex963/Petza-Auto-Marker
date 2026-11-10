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

    # ---- HTTP API on the trained model ----
    import time

    import petza.api as api_mod
    from fastapi.testclient import TestClient

    importlib.reload(api_mod)
    c = TestClient(api_mod.app)
    assert c.post("/api/settings", json={"images_dir": str(tmp / "new"), "output_dir": str(out)}).status_code == 200

    runs = c.get("/api/runs").json()["runs"]
    assert runs[0]["current"] and runs[0]["available"]
    assert c.post(f"/api/runs/{runs[0]['run_id']}/use").status_code == 200
    assert c.post("/api/runs/nope/use").status_code == 404

    hard = c.get("/api/holdout").json()["items"]
    assert hard and hard == sorted(hard, key=lambda h: h["iou"])
    r = c.get(f"/api/compare?id={hard[0]['id']}&max=300")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg" and r.content[:2] == b"\xff\xd8"

    one = str(imgs[0])
    (out / imgs[0].name.replace(".jpg", ".json")).unlink()
    assert c.post("/api/jobs/mark", json={"only": [one]}).status_code == 200
    for _ in range(120):
        job = c.get("/api/jobs/current").json()
        if job["status"] != "running":
            break
        time.sleep(0.5)
    assert job["status"] == "done", job
    assert c.get(f"/api/marking?path={one}").json()["marking"]["regions"]

    # Approve the ticked ones: a person chose them, so they teach the app.
    items = c.get("/api/images").json()["items"]
    picked = min((i for i in items if i["status"] == "auto" and i["regions"]), key=lambda i: i["confidence"] or 0)
    pool_before = c.get("/api/status").json()["pool"]["total"]
    assert c.post("/api/approve", json={"paths": [picked["path"], picked["path"]]}).json()["approved"] == 1
    after = {i["path"]: i for i in c.get("/api/images").json()["items"]}
    assert after[picked["path"]]["status"] == "reviewed" and after[picked["path"]]["approved_by"] == "person"
    assert c.get("/api/status").json()["pool"]["total"] == pool_before + 1

    # Approve all sure ones: sure markings become final, never training examples.
    items = c.get("/api/images").json()["items"]
    sure = [i for i in items if i["status"] == "auto" and i["regions"] and not i["needs_review"]]
    pool_before = c.get("/api/status").json()["pool"]["total"]
    assert c.post("/api/approve-sure").json()["approved"] == len(sure) > 0
    after = {i["path"]: i for i in c.get("/api/images").json()["items"]}
    assert all(after[i["path"]]["status"] == "reviewed" and after[i["path"]]["approved_by"] == "auto" for i in sure)
    assert c.get("/api/status").json()["pool"]["total"] == pool_before
    assert c.post("/api/approve-sure").json()["approved"] == 0
