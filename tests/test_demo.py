"""Demo data: set up on the first start only, removed in one go."""
import importlib
import json

import pytest
from PIL import Image

from petza.schema import Marking, Region, write_marking


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("PETZA_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("PETZA_DEMO", str(tmp_path / "demo"))
    import petza.config as config

    importlib.reload(config)
    import petza.dataset as dataset
    import petza.demo as demo
    import petza.train as train

    demo = importlib.reload(demo)
    d = tmp_path / "demo"
    for sub in ("Examples", "Photos", "Marked JSON", "model"):
        (d / sub).mkdir(parents=True)
    for n in ("a.png", "b.png"):
        Image.new("RGB", (40, 40), (180, 90, 80)).save(d / "Examples" / n)
        write_marking(Marking(image=n, width=40, height=40, regions=[Region(id="r1", points=[[5, 5], [30, 5], [20, 30]])]),
                      d / "Examples" / n.replace(".png", ".json"))
    Image.new("RGB", (40, 40)).save(d / "Photos" / "p.png")
    (d / "model" / "model.onnx").write_bytes(b"x")
    (d / "model" / "model.pt").write_bytes(b"x")
    meta = {"run_id": "m-demo", "created_at": "2026-11-10T00:00:00Z", "classes": ["wound"], "train_count": 1,
            "holdout_ids": ["a.png"], "score": {"iou": 0.77, "per_image": [{"id": "a.png", "iou": 0.77}]}, "ids_are_names": True}
    (d / "model" / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return config, dataset, demo, train, d, tmp_path


def test_seed_only_on_first_start(env):
    config, dataset, demo, train, d, _ = env
    assert demo.seed()
    s = config.load_settings()
    assert s.images_dir == str(d / "Photos") and demo.active(s)
    run = train.current_run_dir()
    assert run and run.name == "m-demo"
    meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
    assert meta["holdout_ids"] == [dataset.item_id(d / "Examples" / "a.png")] == [meta["score"]["per_image"][0]["id"]]
    assert dataset.pool_stats()["total"] == 2
    assert not demo.seed()  # settings exist now: never again


def test_remove_leaves_a_clean_app(env):
    config, dataset, demo, train, d, tmp = env
    demo.seed()
    own = tmp / "own.png"
    Image.new("RGB", (40, 40)).save(own)
    dataset.add_to_pool(own, Marking(image="own.png", width=40, height=40, regions=[Region(id="r1", points=[[5, 5], [30, 5], [20, 30]])]), "example")
    s = config.load_settings()
    s.reviewer_provider = "claude"
    config.save_settings(s)
    demo.remove()
    s = config.load_settings()
    assert not d.exists() and train.current_run_dir() is None and not config.RUNS.exists()
    assert (s.images_dir, s.examples_dir, s.output_dir) == ("", "", "") and s.reviewer_provider == "claude"
    assert list(dataset.load_index()) == [dataset.item_id(own)] and not demo.active(s)
