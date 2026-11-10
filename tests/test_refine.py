"""AI-guided marking: candidates, voting, applying, fall-backs and quality."""
import importlib
import io

import cv2
import numpy as np
import pytest
from PIL import Image

from petza import quality, refine
from petza.refine import ChoiceResult, MissedWound, RegionChoice
from petza.rules import Rules

H = W = 240


def blob(cx, cy, r, peak=1.0):
    yy, xx = np.mgrid[0:H, 0:W]
    return np.clip(peak * (1 - np.hypot(xx - cx, yy - cy) / r), 0, 1)


def probs_map():
    fg = np.maximum(blob(80, 120, 80), blob(195, 60, 40, peak=0.45))  # one clear wound, one faint spot
    return np.stack([1 - fg, fg], -1).astype(np.float32)


class FakePredictor:
    classes = ["wound"]
    size = 64
    run_id = "test"
    meta = {}

    def predict_full(self, img, rules):
        from petza.infer import probs_to_regions

        p = probs_map()
        return (*probs_to_regions(p, self.classes, rules, None), p)

    def run(self, batch):  # the crop re-prediction for a missed wound: "wound everywhere"
        n, _, s, _ = batch.shape
        out = np.zeros((n, 2, s, s), np.float32)
        out[:, 1] = 0.9
        out[:, 0] = 0.1
        return out


def area(polys):
    return sum(cv2.contourArea(np.asarray(p, np.float32)) for p in polys)


def answer(regions, missed=()):
    return ChoiceResult(regions=[RegionChoice(id=i, keep=k, outline=o, label="wound", reason="r") for i, k, o in regions],
                        missed=[MissedWound(box=list(b), reason="m") for b in missed], notes="n")


def test_candidates_have_nested_outlines():
    cands = refine.candidates(probs_map(), ["wound"], Rules(), None)
    assert [c.id for c in cands] == [1, 2]
    big, faint = cands
    assert big.options() == ["A", "B", "C"]
    assert area(big.outlines["A"]) < area(big.outlines["B"]) < area(big.outlines["C"])
    assert faint.options() == ["C"] and faint.default == (False, "B")  # the local model alone drops it


def test_vote_majority_and_ties():
    cands = refine.candidates(probs_map(), ["wound"], Rules(), None)
    d, _ = refine.vote([answer([(1, True, "A"), (2, True, "C")])], cands, ["wound"])
    assert (d[0]["keep"], d[0]["outline"], d[1]["keep"], d[1]["outline"]) == (True, "A", True, "C")
    three = [answer([(1, True, "A")]), answer([(1, True, "A")]), answer([(1, False, "C")])]
    d, _ = refine.vote(three, cands, ["wound"])
    assert d[0]["keep"] and d[0]["outline"] == "A" and d[0]["agreed"] == "2/3"
    tie = [answer([(1, True, "A"), (2, True, "C")]), answer([(1, False, "C"), (2, False, "C")])]
    d, _ = refine.vote(tie, cands, ["wound"])
    assert d[0]["keep"] and not d[1]["keep"]  # a tie keeps what the local model did
    d, _ = refine.vote([answer([(1, True, "A")])], cands, ["wound"])
    assert not d[1]["keep"]  # not mentioned -> local default


def test_missed_boxes_need_a_majority():
    cands = refine.candidates(probs_map(), ["wound"], Rules(), None)
    box = (0.8, 0.8, 0.95, 0.95)
    assert refine.vote([answer([], [box])], cands, ["wound"])[1] == [list(box)]
    assert refine.vote([answer([], [box]), answer([])], cands, ["wound"])[1] == []


@pytest.fixture()
def settings(tmp_path, monkeypatch):
    monkeypatch.setenv("PETZA_DATA", str(tmp_path))
    import petza.config as config

    importlib.reload(config)
    return config.Settings(reviewer_enabled=True, ai_runs=1)


def test_edges_stay_local_by_default(settings, monkeypatch):
    monkeypatch.setattr(refine.reviewer, "ask", lambda *a, **k: (answer([(1, True, "A"), (2, False, "C")]), "Fake", []))
    r = refine.mark(FakePredictor(), np.full((H, W, 3), 180, np.uint8), Rules(), settings, use_ai=True)
    big = refine.candidates(probs_map(), ["wound"], Rules(), None)[0]
    assert len(r.regions) == 1 and abs(area([r.regions[0].points]) - area(big.outlines["B"])) < 1  # A was asked, B kept
    assert r.ai["choices"][0]["kept_local_edge"]


def test_mark_applies_choices_and_finds_missed(settings, monkeypatch):
    settings.ai_edges = True
    img = np.full((H, W, 3), 180, np.uint8)
    asked = []

    def fake_ask(images, prompt, schema, s, salt=""):
        asked.append((images, prompt))
        return answer([(1, True, "A"), (2, True, "C")], [(0.85, 0.85, 0.97, 0.97)]), "Fake account", []

    monkeypatch.setattr(refine.reviewer, "ask", fake_ask)
    r = refine.mark(FakePredictor(), img, Rules(), settings, use_ai=True)
    assert r.ai["ok"] and r.ai["missed"] == 1 and len(r.regions) == 3
    assert len(r.local_regions) == 1
    big = refine.candidates(probs_map(), ["wound"], Rules(), None)[0]
    assert any(abs(area([x.points]) - area(big.outlines["A"])) < 1 for x in r.regions)
    images, prompt = asked[0]
    assert len(images) == 2 and "region 1: A, B, C; region 2: C" in prompt


def test_mark_keeps_local_result_when_ai_fails(settings, monkeypatch):
    monkeypatch.setattr(refine.reviewer, "ask", lambda *a, **k: (None, None, ["offline"]))
    r = refine.mark(FakePredictor(), np.full((H, W, 3), 180, np.uint8), Rules(), settings, use_ai=True)
    assert r.ai["ok"] is False and r.regions == r.local_regions and r.errors == ["offline"]


def test_three_runs_ask_three_times(settings, monkeypatch):
    settings.ai_runs = 3
    salts = []

    def fake_ask(images, prompt, schema, s, salt=""):
        salts.append(salt)
        return answer([(1, True, "B")]), "Fake", []

    monkeypatch.setattr(refine.reviewer, "ask", fake_ask)
    r = refine.mark(FakePredictor(), np.full((H, W, 3), 180, np.uint8), Rules(), settings, use_ai=True)
    assert salts == ["run0", "run1", "run2"] and r.ai["runs"] == 3


def test_quality_and_correction_only_for_poor_photos():
    rng = np.random.default_rng(0)
    good = np.clip(rng.normal(150, 50, (300, 300, 3)), 0, 255).astype(np.uint8)
    dark = (good * 0.25).astype(np.uint8)
    flat = np.full((300, 300, 3), 140, np.uint8)
    flat[::2] = 150
    assert quality.check(good) == "good"
    assert quality.check(dark) == "dark" and quality.needs_correction("dark")
    assert quality.check(flat) == "low_contrast"
    assert not quality.needs_correction("good") and not quality.needs_correction("blurry")
    assert quality.correct(dark).mean() > dark.mean() + 20
    skin = np.zeros((200, 200, 3), np.uint8)
    skin[:] = (120, 70, 55)  # dark skin tone: must stay skin, never turn gray
    skin[60:140, 60:140] = (110, 25, 30)
    fixed = quality.correct(skin).reshape(-1, 3).mean(0)
    assert fixed[0] > fixed[1] + 25 and fixed[0] > fixed[2] + 25


def test_sent_jpeg_has_no_metadata():
    data = refine._jpeg(np.full((50, 60, 3), 120, np.uint8))
    with Image.open(io.BytesIO(data)) as im:
        assert len(im.getexif()) == 0 and "exif" not in im.info and im.size == (60, 50)


def test_ai_items_are_never_held_out(tmp_path, monkeypatch):
    monkeypatch.setenv("PETZA_DATA", str(tmp_path / "data"))
    import petza.config as config

    importlib.reload(config)
    import petza.dataset as dataset

    dataset = importlib.reload(dataset)
    from petza.schema import Marking, Region

    m = Marking(image="x.png", width=10, height=10, regions=[Region(id="r1", points=[[1, 1], [8, 1], [5, 8]])])
    ids = {}
    for k in range(40):
        p = tmp_path / f"img{k}.png"
        Image.new("RGB", (10, 10)).save(p)
        kind = "example" if k < 8 else "ai"
        ids[dataset.add_to_pool(p, m, kind)] = kind
    train, hold = dataset.split(list(ids))
    assert hold and all(ids[i] == "example" for i in hold)
    assert sum(1 for i in train if ids[i] == "ai") == 32
    assert dataset.pool_stats()["ai"] == 32
