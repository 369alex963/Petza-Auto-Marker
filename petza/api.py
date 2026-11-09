"""HTTP API for the local web app (loopback only)."""
from __future__ import annotations

import hashlib
import io
import json
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config, dataset, train
from .batch import mark_folder
from .device import device_info
from .imageio import list_images, load_image, open_image
from .jobs import Busy, JobRunner
from .rules import load_rules
from .schema import Marking, marking_path_for, read_marking, write_marking

app = FastAPI(title="Petza Auto-Marker")
# Blocks DNS-rebinding: a web page on another site can't talk to this local app.
from fastapi.middleware.trustedhost import TrustedHostMiddleware  # noqa: E402

app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"])
RULES_EXTS = {".md", ".txt", ".markdown"}
jobs = JobRunner()
_predictor = {"run": None, "obj": None}
_pred_lock = threading.Lock()


def predictor():
    from .infer import Predictor

    run = train.current_run_dir()
    if run is None:
        raise HTTPException(409, "No trained model yet. Import examples and press Teach first.")
    with _pred_lock:
        if _predictor["run"] != run:
            _predictor["obj"] = Predictor(run)
            _predictor["run"] = run
        return _predictor["obj"]


def settings() -> config.Settings:
    return config.load_settings()


def output_dir(s: config.Settings) -> Path | None:
    return Path(s.output_dir) if s.output_dir else None


def check_path(path: str) -> Path:
    """Only files inside the folders the user picked (or the training pool)."""
    p = Path(path).resolve()
    s = settings()
    roots = [Path(r).resolve() for r in (s.images_dir, s.examples_dir, s.output_dir) if r] + [config.POOL.resolve()]
    if not any(p == r or r in p.parents for r in roots):
        raise HTTPException(403, "That file is outside the chosen folders.")
    if not p.is_file():
        raise HTTPException(404, "File not found.")
    return p


# ---------- status & settings ----------

@app.get("/api/status")
def status():
    s = settings()
    run = train.current_run_dir()
    model = None
    if run:
        meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
        score = meta.get("score") or {}
        model = {"run_id": meta["run_id"], "created_at": meta["created_at"], "classes": meta["classes"],
                 "iou": score.get("iou"), "dice": score.get("dice"), "boundary_f": score.get("boundary_f"),
                 "train_count": meta.get("train_count"), "device": meta.get("device")}
    rules = load_rules(s.rules_path)
    public = s.model_dump()
    public["anthropic_api_key"] = bool(s.anthropic_api_key)
    return {"device": device_info(), "model": model, "pool": dataset.pool_stats(), "settings": public,
            "rules": {"loaded": bool(s.rules_path and Path(s.rules_path).is_file()), "labels": rules.labels,
                      "text": rules.text, "review_below_confidence": rules.review_below_confidence},
            "job": jobs.snapshot()}


class SettingsIn(BaseModel):
    images_dir: str | None = None
    rules_path: str | None = None
    examples_dir: str | None = None
    output_dir: str | None = None
    anthropic_api_key: str | None = None
    reviewer_model: str | None = None
    reviewer_enabled: bool | None = None


@app.post("/api/settings")
def update_settings(body: SettingsIn):
    s = settings()
    data = s.model_dump()
    for k, v in body.model_dump().items():
        if v is not None:
            data[k] = v.strip() if isinstance(v, str) else v
    for k in ("images_dir", "examples_dir", "output_dir"):
        if data[k] and not Path(data[k]).is_dir():
            if k == "output_dir":
                Path(data[k]).mkdir(parents=True, exist_ok=True)
            else:
                raise HTTPException(400, f"Folder not found: {data[k]}")
    if data["rules_path"] and not Path(data["rules_path"]).is_file():
        raise HTTPException(400, f"Rules file not found: {data['rules_path']}")
    if data["rules_path"] and Path(data["rules_path"]).suffix.lower() not in RULES_EXTS:
        raise HTTPException(400, "The rules file must be a .md, .markdown or .txt file.")
    config.save_settings(config.Settings(**data))
    return status()


class BrowseIn(BaseModel):
    kind: str = "folder"  # folder | file
    initial: str = ""


@app.post("/api/browse")
def browse(body: BrowseIn):
    """Native Windows folder/file picker (the browser can't give real paths)."""
    result: dict = {}

    def pick():
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        initial = body.initial if body.initial and Path(body.initial).exists() else None
        if body.kind == "file":
            result["path"] = filedialog.askopenfilename(parent=root, initialdir=initial,
                                                        filetypes=[("Rules", "*.md *.txt *.markdown"), ("All files", "*.*")])
        else:
            result["path"] = filedialog.askdirectory(parent=root, initialdir=initial, mustexist=True)
        root.destroy()

    t = threading.Thread(target=pick)
    t.start()
    t.join(timeout=600)
    return {"path": (result.get("path") or "").replace("/", "\\")}


# ---------- images & markings ----------

def _summary(img: Path, out: Path | None, threshold: float) -> dict:
    jp = marking_path_for(img, out)
    item = {"path": str(img), "name": img.name, "status": None, "confidence": None, "regions": 0, "needs_review": False, "review": None}
    if jp.is_file():
        try:
            m = read_marking(jp)
            item.update(status=m.status, confidence=m.confidence, regions=len(m.regions), review=m.review)
            item["needs_review"] = m.status != "reviewed" and (not m.regions or (m.confidence or 0) < threshold or (m.review is not None and not m.review.get("ok", True)))
        except Exception:
            item["status"] = "foreign"
    return item


@app.get("/api/images")
def images():
    s = settings()
    if not s.images_dir:
        return {"items": []}
    out = output_dir(s)
    threshold = load_rules(s.rules_path).review_below_confidence
    return {"items": [_summary(p, out, threshold) for p in list_images(s.images_dir, skip=out)]}


@app.get("/api/image")
def image(path: str, max: int = 0):
    p = check_path(path)
    key = hashlib.sha1(f"{p}|{p.stat().st_mtime}|{max}".encode()).hexdigest()
    cached = config.CACHE / "img" / f"{key}.jpg"
    if not cached.is_file():
        im = open_image(p)
        if max:
            im.thumbnail((max, max))
        cached.parent.mkdir(parents=True, exist_ok=True)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=92 if not max else 85)
        cached.write_bytes(buf.getvalue())
    return FileResponse(cached, media_type="image/jpeg", headers={"Cache-Control": "max-age=3600"})


@app.get("/api/marking")
def get_marking(path: str):
    p = check_path(path)
    jp = marking_path_for(p, output_dir(settings()))
    if jp.is_file():
        try:
            return {"marking": read_marking(jp).model_dump(), "json_path": str(jp)}
        except Exception:
            pass
    with open_image(p) as im:
        w, h = im.size
    return {"marking": Marking(image=p.name, width=w, height=h, regions=[]).model_dump(), "json_path": str(jp), "empty": True}


class SaveIn(BaseModel):
    path: str
    marking: Marking


@app.put("/api/marking")
def save_marking(body: SaveIn):
    p = check_path(body.path)
    m = body.marking.model_copy(update={"status": "reviewed", "image": p.name})
    jp = marking_path_for(p, output_dir(settings()))
    write_marking(m, jp)
    dataset.add_to_pool(p, m, "reviewed")
    return {"ok": True, "json_path": str(jp), "pool": dataset.pool_stats()}


# ---------- jobs ----------

def _start(kind, fn):
    try:
        jobs.start(kind, fn)
    except Busy as e:
        raise HTTPException(409, str(e))
    return jobs.snapshot()


@app.post("/api/jobs/import")
def job_import():
    s = settings()
    if not s.examples_dir:
        raise HTTPException(400, "Choose the examples folder first.")
    return _start("import", lambda progress, stop: dataset.import_examples(s.examples_dir, progress, stop))


@app.post("/api/jobs/train")
def job_train():
    def run(progress, stop):
        result = train.train(progress, stop)
        with _pred_lock:
            _predictor["run"] = None  # reload on next use
        return result

    return _start("train", run)


class MarkIn(BaseModel):
    only: list[str] | None = None


@app.post("/api/jobs/mark")
def job_mark(body: MarkIn | None = None):
    s = settings()
    if not s.images_dir:
        raise HTTPException(400, "Choose the images folder first.")
    pred = predictor()
    rules = load_rules(s.rules_path)
    only = [str(check_path(p)) for p in body.only] if body and body.only else None

    def run(progress, stop):
        result = mark_folder(s.images_dir, output_dir(s), pred, rules, progress, stop, only=only)
        if s.reviewer_enabled and s.anthropic_api_key and not stop():
            result["ai_review"] = _review_all(s, rules, progress, stop, only)
        return result

    return _start("mark", run)


def _review_all(s, rules, progress, stop, only=None) -> dict:
    from .reviewer import review

    out = output_dir(s)
    paths = [Path(p) for p in only] if only else list_images(s.images_dir, skip=out)
    counts = {"reviewed": 0, "flagged": 0, "unavailable": 0}
    for i, img_path in enumerate(paths):
        if stop():
            break
        jp = marking_path_for(img_path, out)
        if not jp.is_file():
            continue
        m = read_marking(jp)
        if m.status == "reviewed":
            continue
        progress(i / max(1, len(paths)), f"AI second opinion {i + 1}/{len(paths)}")
        r = review(load_image(img_path), m, rules.text, s.anthropic_api_key, s.reviewer_model)
        if r is None:
            counts["unavailable"] += 1
            continue
        write_marking(m.model_copy(update={"review": r.model_dump()}), jp)
        counts["reviewed"] += 1
        counts["flagged"] += 0 if r.ok else 1
    return counts


@app.post("/api/jobs/review")
def job_review():
    s = settings()
    if not s.anthropic_api_key:
        raise HTTPException(400, "Add an Anthropic API key in Settings to use the AI second opinion.")
    rules = load_rules(s.rules_path)
    return _start("review", lambda progress, stop: _review_all(s, rules, progress, stop))


@app.get("/api/jobs/current")
def job_current():
    return jobs.snapshot()


@app.post("/api/jobs/cancel")
def job_cancel():
    jobs.cancel()
    return jobs.snapshot()


@app.get("/api/runs")
def runs():
    return {"runs": train.list_runs()[::-1]}


# ---------- frontend ----------

DIST = config.ROOT / "frontend" / "dist"
if DIST.is_dir():
    app.mount("/", StaticFiles(directory=DIST, html=True), name="ui")
else:
    @app.get("/")
    def no_ui():
        return JSONResponse({"error": "Frontend not built. Run: cd frontend && npm install && npm run build"})
