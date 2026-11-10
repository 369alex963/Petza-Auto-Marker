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

from . import config, dataset, prompts, reviewer, train
from .batch import mark_folder, save_result
from .device import device_info
from .imageio import list_images, load_image, open_image
from .jobs import Busy, JobRunner
from .schema import Marking, marking_path_for, read_marking, write_marking

app = FastAPI(title="Wound-Recognizer")
# Blocks DNS-rebinding: a web page on another site can't talk to this local app.
from fastapi.middleware.trustedhost import TrustedHostMiddleware  # noqa: E402

app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"])

# Load torch and probe the hardware right away, so the first page load is quick.
threading.Thread(target=device_info, daemon=True, name="warmup").start()
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
    rules = prompts.active_rules()
    public = s.model_dump()
    public["anthropic_api_key"] = bool(s.anthropic_api_key)
    public["openai_api_key"] = bool(s.openai_api_key)
    public["groq_api_key"] = bool(s.groq_api_key)
    return {"device": device_info(), "model": model, "pool": dataset.pool_stats(), "settings": public,
            "rules": {"prompt": prompts.load()["active"], "labels": rules.labels,
                      "review_below_confidence": rules.review_below_confidence},
            "reviewer": reviewer.connections(s), "job": jobs.snapshot()}


class SettingsIn(BaseModel):
    images_dir: str | None = None
    examples_dir: str | None = None
    output_dir: str | None = None
    anthropic_api_key: str | None = None
    reviewer_model: str | None = None
    reviewer_enabled: bool | None = None
    reviewer_provider: str | None = None
    openai_api_key: str | None = None
    openai_model: str | None = None
    groq_api_key: str | None = None
    groq_model: str | None = None
    ai_runs: int | None = None


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
    if data["reviewer_provider"] not in reviewer.PROVIDERS:
        raise HTTPException(400, "Unknown AI provider.")
    if data["ai_runs"] not in (1, 2, 3):
        raise HTTPException(400, "Runs per photo must be 1, 2 or 3.")
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
    item = {"path": str(img), "name": img.name, "status": None, "confidence": None, "regions": 0, "needs_review": False,
            "review": None, "ai": None, "quality": None}
    if jp.is_file():
        try:
            m = read_marking(jp)
            item.update(status=m.status, confidence=m.confidence, regions=len(m.regions), review=m.review,
                        ai=(m.ai or {}).get("ok"), quality=m.quality,
                        size=[m.width, m.height], outline=[r.points[:: max(1, len(r.points) // 60)] for r in m.regions])
            item["needs_review"] = m.status != "reviewed" and (
                not m.regions or (m.confidence or 0) < threshold or m.quality == "blurry"
                or (m.review is not None and not m.review.get("ok", True)))
        except Exception:
            item["status"] = "foreign"
    return item


@app.get("/api/images")
def images():
    s = settings()
    if not s.images_dir:
        return {"items": []}
    out = output_dir(s)
    threshold = prompts.active_rules().review_below_confidence
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


def _use_ai(s) -> bool:
    return s.reviewer_enabled and reviewer.connections(s)["ready"]


@app.post("/api/jobs/mark")
def job_mark(body: MarkIn | None = None):
    s = settings()
    if not s.images_dir:
        raise HTTPException(400, "Choose the images folder first.")
    pred = predictor()
    rules = prompts.active_rules()
    only = [str(check_path(p)) for p in body.only] if body and body.only else None
    use_ai = _use_ai(s)
    return _start("mark", lambda progress, stop: mark_folder(s.images_dir, output_dir(s), pred, rules, progress, stop,
                                                             only=only, s=s, use_ai=use_ai))


@app.post("/api/jobs/review")
def job_review():
    """Mark every photo that is not approved again, this time with the AI's choices."""
    s = settings()
    if not reviewer.connections(s)["ready"]:
        raise HTTPException(400, "Connect an AI account or add an API key in Settings first.")
    pred = predictor()
    rules = prompts.active_rules()
    return _start("review", lambda progress, stop: mark_folder(s.images_dir, output_dir(s), pred, rules, progress, stop,
                                                               s=s, use_ai=True))


# ---------- try one photo ----------

class TryIn(BaseModel):
    path: str
    save: bool = False


def _regions_out(regions) -> list[dict]:
    return [r.model_dump() for r in regions]


@app.post("/api/try")
def try_one(body: TryIn):
    """The exact pipeline of "Mark all" on one photo, with every step shown."""
    import base64
    import time

    from .refine import mark

    p = check_path(body.path)
    if jobs.snapshot().get("status") == "running":
        raise HTTPException(409, "Wait for the current job to finish.")
    s = settings()
    pred = predictor()
    rules = prompts.active_rules()
    img = load_image(p)
    started = time.time()
    use_ai = reviewer.connections(s)["ready"]
    r = mark(pred, img, rules, s, use_ai=use_ai, keep_views=True)
    saved = None
    if body.save:
        jp = marking_path_for(p, output_dir(s))
        if jp.is_file() and read_marking(jp).status == "reviewed":
            raise HTTPException(409, "This photo is approved already; its marking is kept as it is.")
        save_result(p, img, r, pred.run_id, output_dir(s))
        saved = str(jp)
    h, w = img.shape[:2]
    return {
        "size": [w, h], "quality": r.quality, "seconds": round(time.time() - started, 1), "used_ai": use_ai,
        "local": _regions_out(r.local_regions), "final": _regions_out(r.regions), "confidence": r.confidence,
        "candidates": [{"id": c.id, "label": c.label, "outlines": c.outlines} for c in r.candidates],
        "options_image": base64.b64encode(r.options_jpeg).decode() if r.options_jpeg else None,
        "ai": r.ai, "errors": r.errors, "saved": saved, "runs": s.ai_runs,
        "provider": s.reviewer_provider,
    }


# ---------- AI accounts (the official apps' own sign-in) ----------

class ProviderIn(BaseModel):
    provider: str


def _account_provider(name: str) -> str:
    if name not in ("claude", "openai"):
        raise HTTPException(400, "Only Claude and ChatGPT have an account sign-in. Groq uses a free API key.")
    return name


@app.get("/api/ai/accounts")
def ai_accounts():
    return {"claude": reviewer.account("claude"), "openai": reviewer.account("openai")}


@app.post("/api/ai/connect")
def ai_connect(body: ProviderIn):
    """Download the official app if it is missing, then open its sign-in in the browser."""
    name = _account_provider(body.provider)

    def run(progress, stop):
        installed = False
        if not reviewer.account(name)["installed"]:
            progress(0.05, "Downloading the official app")
            reviewer.install(name, lambda m: progress(0.5, m))
            installed = True
        if not reviewer.account(name)["signed_in"]:
            progress(0.9, "Opening the sign-in page in your browser")
            reviewer.sign_in(name)
        return {"provider": name, "installed": installed}

    return _start("connect", run)


@app.post("/api/ai/disconnect")
def ai_disconnect(body: ProviderIn):
    reviewer.sign_out(_account_provider(body.provider))
    return ai_accounts()


# ---------- marking prompts (what the online AI reads) ----------

class PromptIn(BaseModel):
    name: str
    text: str


def _prompts_out(store: dict) -> dict:
    return {**store, "default_name": prompts.DEFAULT_NAME, "default_text": prompts.DEFAULT_TEXT}


@app.get("/api/prompts")
def get_prompts():
    return _prompts_out(prompts.load())


@app.put("/api/prompts")
def put_prompt(body: PromptIn):
    name = body.name.strip()
    if not name or len(name) > 80:
        raise HTTPException(400, "Give the prompt a name (up to 80 characters).")
    return _prompts_out(prompts.put(name, body.text))


class NameIn(BaseModel):
    name: str


@app.post("/api/prompts/select")
def select_prompt(body: NameIn):
    try:
        return _prompts_out(prompts.select(body.name))
    except KeyError:
        raise HTTPException(404, "No prompt with that name.")


@app.post("/api/prompts/delete")
def delete_prompt(body: NameIn):
    try:
        return _prompts_out(prompts.delete(body.name))
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/reviewer/test")
def reviewer_test():
    """One small real call, so Settings can show which connection works."""
    return reviewer.test(settings())


@app.get("/api/jobs/current")
def job_current():
    return jobs.snapshot()


@app.post("/api/jobs/cancel")
def job_cancel():
    jobs.cancel()
    return jobs.snapshot()


@app.get("/api/runs")
def runs():
    cur = train.current_run_dir()
    out = []
    for r in train.list_runs()[::-1]:
        r["current"] = bool(cur and cur.name == r["run_id"])
        r["available"] = (config.RUNS / r["run_id"] / "model.onnx").is_file()
        out.append(r)
    return {"runs": out}


@app.post("/api/runs/{run_id}/use")
def use_run(run_id: str):
    """Switch back to an earlier model (or forward to one that was not promoted)."""
    if Path(run_id).name != run_id or not (config.RUNS / run_id / "model.onnx").is_file():
        raise HTTPException(404, "That model is not on this computer.")
    if jobs.snapshot().get("status") == "running":
        raise HTTPException(409, "Wait for the current job to finish.")
    train.set_current(run_id)
    with _pred_lock:
        _predictor["run"] = None
    return status()


@app.get("/api/holdout")
def holdout():
    """Per-photo test scores of the model in use, hardest first."""
    run = train.current_run_dir()
    if run is None:
        return {"items": []}
    meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
    index = dataset.load_index()
    items = []
    for p in sorted((meta.get("score") or {}).get("per_image", []), key=lambda p: p["iou"]):
        entry = index.get(p["id"])
        if entry:
            items.append({**p, "name": Path(entry["source"]).name, "kind": entry.get("kind")})
    return {"items": items}


_compare_cache: dict[tuple[str, str], list] = {}


def _pool_paths(iid: str) -> tuple[Path, Path]:
    for i, img_path, json_path in dataset.pool_items():
        if i == iid:
            return img_path, json_path
    raise HTTPException(404, "That test photo is no longer in the pool.")


@app.get("/api/compare")
def compare(id: str, max: int = 900):
    """A held-out photo with the human outline (dark) and the app's outline (pink)."""
    import cv2
    import numpy as np

    img_path, json_path = _pool_paths(id)
    img = load_image(img_path)
    truth = read_marking(json_path)
    pred = predictor()
    key = (str(_predictor["run"]), id)
    if key not in _compare_cache:  # one prediction serves the thumbnail and the large view
        _compare_cache[key] = pred.predict(img, prompts.active_rules())[0]
    regions = _compare_cache[key]
    scale = min(1.0, max / float(np.max(img.shape[:2])))
    view = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else img.copy()
    view = np.ascontiguousarray(view[:, :, ::-1])  # RGB -> BGR for cv2
    width = 2 if max < 600 else 3  # (`max` is the size argument here, not the builtin)

    def draw(regs, color):
        for r in regs:
            pts = np.round(np.asarray(r.points, dtype=np.float64) * scale).astype(np.int32)
            cv2.polylines(view, [pts], True, color, width, cv2.LINE_AA)

    draw(truth.regions, (60, 40, 40))
    draw(regions, (170, 70, 235))
    ok, buf = cv2.imencode(".jpg", view, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return Response(buf.tobytes(), media_type="image/jpeg", headers={"Cache-Control": "no-store"})


# ---------- frontend ----------

DIST = config.ROOT / "frontend" / "dist"
if DIST.is_dir():
    app.mount("/", StaticFiles(directory=DIST, html=True), name="ui")
else:
    @app.get("/")
    def no_ui():
        return JSONResponse({"error": "Frontend not built. Run: cd frontend && npm install && npm run build"})
