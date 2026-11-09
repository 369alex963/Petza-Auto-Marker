"""Teach: train on the pool, score on the held-out set, promote only if not worse.

The held-out images are never trained on. The new model and the current
model are both scored on the same held-out set with the full production
pipeline (two-stage + TTA), and the new one replaces the current one only
if its mean IoU is at least as good.
"""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import cv2
import numpy as np

from . import config, dataset
from .geometry import letterbox
from .imageio import load_image
from .infer import Predictor, predict_probs, preprocess
from .metrics import score_pair
from .model import build_model, config_for, export_onnx
from .schema import now_iso, read_marking

CACHE_LONG_SIDE = 1600
MIN_TRAIN = 3


class TrainError(Exception):
    pass


# ---------- current-model pointer ----------

def current_run_dir() -> Path | None:
    ptr = config.RUNS / "current.json"
    if ptr.is_file():
        run = config.RUNS / json.loads(ptr.read_text(encoding="utf-8"))["run_id"]
        if (run / "model.onnx").is_file():
            return run
    return None


def set_current(run_id: str) -> None:
    config.RUNS.mkdir(parents=True, exist_ok=True)
    (config.RUNS / "current.json").write_text(json.dumps({"run_id": run_id}), encoding="utf-8")


def list_runs() -> list[dict]:
    log = config.RUNS / "runs.jsonl"
    if not log.is_file():
        return []
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]


# ---------- data ----------

def _cached(iid: str, img_path: Path, json_path: Path, classes: list[str]):
    """Downscaled image + class mask, cached as PNG files next to the pool."""
    stamp = int(max(img_path.stat().st_mtime, json_path.stat().st_mtime))
    folder = config.CACHE / "train"
    ip, mp = folder / f"{iid}_{stamp}.jpg", folder / f"{iid}_{stamp}_m.png"
    if ip.is_file() and mp.is_file():
        img = cv2.cvtColor(cv2.imread(str(ip)), cv2.COLOR_BGR2RGB)
        return img, cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
    full = load_image(img_path)
    h, w = full.shape[:2]
    scale = min(1.0, CACHE_LONG_SIDE / max(h, w))
    img = cv2.resize(full, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else full
    mask = dataset.class_mask(read_marking(json_path), classes, img.shape[0], img.shape[1], scale)
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob(f"{iid}_*"):
        old.unlink()
    cv2.imwrite(str(ip), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 97])
    cv2.imwrite(str(mp), mask)
    return img, mask


def build_augment(size: int):
    import albumentations as A

    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.3),
        A.Affine(scale=(0.8, 1.25), rotate=(-25, 25), translate_percent=(-0.06, 0.06), p=0.7),
        A.Perspective(scale=(0.02, 0.06), p=0.25),
        A.OneOf([A.GaussianBlur(blur_limit=(3, 7)), A.MotionBlur(blur_limit=(3, 9)), A.Defocus(radius=(1, 3))], p=0.35),
        A.ImageCompression(quality_range=(35, 95), p=0.4),
        A.GaussNoise(std_range=(0.01, 0.06), p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
        A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=25, val_shift_limit=20, p=0.5),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.3),
        A.RandomGamma(gamma_limit=(70, 140), p=0.3),
        A.CLAHE(p=0.15),
        A.RandomShadow(p=0.15),
    ])


def sample(img: np.ndarray, mask: np.ndarray, size: int, rng: random.Random):
    """Half the time the whole image (like stage 1), half a zoomed crop around a wound (like stage 2)."""
    h, w = mask.shape
    if rng.random() < 0.5 and mask.any():
        n, _, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8))
        if n > 1:
            i = rng.randrange(1, n)
            x, y, bw, bh, _ = stats[i]
            margin = rng.uniform(0.1, 0.6)
            mx, my = max(48, int(margin * bw)), max(48, int(margin * bh))
            jx, jy = int(rng.uniform(-0.15, 0.15) * bw), int(rng.uniform(-0.15, 0.15) * bh)
            x0, y0 = max(0, x - mx + jx), max(0, y - my + jy)
            x1, y1 = min(w, x + bw + mx + jx), min(h, y + bh + my + jy)
            if x1 - x0 > 16 and y1 - y0 > 16:
                img, mask = img[y0:y1, x0:x1], mask[y0:y1, x0:x1]
    bi, _ = letterbox(img, size)
    bm, _ = letterbox(mask, size)
    return bi, bm


class PoolDataset:
    def __init__(self, items, size: int, length: int, augment):
        self.items, self.size, self.length, self.augment = items, size, length, augment

    def __len__(self):
        return self.length

    def __getitem__(self, k):
        import torch

        rng = random.Random()
        img, mask = self.items[rng.randrange(len(self.items))]
        bi, bm = sample(img, mask, self.size, rng)
        out = self.augment(image=bi, mask=bm)
        x = torch.from_numpy(preprocess(out["image"]).copy())
        return x, torch.from_numpy(out["mask"].astype(np.int64))


# ---------- scoring ----------

def evaluate(run_fn, size: int, classes: list[str], holdout, tta=True, refine=True) -> dict:
    """Mean IoU / Dice / boundary-F over held-out images (full resolution)."""
    per_image = []
    for iid, img_path, json_path in holdout:
        img = load_image(img_path)
        h, w = img.shape[:2]
        gt = dataset.class_mask(read_marking(json_path), classes, h, w)
        probs = predict_probs(run_fn, img, size, tta=tta, refine=refine)
        pred = probs.argmax(-1)
        scores = [score_pair(pred == c, gt == c) for c in range(1, len(classes) + 1) if (gt == c).any() or (pred == c).any()]
        s = {k: float(np.mean([x[k] for x in scores])) if scores else 1.0 for k in ("iou", "dice", "boundary_f")}
        per_image.append({"id": iid, **{k: round(v, 4) for k, v in s.items()}})
    agg = {k: round(float(np.mean([p[k] for p in per_image])), 4) if per_image else None for k in ("iou", "dice", "boundary_f")}
    return {**agg, "per_image": per_image}


def _torch_runner(model, device):
    import torch

    def run(batch: np.ndarray) -> np.ndarray:
        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=device == "cuda"):
            x = torch.from_numpy(batch).to(device)
            return torch.softmax(model(x).float(), dim=1).cpu().numpy()

    return run


# ---------- main entry ----------

def train(progress=lambda f, m: None, should_stop=lambda: False, epochs: int | None = None, cfg=None) -> dict:
    import segmentation_models_pytorch as smp
    import torch
    from torch.utils.data import DataLoader

    from .device import torch_device

    device = torch_device()
    cfg = cfg or config_for(device)
    classes = dataset.labels()
    items = dataset.pool_items()
    train_ids, hold_ids = dataset.split([i for i, _, _ in items])
    if len(train_ids) < MIN_TRAIN:
        raise TrainError(f"Need at least {MIN_TRAIN} marked examples to train (have {len(train_ids)}). Import examples first.")
    by_id = {i: (i, p, j) for i, p, j in items}
    holdout = [by_id[i] for i in hold_ids]

    progress(0.01, "Preparing images")
    data = []
    for n, iid in enumerate(train_ids):
        _, p, j = by_id[iid]
        data.append(_cached(iid, p, j, classes))
        progress(0.01 + 0.04 * n / len(train_ids), f"Preparing images {n + 1}/{len(train_ids)}")

    # Fine-tune from the current model when compatible; otherwise start from ImageNet.
    model = build_model(cfg, len(classes) + 1)
    current = current_run_dir()
    finetune = False
    if current:
        meta = json.loads((current / "meta.json").read_text(encoding="utf-8"))
        if meta.get("classes") == classes and meta.get("config", {}).get("encoder") == cfg.encoder and (current / "model.pt").is_file():
            model.load_state_dict(torch.load(current / "model.pt", map_location="cpu", weights_only=True))
            finetune = True
    model.to(device)

    n_train = len(train_ids)
    if epochs is None:
        epochs = (20 if finetune else 45) if device == "cuda" else (10 if finetune else 25)
    steps_per_epoch = max(16, math.ceil(max(n_train, 64) / cfg.batch))
    loader = DataLoader(PoolDataset(data, cfg.size, steps_per_epoch * cfg.batch, build_augment(cfg.size)),
                        batch_size=cfg.batch, num_workers=0, drop_last=True)
    lr = cfg.lr * (0.4 if finetune else 1.0)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    total = epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=total, pct_start=0.1)
    dice_loss = smp.losses.DiceLoss("multiclass", from_logits=True)
    ce = torch.nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(enabled=device == "cuda")

    run_id = time.strftime("m-%Y%m%d-%H%M%S")
    run_dir = config.RUNS / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    best = {"iou": -1.0}
    eval_every = max(1, epochs // 6)
    started = time.time()
    step = 0
    for epoch in range(epochs):
        model.train()
        for x, y in loader:
            if should_stop():
                raise TrainError("Stopped")
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type="cuda", enabled=device == "cuda"):
                logits = model(x)
                loss = ce(logits, y) + dice_loss(logits, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            if step % 5 == 0:
                left = (time.time() - started) / step * (total - step)
                progress(0.05 + 0.85 * step / total, f"Epoch {epoch + 1}/{epochs}, loss {loss.item():.3f}, about {int(left // 60)} min {int(left % 60)} s left")
        last = epoch == epochs - 1
        if holdout and ((epoch + 1) % eval_every == 0 or last):
            model.eval()
            score = evaluate(_torch_runner(model, device), cfg.size, classes, holdout, tta=False, refine=True)
            if score["iou"] > best["iou"]:
                best = {"iou": score["iou"], "epoch": epoch + 1}
                torch.save(model.state_dict(), run_dir / "model.pt")
            progress(0.05 + 0.85 * step / total, f"Epoch {epoch + 1}: held-out IoU {score['iou']:.3f} (best {best['iou']:.3f})")
    if not holdout:
        torch.save(model.state_dict(), run_dir / "model.pt")

    progress(0.92, "Exporting model")
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location="cpu", weights_only=True))
    model.eval()
    export_onnx(model.cpu(), cfg.size, run_dir / "model.onnx")
    meta = {
        "run_id": run_id, "created_at": now_iso(), "size": cfg.size, "classes": classes, "config": cfg.to_dict(),
        "device": device, "epochs": epochs, "finetuned_from": json.loads((current / "meta.json").read_text())["run_id"] if finetune else None,
        "train_count": n_train, "holdout_ids": hold_ids, "spacing_frac": dataset.spacing_fraction(),
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    progress(0.95, "Scoring new model on held-out images")
    new_pred = Predictor(run_dir)
    new_score = evaluate(new_pred.run, cfg.size, classes, holdout) if holdout else {"iou": None, "per_image": []}
    old_score = None
    if current and holdout:
        try:
            old_pred = Predictor(current)
            if old_pred.classes == classes:
                old_score = evaluate(old_pred.run, old_pred.size, classes, holdout)
        except Exception:
            old_score = None
    promoted = old_score is None or new_score["iou"] is None or new_score["iou"] >= old_score["iou"]
    meta["score"] = new_score
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if promoted:
        set_current(run_id)
    summary = {
        "run_id": run_id, "created_at": meta["created_at"], "train_count": n_train, "holdout_count": len(holdout),
        "iou": new_score.get("iou"), "dice": new_score.get("dice"), "boundary_f": new_score.get("boundary_f"),
        "previous_iou": old_score["iou"] if old_score else None, "promoted": promoted, "device": device,
        "minutes": round((time.time() - started) / 60, 1), "finetuned": finetune,
    }
    with open(config.RUNS / "runs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(summary) + "\n")
    progress(1.0, "New model is live" if promoted else "Kept the previous model (it scored better)")
    return summary
