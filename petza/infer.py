"""Image -> regions. One pipeline, two backends (ONNX Runtime in the app,
PyTorch during training evaluation), so what we measure is what we ship.

Stage 1: the whole image, letterboxed to the model size, with flip TTA.
Stage 2: each wound found in stage 1 is cropped (+25% margin) and predicted
again at full model resolution, then blended back. This gives much finer
edges on big photos where the wound is a small part of the frame.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .geometry import letterbox, mask_to_polygons, polygons_to_mask, unletterbox
from .rules import Rules
from .schema import Region

MEAN = np.array([0.485, 0.456, 0.406], np.float32) * 255
STD = np.array([0.229, 0.224, 0.225], np.float32) * 255

RunFn = Callable[[np.ndarray], np.ndarray]  # (N,3,S,S) float32 -> (N,C,S,S) probabilities


def preprocess(img: np.ndarray) -> np.ndarray:
    return ((img.astype(np.float32) - MEAN) / STD).transpose(2, 0, 1)


def predict_window(run: RunFn, img: np.ndarray, size: int, tta: bool) -> np.ndarray:
    """Probabilities (h, w, C) for one image or crop."""
    boxed, lb = letterbox(img, size)
    x = preprocess(boxed)
    batch = [x, x[:, :, ::-1], x[:, ::-1, :]] if tta else [x]
    out = run(np.ascontiguousarray(np.stack(batch)))
    probs = out[0]
    if tta:
        probs = (probs + out[1][:, :, ::-1] + out[2][:, ::-1, :]) / 3
    return unletterbox(np.ascontiguousarray(probs.transpose(1, 2, 0)), lb).reshape(img.shape[0], img.shape[1], -1)


def _feather(h: int, w: int, border: int) -> np.ndarray:
    """Weight map that fades to ~0 at crop edges, so crops blend smoothly."""
    ramp_y = np.clip(np.minimum(np.arange(h), np.arange(h)[::-1]) / max(1, border), 0.05, 1)
    ramp_x = np.clip(np.minimum(np.arange(w), np.arange(w)[::-1]) / max(1, border), 0.05, 1)
    return (ramp_y[:, None] * ramp_x[None, :]).astype(np.float32)


def predict_probs(run: RunFn, img: np.ndarray, size: int, tta: bool = True, refine: bool = True) -> np.ndarray:
    h, w = img.shape[:2]
    probs = predict_window(run, img, size, tta)
    if not refine:
        return probs
    fg = (1 - probs[..., 0]) > 0.5
    n, _, stats, _ = cv2.connectedComponentsWithStats(fg.astype(np.uint8))
    acc = probs.copy()
    weight = np.ones((h, w), np.float32)
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < 0.0001 * h * w:
            continue
        mx, my = max(64, int(0.25 * bw)), max(64, int(0.25 * bh))
        x0, y0, x1, y1 = max(0, x - mx), max(0, y - my), min(w, x + bw + mx), min(h, y + bh + my)
        if max(x1 - x0, y1 - y0) > 0.8 * max(h, w):
            continue  # crop would be almost the whole image: no gain
        crop = predict_window(run, img[y0:y1, x0:x1], size, tta)
        f = 3.0 * _feather(y1 - y0, x1 - x0, max(8, min(mx, my) // 2))
        acc[y0:y1, x0:x1] += crop * f[..., None]
        weight[y0:y1, x0:x1] += f
    return acc / weight[..., None]


def probs_to_regions(probs: np.ndarray, classes: list[str], rules: Rules, spacing_frac: float | None) -> tuple[list[Region], float]:
    """Polygons per class + overall confidence (0..1)."""
    h, w = probs.shape[:2]
    labels = probs.argmax(-1)
    diag = (h * h + w * w) ** 0.5
    spacing = rules.point_spacing_px or (spacing_frac * diag if spacing_frac else None)
    regions: list[Region] = []
    for c in range(1, probs.shape[-1]):
        polys = mask_to_polygons(
            labels == c,
            min_area_px=rules.min_area_for(h, w),
            simplify_tolerance_px=rules.tolerance_for(h, w),
            point_spacing_px=spacing,
            max_regions=rules.max_regions,
        )
        for p in polys:
            inside = polygons_to_mask([p], h, w)
            conf = float(probs[..., c][inside].mean()) if inside.any() else 0.0
            regions.append(Region(id=f"r{len(regions) + 1}", label=classes[c - 1], points=p, confidence=round(conf, 4)))
    if rules.max_regions and len(regions) > rules.max_regions:
        regions = sorted(regions, key=lambda r: -len(r.points))[: rules.max_regions]
    if regions:
        confidence = min(r.confidence or 0 for r in regions)
    else:
        confidence = float(1 - (1 - probs[..., 0]).max())  # sure there is no wound?
    return regions, round(confidence, 4)


class Predictor:
    """Loads a trained run on the best device.

    NVIDIA: PyTorch CUDA (model.pt), which needs no extra CUDA install.
    Any other GPU or CPU: ONNX Runtime (model.onnx) with DirectML, else CPU.
    """

    def __init__(self, run_dir: str | Path, backend: str | None = None):
        from .device import inference_backend

        run_dir = Path(run_dir)
        self.meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
        self.size = int(self.meta["size"])
        self.classes = self.meta["classes"]
        self.run_id = self.meta["run_id"]
        self.backend = backend or inference_backend()
        if self.backend == "torch-cuda" and (run_dir / "model.pt").is_file():
            import torch

            from .model import ModelConfig, build_model

            self._torch = torch
            self.model = build_model(ModelConfig(**self.meta["config"]), len(self.classes) + 1, pretrained=False)
            self.model.load_state_dict(torch.load(run_dir / "model.pt", map_location="cpu", weights_only=True))
            self.model.eval().cuda()
            self.provider = "CUDA (PyTorch)"
            return
        import onnxruntime as ort

        from .device import ort_providers

        self.backend = "onnx"
        providers = [p for p in ort_providers() if p != "CUDAExecutionProvider" or backend == "onnx"]
        opts = ort.SessionOptions()
        if "DmlExecutionProvider" in providers:
            opts.enable_mem_pattern = False
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(str(run_dir / "model.onnx"), sess_options=opts, providers=providers)
        self.provider = self.session.get_providers()[0]

    def run(self, batch: np.ndarray) -> np.ndarray:
        if self.backend == "torch-cuda":
            torch = self._torch
            with torch.no_grad(), torch.autocast(device_type="cuda"):
                x = torch.from_numpy(batch.astype(np.float32)).cuda()
                return torch.softmax(self.model(x).float(), dim=1).cpu().numpy()
        return self.session.run(None, {"image": batch.astype(np.float32)})[0]

    def predict(self, img: np.ndarray, rules: Rules, tta: bool = True, refine: bool = True):
        return self.predict_full(img, rules, tta, refine)[:2]

    def predict_full(self, img: np.ndarray, rules: Rules, tta: bool = True, refine: bool = True):
        """(regions, confidence, probability map); the map feeds the AI's options."""
        probs = predict_probs(self.run, img, self.size, tta=tta, refine=refine)
        return (*probs_to_regions(probs, self.classes, rules, self.meta.get("spacing_frac")), probs)
