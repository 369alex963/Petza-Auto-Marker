"""Pick the best hardware available on this PC.

Inference: ONNX Runtime with CUDA (NVIDIA) -> DirectML (any DX12 GPU:
AMD, Intel, integrated) -> CPU. Training: PyTorch CUDA if present, else CPU.
"""
from __future__ import annotations

import functools
import os

ORT_ORDER = ["CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]


@functools.lru_cache(maxsize=1)
def torch_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


@functools.lru_cache(maxsize=1)
def ort_providers() -> list[str]:
    import onnxruntime as ort

    available = set(ort.get_available_providers())
    return [p for p in ORT_ORDER if p in available] or ["CPUExecutionProvider"]


@functools.lru_cache(maxsize=1)
def inference_backend() -> str:
    """'torch-cuda' on NVIDIA (no separate CUDA install needed), else 'onnx'."""
    try:
        import torch

        if torch.cuda.is_available():
            return "torch-cuda"
    except Exception:
        pass
    return "onnx"


@functools.lru_cache(maxsize=1)
def device_info() -> dict:
    info = {"train_device": "cpu", "gpu_name": None, "cpu_threads": os.cpu_count(), "ort_providers": []}
    try:
        import torch

        if torch.cuda.is_available():
            info["train_device"] = "cuda"
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
    except Exception as e:  # torch missing or broken: inference can still run
        info["torch_error"] = str(e)
    try:
        info["ort_providers"] = ort_providers()
    except Exception as e:
        info["ort_error"] = str(e)
    names = {"CUDAExecutionProvider": "NVIDIA GPU (CUDA)", "DmlExecutionProvider": "GPU (DirectML)", "CPUExecutionProvider": "CPU"}
    if inference_backend() == "torch-cuda":
        info["inference_label"] = f"NVIDIA GPU (CUDA), {info['gpu_name']}"
    else:
        onnx = [p for p in info["ort_providers"] if p != "CUDAExecutionProvider"] or ["CPUExecutionProvider"]
        info["inference_label"] = names.get(onnx[0])
    return info
