"""Model definition, bundled pretrained weights and ONNX export.

Pretrained ImageNet encoder weights are fetched once (setup) into
models/pretrained/ inside this project. Training never touches the
internet after that.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from . import config

PRETRAINED = config.MODELS / "pretrained"


@dataclass
class ModelConfig:
    arch: str
    encoder: str
    size: int
    batch: int
    lr: float

    def to_dict(self) -> dict:
        return asdict(self)


# Measured on an RTX 3070 Ti Laptop (8 GB): ~4 GB and ~8 img/s at batch 2, so it
# still fits when another app shares the card. Bigger cards get a bigger batch.
GPU_CONFIG = ModelConfig(arch="Unet", encoder="efficientnet-b4", size=768, batch=2, lr=3e-4)
CPU_CONFIG = ModelConfig(arch="Unet", encoder="resnet34", size=512, batch=4, lr=3e-4)


def config_for(device: str) -> ModelConfig:
    if device != "cuda":
        return CPU_CONFIG
    import torch

    gb = torch.cuda.get_device_properties(0).total_memory / 2**30
    free = (torch.cuda.mem_get_info()[0] + torch.cuda.memory_reserved()) / 2**30  # other apps may hold part of the card
    batch = 6 if gb >= 15 else 4 if gb >= 11 else 2
    if free < 5.5:  # 768 px would spill into shared memory and run 50x slower
        return ModelConfig(**{**GPU_CONFIG.to_dict(), "size": 512, "batch": 2 if free < 3.5 else 4})
    return ModelConfig(**{**GPU_CONFIG.to_dict(), "batch": batch})


def build_model(cfg: ModelConfig, num_classes: int, pretrained: bool = True):
    import segmentation_models_pytorch as smp
    import torch

    Arch = getattr(smp, cfg.arch)
    model = Arch(encoder_name=cfg.encoder, encoder_weights=None, classes=num_classes, decoder_attention_type="scse")
    weights = PRETRAINED / f"{cfg.encoder}.pth"
    if pretrained and weights.is_file():
        model.encoder.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    return model


def fetch_pretrained(encoder: str) -> bool:
    """Download ImageNet weights once and keep them inside the project."""
    import segmentation_models_pytorch as smp
    import torch

    target = PRETRAINED / f"{encoder}.pth"
    if target.is_file():
        return False
    PRETRAINED.mkdir(parents=True, exist_ok=True)
    m = smp.Unet(encoder_name=encoder, encoder_weights="imagenet", classes=1)
    torch.save(m.encoder.state_dict(), target)
    return True


class _WithSoftmax:
    """Wraps the network so the exported ONNX returns probabilities."""

    def __new__(cls, model):
        import torch

        class Wrapped(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, x):
                return torch.softmax(self.inner(x), dim=1)

        return Wrapped(model)


def export_onnx(model, size: int, path) -> None:
    import torch

    wrapped = _WithSoftmax(model).eval().cpu()
    dummy = torch.zeros(1, 3, size, size)
    kwargs = dict(input_names=["image"], output_names=["probs"], dynamic_axes={"image": {0: "n"}, "probs": {0: "n"}}, opset_version=17)
    try:
        torch.onnx.export(wrapped, dummy, str(path), dynamo=False, **kwargs)
    except TypeError:  # very old torch without the dynamo flag
        torch.onnx.export(wrapped, dummy, str(path), **kwargs)
