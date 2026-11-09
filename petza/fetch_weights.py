"""Download the ImageNet encoder weights once into models/pretrained (inside this folder)."""
from .model import CPU_CONFIG, GPU_CONFIG, PRETRAINED, fetch_pretrained

if __name__ == "__main__":
    for cfg in (GPU_CONFIG, CPU_CONFIG):
        print(("Downloaded " if fetch_pretrained(cfg.encoder) else "Already have ") + f"{cfg.encoder} weights in {PRETRAINED}")
