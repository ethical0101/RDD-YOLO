"""Hardware detection and training-parameter recommendations.

The defaults are tuned for the primary development machine (RTX 3050 4 GB,
16 GB RAM, Windows) but adapt to whatever hardware is detected.
"""
from __future__ import annotations

import os
import platform
from dataclasses import asdict, dataclass


@dataclass
class HardwareInfo:
    os: str
    python: str
    cpu_cores: int
    ram_gb: float
    cuda_available: bool
    gpu_name: str | None
    vram_gb: float | None
    torch_version: str
    cuda_version: str | None

    def to_dict(self) -> dict:
        return asdict(self)


def _total_ram_gb() -> float:
    try:
        import psutil

        return round(psutil.virtual_memory().total / 1024**3, 1)
    except Exception:
        return 0.0


def detect_hardware() -> HardwareInfo:
    import torch

    cuda = torch.cuda.is_available()
    gpu_name = vram = None
    if cuda:
        props = torch.cuda.get_device_properties(0)
        gpu_name = props.name
        vram = round(props.total_memory / 1024**3, 2)
    return HardwareInfo(
        os=f"{platform.system()} {platform.release()}",
        python=platform.python_version(),
        cpu_cores=os.cpu_count() or 1,
        ram_gb=_total_ram_gb(),
        cuda_available=cuda,
        gpu_name=gpu_name,
        vram_gb=vram,
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
    )


@dataclass
class TrainingRecommendation:
    device: str
    batch: int
    imgsz: int
    workers: int
    amp: bool
    model_scale: str
    reason: str


def recommend_training_params(hw: HardwareInfo, imgsz: int = 640) -> TrainingRecommendation:
    """Pick conservative batch/workers/AMP settings for the detected hardware."""
    # Windows DataLoader workers are separate processes; keep them moderate to
    # avoid exhausting 16 GB of RAM.
    workers = max(0, min(6, hw.cpu_cores // 2, int(hw.ram_gb // 3) if hw.ram_gb else 4))
    if not hw.cuda_available:
        return TrainingRecommendation("cpu", 4, min(imgsz, 416), min(workers, 2), False, "n",
                                      "No CUDA GPU detected: CPU fallback (slow; use for smoke tests).")
    vram = hw.vram_gb or 0
    if vram < 3.5:
        batch, scale = 4, "n"
    elif vram < 6:  # RTX 3050 4 GB lands here
        batch, scale = 8, "n"
    elif vram < 10:
        batch, scale = 16, "s"
    else:
        batch, scale = 32, "s"
    return TrainingRecommendation("0", batch, imgsz, workers, True, scale,
                                  f"{hw.gpu_name} with {vram} GB VRAM: batch {batch}, AMP on, scale '{scale}'.")


if __name__ == "__main__":
    import json

    hw = detect_hardware()
    print(json.dumps(hw.to_dict(), indent=2))
    print(json.dumps(asdict(recommend_training_params(hw)), indent=2))
