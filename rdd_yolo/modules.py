"""Custom network modules for the RDD-YOLO architecture.

Only SimAM needs a custom implementation: GhostConv already ships with
Ultralytics and bilinear upsampling is expressed with ``nn.Upsample`` in the
model YAML. ``register_custom_modules()`` must be called before a model YAML
that references ``SimAM`` is parsed, and before a checkpoint containing it is
unpickled (checkpoints reference ``rdd_yolo.modules.SimAM`` by import path).
"""
from __future__ import annotations

import torch
from torch import nn


class SimAM(nn.Module):
    """Simple, parameter-free attention module (Yang et al., ICML 2021).

    For every neuron t in a channel, the minimal energy is
        e_t* = 4(σ² + λ) / ((t − μ)² + 2σ² + 2λ)
    where μ, σ² are the channel's spatial mean and variance. Lower energy means
    the neuron is more distinct from its neighbours and thus more important.
    The feature map is refined as  X̃ = sigmoid(1 / E) ⊙ X.

    In code (following the reference implementation):
        y = (x − μ)² / (4 (σ² + λ)) + 0.5   ==  1 / e_t*
        out = x * sigmoid(y)

    Accepts (and ignores) an optional channel argument so it can be declared
    in Ultralytics YAML files as ``[-1, 1, SimAM, []]``.
    """

    def __init__(self, *args, e_lambda: float = 1e-4):
        super().__init__()
        self.e_lambda = e_lambda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        n = max(h * w - 1, 1)
        d = (x - x.mean(dim=(2, 3), keepdim=True)).pow(2)
        y = d / (4 * (d.sum(dim=(2, 3), keepdim=True) / n + self.e_lambda)) + 0.5
        return x * torch.sigmoid(y)

    def extra_repr(self) -> str:
        return f"e_lambda={self.e_lambda}"


def register_custom_modules() -> None:
    """Expose custom modules to the Ultralytics YAML parser (``parse_model``)."""
    from ultralytics.nn import tasks

    tasks.SimAM = SimAM  # parse_model resolves module names via tasks' globals()
