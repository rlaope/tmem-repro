"""SVD-initialized fast-weight LoRA (TMEM, arXiv:2606.04536, Sec. 3.2).

For a frozen linear weight W (d_out x d_in) with W = U S V^T:
    A0 = S_r V_r^T        (r x d_in, frozen)
    B0 = 0                (d_out x r, trained online)
    y  = W x + B (A0 x)

Only B is updated at test time. Because B0 = 0 the wrapped model starts
exactly at the base policy theta_0.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import torch
from torch import nn

DEFAULT_TARGETS = ("gate_proj", "up_proj", "down_proj")


class SVDLoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, scale: float = 1.0):
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.rank = rank
        self.scale = scale
        W = base.weight.detach().to(device="cpu", dtype=torch.float32)
        # Top-r right singular vectors scaled by singular values.
        _, S, Vh = torch.linalg.svd(W, full_matrices=False)
        A0 = S[:rank, None] * Vh[:rank]  # (r, d_in)
        self.register_buffer("A", A0.to(base.weight.device), persistent=True)
        # requires_grad stays off outside online_sft so ordinary inference never builds a graph.
        self.B = nn.Parameter(
            torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32),
            requires_grad=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        z = x.to(torch.float32) @ self.A.t()  # (..., r)
        delta = (z @ self.B.t()) * self.scale
        return out + delta.to(out.dtype)


@dataclass
class FastWeights:
    """Handle on every SVDLoRALinear inside a model."""

    modules: Dict[str, SVDLoRALinear]

    def parameters(self) -> List[nn.Parameter]:
        return [m.B for m in self.modules.values()]

    def state(self) -> Dict[str, torch.Tensor]:
        return {k: m.B.detach().clone() for k, m in self.modules.items()}

    def load(self, state: Dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            for k, m in self.modules.items():
                m.B.copy_(state[k])

    def reset(self) -> None:
        with torch.no_grad():
            for m in self.modules.values():
                m.B.zero_()

    def norm(self) -> float:
        return float(sum(m.B.detach().float().pow(2).sum() for m in self.modules.values()) ** 0.5)


def _decoder_layers(model: nn.Module) -> nn.ModuleList:
    for path in ("model.layers", "transformer.h", "model.decoder.layers"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
        except AttributeError:
            continue
        return obj
    raise ValueError("could not locate decoder layers on this model")


def attach_svd_lora(
    model: nn.Module,
    rank: int = 6,
    last_n_layers: int = 4,
    targets: Iterable[str] = DEFAULT_TARGETS,
    scale: float = 1.0,
) -> FastWeights:
    """Wrap FFN projections of the last `last_n_layers` layers. Freezes the rest of the model."""
    for p in model.parameters():
        p.requires_grad_(False)
    layers = _decoder_layers(model)
    n = len(layers)
    if not 0 < last_n_layers <= n:
        raise ValueError(f"last_n_layers must be in 1..{n}")
    modules: Dict[str, SVDLoRALinear] = {}
    for li in range(n - last_n_layers, n):
        mlp = layers[li].mlp
        for name in targets:
            base = getattr(mlp, name)
            if isinstance(base, SVDLoRALinear):
                raise ValueError(f"layer {li}.{name} already wrapped")
            wrapped = SVDLoRALinear(base, rank=rank, scale=scale)
            setattr(mlp, name, wrapped)
            modules[f"layers.{li}.mlp.{name}"] = wrapped
    return FastWeights(modules)
