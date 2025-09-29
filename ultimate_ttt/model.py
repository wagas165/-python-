"""Policy-value network for Ultimate Tic-Tac-Toe."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:  # pragma: no cover - optional dependency
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover - torch might be unavailable
    torch = None
    nn = None
    F = None


@dataclass
class NetworkConfig:
    channels: int = 64
    residual_blocks: int = 8


if torch is not None:  # pragma: no branch

    class ResidualBlock(nn.Module):
        def __init__(self, channels: int) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(channels)
            self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(channels)

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            out = self.conv1(x)
            out = F.relu(self.bn1(out))
            out = self.conv2(out)
            out = self.bn2(out)
            return F.relu(out + x)


class PolicyValueNet(nn.Module if torch is not None else object):
    """A lightweight residual network that outputs policy logits and value."""

    def __init__(self, input_planes: int = 8, config: Optional[NetworkConfig] = None) -> None:
        if torch is None:  # pragma: no cover - executed only when torch missing
            raise ImportError("PyTorch is required to instantiate PolicyValueNet")
        super().__init__()
        cfg = config or NetworkConfig()
        self.stem = nn.Sequential(
            nn.Conv2d(input_planes, cfg.channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(cfg.channels),
            nn.ReLU(inplace=True),
        )
        blocks = [ResidualBlock(cfg.channels) for _ in range(cfg.residual_blocks)]
        self.trunk = nn.Sequential(*blocks)

        self.policy_head = nn.Sequential(
            nn.Conv2d(cfg.channels, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(2 * 9 * 9, 81),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(cfg.channels, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(9 * 9, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Tanh(),
        )

    def forward(  # type: ignore[override]
        self,
        planes: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(planes)
        x = self.trunk(x)
        policy_logits = self.policy_head(x)
        if legal_mask is not None:
            mask = (legal_mask <= 0)
            policy_logits = policy_logits.masked_fill(mask, float("-inf"))
        value = self.value_head(x)
        return policy_logits, value

    @torch.no_grad()  # type: ignore[misc]
    def predict(
        self,
        planes: np.ndarray,
        legal_mask: np.ndarray,
        device: Optional[torch.device] = None,
    ) -> Tuple[np.ndarray, float]:
        if torch is None:  # pragma: no cover
            raise ImportError("PyTorch is required for prediction")
        dev = device or torch.device("cpu")
        tensor = torch.from_numpy(planes).float().unsqueeze(0).to(dev)
        mask = torch.from_numpy(legal_mask.astype(np.float32)).unsqueeze(0).to(dev)
        logits, value = self.forward(tensor, mask)
        policy = logits.squeeze(0).detach().cpu().numpy()
        value_scalar = float(value.squeeze().detach().cpu().numpy())
        return policy, value_scalar


__all__ = ["NetworkConfig", "PolicyValueNet"]
