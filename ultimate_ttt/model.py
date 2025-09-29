"""Neural network used by the AlphaZero-style agent."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .features import FEATURE_CHANNELS

__all__ = ["PolicyValueNet", "masked_softmax"]


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(dtype=torch.bool)
    masked_logits = logits.masked_fill(~mask, -1e9)
    return torch.softmax(masked_logits, dim=-1)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + x
        return F.relu(out)


class PolicyValueNet(nn.Module):
    def __init__(
        self,
        channels: int = 64,
        num_blocks: int = 8,
        policy_hidden: int = 2,
        value_hidden: int = 128,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(FEATURE_CHANNELS, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.trunk = nn.ModuleList(ResidualBlock(channels) for _ in range(num_blocks))

        self.policy_conv = nn.Conv2d(channels, policy_hidden, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(policy_hidden)
        self.policy_fc = nn.Linear(policy_hidden * 9 * 9, 81)

        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(9 * 9, value_hidden)
        self.value_fc2 = nn.Linear(value_hidden, 1)

    def forward(
        self, x: torch.Tensor, legal_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        out = self.stem(x)
        for block in self.trunk:
            out = block(out)

        policy = self.policy_conv(out)
        policy = self.policy_bn(policy)
        policy = F.relu(policy)
        policy = policy.view(policy.size(0), -1)
        policy = self.policy_fc(policy)

        if legal_mask is not None:
            policy = policy.masked_fill(~legal_mask.to(dtype=torch.bool), -1e9)

        value = self.value_conv(out)
        value = self.value_bn(value)
        value = F.relu(value)
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        return policy, value

    @torch.no_grad()
    def inference(
        self,
        planes: np.ndarray,
        legal_mask: np.ndarray,
        device: Optional[torch.device] = None,
    ) -> Tuple[np.ndarray, float]:
        self.eval()
        tensor = torch.from_numpy(planes).unsqueeze(0)
        legal = torch.from_numpy(legal_mask.astype(np.bool_)).unsqueeze(0)
        if device is not None:
            tensor = tensor.to(device)
            legal = legal.to(device)
        logits, value = self.forward(tensor, legal_mask=legal)
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        return probs, float(value.item())
