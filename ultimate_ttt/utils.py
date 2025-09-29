"""Utility helpers shared across reinforcement-learning pipelines."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .game import Player


def opponent(player: Player) -> Player:
    return "O" if player == "X" else "X"


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))


def masked_softmax(logits: Sequence[float], mask: Sequence[bool], temperature: float = 1.0) -> np.ndarray:
    arr = np.asarray(logits, dtype=np.float64)
    mask_arr = np.asarray(mask, dtype=bool)
    if arr.shape != mask_arr.shape:
        raise ValueError("logits and mask must share the same shape")
    if not mask_arr.any():
        return np.full_like(arr, 1.0 / arr.size if arr.size else 0.0)
    scaled = arr / max(temperature, 1e-6)
    scaled[~mask_arr] = -np.inf
    max_val = np.max(scaled[mask_arr])
    exps = np.exp(np.clip(scaled - max_val, -50, 50))
    exps[~mask_arr] = 0.0
    total = exps.sum()
    if total <= 0:
        uniform = np.zeros_like(arr)
        uniform[mask_arr] = 1.0 / mask_arr.sum()
        return uniform
    return exps / total


def safe_normalise(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    total = arr.sum()
    if total <= 0:
        if arr.size == 0:
            return arr
        return np.full_like(arr, 1.0 / arr.size)
    return arr / total


def dirichlet_noise(alpha: float, size: int) -> np.ndarray:
    return np.random.dirichlet([alpha] * size)


@dataclass
class ReplaySample:
    planes: np.ndarray
    policy: np.ndarray
    value: float
    legal_mask: np.ndarray


class ReplayBuffer:
    """A simple FIFO replay buffer for AlphaZero-style training."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._storage: list[ReplaySample] = []

    def __len__(self) -> int:
        return len(self._storage)

    def append(self, sample: ReplaySample) -> None:
        self._storage.append(sample)
        if len(self._storage) > self.capacity:
            overflow = len(self._storage) - self.capacity
            if overflow > 0:
                del self._storage[:overflow]

    def sample_batch(self, batch_size: int) -> list[ReplaySample]:
        if len(self._storage) < batch_size:
            raise ValueError("Not enough samples in replay buffer")
        indices = np.random.choice(len(self._storage), size=batch_size, replace=False)
        return [self._storage[i] for i in indices]


__all__ = [
    "ReplayBuffer",
    "ReplaySample",
    "dirichlet_noise",
    "masked_softmax",
    "opponent",
    "safe_normalise",
    "set_random_seed",
]
