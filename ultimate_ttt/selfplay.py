"""Self-play data generation for the AlphaZero-style agent."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from .features import encode_state
from .game import UltimateTicTacToe, index_to_action
from .mcts import MCTS
from .symmetry import SYMMETRIES

__all__ = ["SelfPlaySample", "play_game", "augment_samples"]


@dataclass
class SelfPlaySample:
    planes: np.ndarray
    policy: np.ndarray
    legal_mask: np.ndarray
    value_x: float


def _select_action(visit_counts: np.ndarray, legal_mask: np.ndarray, temperature: float, rng: np.random.Generator) -> int:
    legal_indices = np.flatnonzero(legal_mask)
    if legal_indices.size == 0:
        raise RuntimeError("No legal moves available for selection")

    visits = visit_counts[legal_indices].astype(np.float64)
    if temperature <= 1e-6:
        return int(legal_indices[int(np.argmax(visits))])

    transformed = visits ** (1.0 / max(temperature, 1e-6))
    total = transformed.sum()
    if total <= 0:
        transformed = np.ones_like(transformed) / transformed.size
    else:
        transformed /= total
    choice = rng.choice(len(legal_indices), p=transformed)
    return int(legal_indices[choice])


def play_game(
    mcts: MCTS,
    temperature_moves: int = 16,
    rng: Optional[np.random.Generator] = None,
) -> List[SelfPlaySample]:
    rng = rng or np.random.default_rng()
    game = UltimateTicTacToe()
    samples: List[SelfPlaySample] = []
    move_index = 0

    while not game.terminal:
        planes, legal_mask = encode_state(game)
        _, root = mcts.run(game)
        visit_counts = root.visit_counts.astype(np.float32)

        temperature = 1.0 if move_index < temperature_moves else 1e-6
        action_index = _select_action(visit_counts, legal_mask, temperature, rng)

        policy_target = np.zeros(81, dtype=np.float32)
        legal_indices = np.flatnonzero(legal_mask)
        if temperature <= 1e-6:
            policy_target[action_index] = 1.0
        elif legal_indices.size > 0:
            transformed = visit_counts[legal_indices] ** (1.0 / temperature)
            total = transformed.sum()
            if total > 0:
                transformed /= total
            else:
                transformed = np.ones_like(transformed) / transformed.size
            policy_target[legal_indices] = transformed

        samples.append(
            SelfPlaySample(
                planes=planes,
                policy=policy_target,
                legal_mask=legal_mask.astype(bool),
                value_x=0.0,
            )
        )

        player = game.active_player()
        game.make_move(player, index_to_action(action_index))
        move_index += 1

    outcome = game.terminal_outcome_from_x_perspective()
    for sample in samples:
        sample.value_x = outcome
    return samples


def augment_samples(samples: Sequence[SelfPlaySample]) -> List[SelfPlaySample]:
    augmented: List[SelfPlaySample] = []
    for sample in samples:
        for symmetry in SYMMETRIES:
            planes = symmetry.apply_planes(sample.planes)
            policy = symmetry.apply_policy(sample.policy)
            legal_mask = symmetry.apply_policy(sample.legal_mask.astype(np.float32)) > 0.5
            augmented.append(
                SelfPlaySample(
                    planes=planes,
                    policy=policy,
                    legal_mask=legal_mask,
                    value_x=sample.value_x,
                )
            )
    return augmented
