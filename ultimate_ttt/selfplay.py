"""Self-play data generation for AlphaZero-style training."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np

from .features import EncodedState, augment_state, encode_state
from .game import Player, UltimateTicTacToe, index_to_action
from .mcts import MCTS, MCTSConfig
from .symmetry import SYMMETRIES, apply_symmetry_to_policy
from .utils import ReplaySample, opponent, safe_normalise


@dataclass
class SelfPlayConfig:
    mcts_config: MCTSConfig = MCTSConfig()
    temperature_moves: int = 16
    temperature: float = 1.0
    final_temperature: float = 1e-3
    symmetry_samples: int = 1


def _sample_action(policy: np.ndarray, legal_mask: np.ndarray, temperature: float) -> int:
    legal_indices = np.where(legal_mask)[0]
    if legal_indices.size == 0:
        raise RuntimeError("No legal actions available")
    if temperature <= 1e-6:
        best = legal_indices[np.argmax(policy[legal_indices])]
        return int(best)
    adjusted = np.power(policy[legal_indices], 1.0 / max(temperature, 1e-6))
    adjusted = safe_normalise(adjusted)
    choice = int(np.random.choice(legal_indices, p=adjusted))
    return choice


def _state_value_from_outcome(state: EncodedState, outcome_x: float) -> float:
    return outcome_x if state.to_move_is_x else -outcome_x


def play_game(
    model: Optional[object],
    config: Optional[SelfPlayConfig] = None,
) -> List[ReplaySample]:
    cfg = config or SelfPlayConfig()
    mcts = MCTS(model, cfg.mcts_config)
    game = UltimateTicTacToe()
    current_player: Player = "X"
    history: List[tuple[EncodedState, np.ndarray]] = []
    move_index = 0

    while not game.terminal:
        state = encode_state(game, current_player)
        policy = mcts.run(game, current_player)
        temperature = cfg.temperature if move_index < cfg.temperature_moves else cfg.final_temperature
        action_index = _sample_action(policy, state.legal_actions, temperature)
        history.append((state, policy))
        move = index_to_action(action_index)
        game.make_move(current_player, move)
        current_player = opponent(current_player)
        move_index += 1

    outcome_x = game.terminal_outcome_from_x_perspective()
    samples: List[ReplaySample] = []
    for state, policy in history:
        value = _state_value_from_outcome(state, outcome_x)
        for symmetry in _choose_symmetries(cfg.symmetry_samples):
            augmented_state = augment_state(state, symmetry)
            augmented_policy = apply_symmetry_to_policy(policy, symmetry)
            samples.append(
                ReplaySample(
                    planes=augmented_state.planes,
                    policy=augmented_policy.astype(np.float32),
                    value=value,
                    legal_mask=augmented_state.legal_actions.astype(bool),
                )
            )
    return samples


def _choose_symmetries(count: int) -> Iterable:
    if count <= 1:
        yield SYMMETRIES[0]
        return
    indices = np.random.choice(len(SYMMETRIES), size=count, replace=True)
    for idx in indices:
        yield SYMMETRIES[int(idx)]


__all__ = ["SelfPlayConfig", "play_game"]
