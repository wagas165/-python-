"""Evaluation arena for comparing two agents."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .game import UltimateTicTacToe, index_to_action
from .mcts import MCTS, MCTSConfig
from .utils import opponent


@dataclass
class ArenaResult:
    wins_a: int = 0
    wins_b: int = 0
    draws: int = 0

    @property
    def total(self) -> int:
        return self.wins_a + self.wins_b + self.draws

    def win_rate_a(self) -> float:
        return self.wins_a / self.total if self.total else 0.0


def play_match(
    model_a: Optional[object],
    model_b: Optional[object],
    games: int = 20,
    mcts_config: Optional[MCTSConfig] = None,
) -> ArenaResult:
    config = mcts_config or MCTSConfig()
    result = ArenaResult()

    for game_index in range(games):
        game = UltimateTicTacToe()
        first_is_a = game_index % 2 == 0
        players = {
            "X": model_a if first_is_a else model_b,
            "O": model_b if first_is_a else model_a,
        }
        mcts_instances = {
            "X": MCTS(players["X"], config),
            "O": MCTS(players["O"], config),
        }
        current_player = "X"
        while not game.terminal:
            mcts = mcts_instances[current_player]
            policy = mcts.run(game, current_player)
            legal = game.legal_action_mask()
            legal_indices = np.where(legal)[0]
            if legal_indices.size == 0:
                break
            best_index = legal_indices[np.argmax(policy[legal_indices])]
            move = index_to_action(int(best_index))
            game.make_move(current_player, move)
            current_player = opponent(current_player)

        winner = game.winner
        if winner is None:
            result.draws += 1
        elif (winner == "X" and first_is_a) or (winner == "O" and not first_is_a):
            result.wins_a += 1
        else:
            result.wins_b += 1

    return result


__all__ = ["ArenaResult", "play_match"]
