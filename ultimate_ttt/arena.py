"""Evaluation arena comparing two agents via self-play matches."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .game import UltimateTicTacToe, index_to_action
from .mcts import MCTS
from .model import PolicyValueNet

__all__ = ["Arena", "ArenaResult"]


@dataclass
class ArenaResult:
    wins: int
    losses: int
    draws: int

    @property
    def total(self) -> int:
        return self.wins + self.losses + self.draws

    @property
    def win_rate(self) -> float:
        return self.wins / self.total if self.total else 0.0


@dataclass
class Arena:
    challenger: PolicyValueNet
    baseline: PolicyValueNet
    mcts_simulations: int = 200
    c_puct: float = 2.0
    rng: Optional[np.random.Generator] = None

    def _create_mcts(self, model: PolicyValueNet) -> MCTS:
        return MCTS(
            model=model,
            num_simulations=self.mcts_simulations,
            c_puct=self.c_puct,
            dirichlet_epsilon=0.0,
            dirichlet_alpha=0.3,
            rng=self.rng,
        )

    def play_matches(self, num_games: int = 200) -> ArenaResult:
        results = ArenaResult(wins=0, losses=0, draws=0)

        for game_index in range(num_games):
            game = UltimateTicTacToe()
            challenger_first = (game_index % 2 == 0)
            players = {
                "X": self.challenger if challenger_first else self.baseline,
                "O": self.baseline if challenger_first else self.challenger,
            }
            mcts_cache = {role: self._create_mcts(model) for role, model in players.items()}

            while not game.terminal:
                player_to_move = game.active_player()
                mcts = mcts_cache[player_to_move]
                _, root = mcts.run(game, add_noise=False)
                visit_counts = root.visit_counts.astype(np.float32)
                legal = game.legal_action_mask()
                legal_indices = np.flatnonzero(legal)
                if legal_indices.size == 0:
                    break
                best_action = int(legal_indices[int(np.argmax(visit_counts[legal_indices]))])
                game.make_move(player_to_move, index_to_action(best_action))

            outcome = game.terminal_outcome_from_x_perspective()
            if outcome > 0:
                if challenger_first:
                    results.wins += 1
                else:
                    results.losses += 1
            elif outcome < 0:
                if challenger_first:
                    results.losses += 1
                else:
                    results.wins += 1
            else:
                results.draws += 1

        return results
