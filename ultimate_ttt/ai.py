"""Reinforcement-learning powered opponents for Ultimate Tic-Tac-Toe."""
from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .game import Move, Player, UltimateTicTacToe


def _move_to_key(move: Move) -> str:
    return f"{move[0]}-{move[1]}"

@dataclass
class UltimateTTTRLAI:
    """Simple Q-learning agent for Ultimate Tic-Tac-Toe."""

    alpha: float = 0.4
    gamma: float = 0.95
    default_q: float = 0.0
    q_values: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def _state_key(self, game: UltimateTicTacToe, player: Player) -> str:
        return game.serialize(player)

    def _ensure_state(self, state_key: str, moves: Sequence[Move]) -> Dict[str, float]:
        table = self.q_values.setdefault(state_key, {})
        for move in moves:
            key = _move_to_key(move)
            if key not in table:
                table[key] = self.default_q
        return table

    def choose_action(
        self,
        state_key: str,
        moves: Sequence[Move],
        epsilon: float,
    ) -> Move:
        self._ensure_state(state_key, moves)
        if not moves:
            raise ValueError("No available moves to choose from")
        if random.random() < epsilon:
            return random.choice(list(moves))
        table = self.q_values[state_key]
        best_value = -math.inf
        best_moves: List[Move] = []
        for move in moves:
            value = table.get(_move_to_key(move), self.default_q)
            if value > best_value:
                best_value = value
                best_moves = [move]
            elif value == best_value:
                best_moves.append(move)
        return random.choice(best_moves)

    def best_value(self, state_key: str, moves: Sequence[Move]) -> float:
        if not moves:
            return 0.0
        table = self._ensure_state(state_key, moves)
        return max(table.get(_move_to_key(move), self.default_q) for move in moves)

    def update(
        self,
        state_key: str,
        move: Move,
        reward: float,
        next_state_key: Optional[str],
        next_moves: Sequence[Move],
    ) -> None:
        move_key = _move_to_key(move)
        table = self._ensure_state(state_key, (move,))
        old_value = table.get(move_key, self.default_q)
        if next_state_key is None:
            target = reward
        else:
            opponent_best = self.best_value(next_state_key, next_moves)
            target = reward - self.gamma * opponent_best
        table[move_key] = old_value + self.alpha * (target - old_value)

    def select_move(
        self,
        game: UltimateTicTacToe,
        player: Player,
        epsilon: float = 0.0,
    ) -> Move:
        state_key = self._state_key(game, player)
        moves = game.available_moves()
        return self.choose_action(state_key, moves, epsilon)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.q_values, fh)

    @classmethod
    def load(cls, path: str) -> "UltimateTTTRLAI":
        agent = cls()
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            agent.q_values = {
                state: {move: float(value) for move, value in table.items()}
                for state, table in data.items()
            }
        return agent

    def to_serialisable(self) -> Dict[str, Dict[str, float]]:
        return self.q_values

    def load_from_dict(self, data: Dict[str, Dict[str, float]]) -> None:
        self.q_values = {
            state: {move: float(value) for move, value in table.items()}
            for state, table in data.items()
        }


def immediate_winning_move(game: UltimateTicTacToe, player: Player) -> Optional[Move]:
    for move in game.available_moves():
        test_game = game.clone()
        test_game.make_move(player, move)
        if test_game.winner == player:
            return move
    return None


def block_opponent_move(game: UltimateTicTacToe, player: Player) -> Optional[Move]:
    opponent = "O" if player == "X" else "X"
    for move in game.available_moves():
        test_game = game.clone()
        test_game.make_move(opponent, move)
        if test_game.winner == opponent:
            return move
    return None
