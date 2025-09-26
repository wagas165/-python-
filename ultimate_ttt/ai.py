"""Reinforcement-learning powered opponents for Ultimate Tic-Tac-Toe."""
from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .game import Move, Player, UltimateTicTacToe, canonicalize_state


def _move_to_key(move: Move) -> str:
    return f"{move[0]}-{move[1]}"


def _parse_serialized_state(
    state_key: str,
) -> Optional[Tuple[Player, List[List[str]], List[str], Optional[int]]]:
    try:
        player_part, remainder = state_key.split(":", 1)
        boards_part, macro_part, forced_part = remainder.split("#")
    except ValueError:
        return None

    boards_strings = boards_part.split("|")
    if len(boards_strings) != 9 or len(macro_part) != 9:
        return None
    if not all(len(board) == 9 for board in boards_strings):
        return None

    boards = [list(board) for board in boards_strings]
    macro_board = list(macro_part)

    forced_board: Optional[int]
    if forced_part == "*":
        forced_board = None
    else:
        try:
            forced_board = int(forced_part)
        except ValueError:
            return None

    return player_part, boards, macro_board, forced_board


def _canonicalise_q_tables(
    q_values: Dict[str, Dict[str, float]]
) -> Dict[str, Dict[str, float]]:
    identity_mapping: Tuple[int, ...] = tuple(range(9))
    state_sums: Dict[str, Dict[str, float]] = {}
    state_counts: Dict[str, Dict[str, int]] = {}

    for state_key, table in q_values.items():
        parsed = _parse_serialized_state(state_key)
        if parsed is None:
            canonical_key = state_key
            mapping = identity_mapping
        else:
            player, boards, macro, forced = parsed
            canonical_key, mapping = canonicalize_state(player, boards, macro, forced)

        sum_table = state_sums.setdefault(canonical_key, {})
        count_table = state_counts.setdefault(canonical_key, {})

        for move_key, value in table.items():
            try:
                sub_part, cell_part = move_key.split("-", 1)
                sub_index = int(sub_part)
                cell_index = int(cell_part)
                transformed_key = f"{mapping[sub_index]}-{mapping[cell_index]}"
            except (ValueError, IndexError):
                transformed_key = move_key

            val = float(value)
            sum_table[transformed_key] = sum_table.get(transformed_key, 0.0) + val
            count_table[transformed_key] = count_table.get(transformed_key, 0) + 1

    canonical: Dict[str, Dict[str, float]] = {}
    for state_key, move_sums in state_sums.items():
        counts = state_counts[state_key]
        move_entries = {
            move_key: move_sums[move_key] / counts[move_key]
            for move_key in move_sums
        }
        canonical[state_key] = move_entries

    return canonical

@dataclass
class UltimateTTTRLAI:
    """Simple Q-learning agent for Ultimate Tic-Tac-Toe."""

    alpha: float = 0.4
    gamma: float = 0.95
    default_q: float = 0.0
    q_values: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def _state_key(self, game: UltimateTicTacToe, player: Player) -> str:
        return game.serialize_canonical(player)

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
        canonical_q_values = _canonicalise_q_tables(self.q_values)
        self.q_values = canonical_q_values

        data = {
            "algorithm": self.__class__.__name__,
            "format_version": 2,
            "canonical_keys": True,
            "q_values": canonical_q_values,
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh)

    @classmethod
    def load(cls, path: str) -> "UltimateTTTRLAI":
        agent = cls()
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)

            if isinstance(data, dict) and "algorithm" in data:
                raw_tables = data.get("q_values", {})
            elif isinstance(data, dict):
                raw_tables = data
            else:
                raw_tables = {}

            agent.q_values = _canonicalise_q_tables(raw_tables)
        return agent

    def to_serialisable(self) -> Dict[str, Dict[str, float]]:
        return self.q_values

    def load_from_dict(self, data: Dict[str, Dict[str, float]]) -> None:
        self.q_values = _canonicalise_q_tables(data)


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
