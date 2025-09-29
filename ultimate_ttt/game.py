"""Core game logic for the Ultimate Tic-Tac-Toe variant used in the GUI and AI.

The implementation keeps the representation independent from the UI and the
reinforcement learning agent.  The board is organised as nine 3x3 sub-boards.
Each move is represented as a tuple ``(sub_board_index, cell_index)`` where both
values are in the range ``0..8``.  The ``sub_board_index`` selects one of the
nine local boards and the ``cell_index`` selects a cell inside the local board
using row-major order.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np

Player = str  # Either "X" or "O"
Move = Tuple[int, int]

WIN_LINES: Tuple[Tuple[int, int, int], ...] = (
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),
    (0, 4, 8),
    (2, 4, 6),
)


def _build_transformations() -> Tuple[Tuple[int, ...], ...]:
    """Return the eight symmetries of a 3x3 grid as index mappings."""

    def transform(func: Callable[[int, int], Tuple[int, int]]) -> Tuple[int, ...]:
        mapping: List[int] = []
        for index in range(9):
            row, col = divmod(index, 3)
            new_row, new_col = func(row, col)
            mapping.append(new_row * 3 + new_col)
        return tuple(mapping)

    operations = (
        lambda r, c: (r, c),
        lambda r, c: (c, 2 - r),
        lambda r, c: (2 - r, 2 - c),
        lambda r, c: (2 - c, r),
        lambda r, c: (r, 2 - c),
        lambda r, c: (2 - r, c),
        lambda r, c: (c, r),
        lambda r, c: (2 - c, 2 - r),
    )

    return tuple(transform(op) for op in operations)


_TRANSFORMATIONS: Tuple[Tuple[int, ...], ...] = _build_transformations()


class InvalidMoveError(RuntimeError):
    """Raised when a move is attempted that is not legal in the current state."""


@dataclass
class UltimateTicTacToe:
    """Representation of a single Ultimate Tic-Tac-Toe game session."""

    boards: List[List[str]] = field(
        default_factory=lambda: [[" "] * 9 for _ in range(9)]
    )
    macro_board: List[str] = field(default_factory=lambda: [" "] * 9)
    last_move: Optional[Move] = None
    winner: Optional[Player] = None
    terminal: bool = False

    def clone(self) -> "UltimateTicTacToe":
        copy = UltimateTicTacToe()
        copy.boards = [row[:] for row in self.boards]
        copy.macro_board = self.macro_board[:]
        copy.last_move = None if self.last_move is None else tuple(self.last_move)
        copy.winner = self.winner
        copy.terminal = self.terminal
        return copy

    @property
    def is_draw(self) -> bool:
        return self.terminal and self.winner is None

    def reset(self) -> None:
        self.boards = [[" "] * 9 for _ in range(9)]
        self.macro_board = [" "] * 9
        self.last_move = None
        self.winner = None
        self.terminal = False

    def available_moves(self) -> List[Move]:
        if self.terminal:
            return []

        forced_board = self._forced_board_index()
        moves: List[Move] = []
        if forced_board is not None:
            for cell_idx, value in enumerate(self.boards[forced_board]):
                if value == " ":
                    moves.append((forced_board, cell_idx))
            return moves

        for sub_idx in range(9):
            if self.macro_board[sub_idx] != " ":
                continue
            for cell_idx, value in enumerate(self.boards[sub_idx]):
                if value == " ":
                    moves.append((sub_idx, cell_idx))
        return moves

    def forced_board_index(self) -> Optional[int]:
        """Public accessor exposing the index of the forced sub-board, if any."""

        return self._forced_board_index()

    def _forced_board_index(self) -> Optional[int]:
        if self.last_move is None:
            return None
        forced = self.last_move[1]
        if self.macro_board[forced] != " ":
            return None
        if any(cell == " " for cell in self.boards[forced]):
            return forced
        return None

    def make_move(self, player: Player, move: Move) -> None:
        if player not in ("X", "O"):
            raise ValueError("player must be 'X' or 'O'")
        if self.terminal:
            raise InvalidMoveError("Game has already finished")

        allowed = self.available_moves()
        if move not in allowed:
            raise InvalidMoveError("Move is not legal in the current state")

        board_idx, cell_idx = move
        self.boards[board_idx][cell_idx] = player
        self.last_move = move

        sub_status = self._update_sub_board(board_idx)
        if sub_status == player:
            self.macro_board[board_idx] = player
        elif sub_status == "T":
            self.macro_board[board_idx] = "T"

        self._update_macro_board()

    def legal_action_mask(self) -> np.ndarray:
        """Return a boolean mask over the 81 global actions that are legal."""

        mask = np.zeros(81, dtype=bool)
        for sub_idx, cell_idx in self.available_moves():
            mask[sub_idx * 9 + cell_idx] = True
        return mask

    def terminal_outcome_from_x_perspective(self) -> float:
        """Return +1/-1/0 from X's perspective for terminal positions."""

        if not self.terminal:
            raise ValueError("terminal_outcome_from_x_perspective called before game end")
        if self.winner == "X":
            return 1.0
        if self.winner == "O":
            return -1.0
        return 0.0

    def _update_sub_board(self, board_index: int) -> str:
        board = self.boards[board_index]
        for line in WIN_LINES:
            a, b, c = line
            if board[a] != " " and board[a] == board[b] == board[c]:
                return board[a]
        if all(cell != " " for cell in board):
            return "T"
        return " "

    def _update_macro_board(self) -> None:
        for line in WIN_LINES:
            a, b, c = line
            if (
                self.macro_board[a] in ("X", "O")
                and self.macro_board[a] == self.macro_board[b] == self.macro_board[c]
            ):
                self.winner = self.macro_board[a]
                self.terminal = True
                return

        if all(status in ("X", "O", "T") for status in self.macro_board):
            self.winner = None
            self.terminal = True

    def serialize(self, current_player: Player) -> str:
        forced = self._forced_board_index()
        return _serialize_components(current_player, self.boards, self.macro_board, forced)

    def serialize_canonical(self, current_player: Player) -> str:
        """Return the canonical serialization while respecting forced boards."""
        forced = self._forced_board_index()
        canonical, _ = canonicalize_state(
            current_player, self.boards, self.macro_board, forced
        )
        return canonical

    def render_ascii(self) -> str:
        def cell_value(board: Sequence[str], idx: int) -> str:
            value = board[idx]
            return value if value != " " else "."

        rows: List[str] = []
        for big_row in range(3):
            for inner_row in range(3):
                row_cells: List[str] = []
                for big_col in range(3):
                    board_index = big_row * 3 + big_col
                    board = self.boards[board_index]
                    start = inner_row * 3
                    row_cells.append(
                        " ".join(
                            cell_value(board, start + offset) for offset in range(3)
                        )
                    )
                rows.append(" || ".join(row_cells))
            if big_row < 2:
                rows.append("==++==++==")
        return "\n".join(rows)

    def highlight_boards(self) -> Sequence[int]:
        forced = self._forced_board_index()
        if forced is not None:
            return (forced,)
        return tuple(idx for idx, status in enumerate(self.macro_board) if status == " ")

    def macro_owner(self, index: int) -> str:
        return self.macro_board[index]

    def cells(self, index: int) -> Sequence[str]:
        return self.boards[index]

    def active_player(self) -> Player:
        x_count = sum(cell == "X" for board in self.boards for cell in board)
        o_count = sum(cell == "O" for board in self.boards for cell in board)
        return "X" if x_count == o_count else "O"


def moves_to_indices(moves: Iterable[Move]) -> Sequence[int]:
    return tuple(action_to_index(move) for move in moves)


def action_to_index(move: Move) -> int:
    sub_idx, cell_idx = move
    if not 0 <= sub_idx < 9 or not 0 <= cell_idx < 9:
        raise ValueError("move components must be in range 0..8")
    return sub_idx * 9 + cell_idx


def index_to_action(index: int) -> Move:
    if not 0 <= index < 81:
        raise ValueError("action index must be in range 0..80")
    return divmod(index, 9)


def apply_mapping_to_move(move: Move, mapping: Sequence[int]) -> Move:
    """Return the move transformed by the given symmetry mapping."""

    return mapping[move[0]], mapping[move[1]]


def _serialize_components(
    current_player: Player,
    boards: Sequence[Sequence[str]],
    macro_board: Sequence[str],
    forced_board: Optional[int],
) -> str:
    boards_repr = "|".join("".join(board) for board in boards)
    macro_repr = "".join(macro_board)
    forced_repr = "*" if forced_board is None else str(forced_board)
    return f"{current_player}:{boards_repr}#{macro_repr}#{forced_repr}"


def canonicalize_state(
    current_player: Player,
    boards: Sequence[Sequence[str]],
    macro_board: Sequence[str],
    forced_board: Optional[int],
) -> Tuple[str, Tuple[int, ...]]:
    """Canonicalize a state under all symmetries, aligning forced sub-boards."""
    best_serialized: Optional[str] = None
    best_mapping: Tuple[int, ...] = _TRANSFORMATIONS[0]

    for mapping in _TRANSFORMATIONS:
        transformed_boards: List[List[str]] = [[" "] * 9 for _ in range(9)]
        transformed_macro: List[str] = [" "] * 9

        for old_macro_index, board in enumerate(boards):
            new_macro_index = mapping[old_macro_index]
            transformed_macro[new_macro_index] = macro_board[old_macro_index]

            new_board = [" "] * 9
            for old_cell_index, value in enumerate(board):
                new_cell_index = mapping[old_cell_index]
                new_board[new_cell_index] = value
            transformed_boards[new_macro_index] = new_board

        new_forced = None if forced_board is None else mapping[forced_board]
        serialized = _serialize_components(
            current_player, transformed_boards, transformed_macro, new_forced
        )

        if best_serialized is None or serialized < best_serialized:
            best_serialized = serialized
            best_mapping = mapping

    assert best_serialized is not None
    return best_serialized, best_mapping
