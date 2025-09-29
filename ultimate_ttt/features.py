"""Feature encoding utilities for Ultimate Tic-Tac-Toe."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .game import UltimateTicTacToe, action_to_index
from .symmetry import SYMMETRIES, Symmetry, apply_symmetry_to_planes, apply_symmetry_to_policy


@dataclass(frozen=True)
class EncodedState:
    """Bundle of feature planes and auxiliary metadata for a position."""

    planes: np.ndarray  # (C, 9, 9)
    legal_actions: np.ndarray  # (81,) boolean mask
    to_move_is_x: bool
    forced_board: Optional[int]
    last_move: Optional[int]


def _board_to_planes(game: UltimateTicTacToe) -> tuple[np.ndarray, np.ndarray]:
    x_plane = np.zeros((9, 9), dtype=np.float32)
    o_plane = np.zeros((9, 9), dtype=np.float32)
    for macro_index, board in enumerate(game.boards):
        macro_row, macro_col = divmod(macro_index, 3)
        base_row = macro_row * 3
        base_col = macro_col * 3
        for cell_index, value in enumerate(board):
            row_offset, col_offset = divmod(cell_index, 3)
            row = base_row + row_offset
            col = base_col + col_offset
            if value == "X":
                x_plane[row, col] = 1.0
            elif value == "O":
                o_plane[row, col] = 1.0
    return x_plane, o_plane


def _macro_to_plane(game: UltimateTicTacToe, player: str) -> np.ndarray:
    plane = np.zeros((9, 9), dtype=np.float32)
    for macro_index, status in enumerate(game.macro_board):
        if status != player:
            continue
        macro_row, macro_col = divmod(macro_index, 3)
        base_row = macro_row * 3
        base_col = macro_col * 3
        plane[base_row : base_row + 3, base_col : base_col + 3] = 1.0
    return plane


def encode_state(game: UltimateTicTacToe, current_player: str) -> EncodedState:
    """Encode the game state into stacked feature planes."""

    x_plane, o_plane = _board_to_planes(game)
    to_move_is_x = current_player == "X"
    to_move_plane = np.full((9, 9), 1.0 if to_move_is_x else 0.0, dtype=np.float32)
    legal_mask = game.legal_action_mask().astype(bool)

    legal_plane = legal_mask.reshape(9, 9).astype(np.float32)
    forced_index = game.forced_board_index()
    forced_plane = np.zeros((9, 9), dtype=np.float32)
    if forced_index is not None:
        macro_row, macro_col = divmod(forced_index, 3)
        forced_plane[macro_row * 3 : macro_row * 3 + 3, macro_col * 3 : macro_col * 3 + 3] = 1.0

    macro_x_plane = _macro_to_plane(game, "X")
    macro_o_plane = _macro_to_plane(game, "O")

    last_move_index: Optional[int] = None
    last_move_plane = np.zeros((9, 9), dtype=np.float32)
    if game.last_move is not None:
        last_move_index = action_to_index(game.last_move)
        row, col = divmod(last_move_index, 9)
        last_move_plane[row, col] = 1.0

    planes = np.stack(
        (
            x_plane,
            o_plane,
            to_move_plane,
            legal_plane,
            forced_plane,
            macro_x_plane,
            macro_o_plane,
            last_move_plane,
        ),
        axis=0,
    )

    return EncodedState(
        planes=planes,
        legal_actions=legal_mask,
        to_move_is_x=to_move_is_x,
        forced_board=forced_index,
        last_move=last_move_index,
    )


def augment_state(state: EncodedState, symmetry: Symmetry) -> EncodedState:
    """Apply a symmetry transform to the encoded state."""

    transformed_planes = apply_symmetry_to_planes(state.planes, symmetry)
    transformed_legal = apply_symmetry_to_policy(state.legal_actions, symmetry).astype(bool)
    forced = None if state.forced_board is None else symmetry.macro[state.forced_board]
    last_move = None if state.last_move is None else symmetry.apply_global_index(state.last_move)
    return EncodedState(
        planes=transformed_planes.astype(np.float32),
        legal_actions=transformed_legal,
        to_move_is_x=state.to_move_is_x,
        forced_board=forced,
        last_move=last_move,
    )


__all__ = [
    "EncodedState",
    "SYMMETRIES",
    "encode_state",
    "augment_state",
]
