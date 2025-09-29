"""Feature construction utilities for the AlphaZero-style agent."""
from __future__ import annotations

from typing import Tuple

import numpy as np

from .game import UltimateTicTacToe

__all__ = ["encode_state", "FEATURE_CHANNELS"]

FEATURE_CHANNELS = 8


def _global_coords(sub_index: int, cell_index: int) -> Tuple[int, int]:
    macro_row, macro_col = divmod(sub_index, 3)
    cell_row, cell_col = divmod(cell_index, 3)
    return macro_row * 3 + cell_row, macro_col * 3 + cell_col


def _mark_sub_board(mask: np.ndarray, sub_index: int, value: float) -> None:
    row_offset = (sub_index // 3) * 3
    col_offset = (sub_index % 3) * 3
    mask[row_offset : row_offset + 3, col_offset : col_offset + 3] = value


def encode_state(game: UltimateTicTacToe) -> Tuple[np.ndarray, np.ndarray]:
    """Return (planes, legal_mask) for the provided game state."""

    planes = np.zeros((FEATURE_CHANNELS, 9, 9), dtype=np.float32)
    legal_mask = game.legal_action_mask()

    # Stones for X and O
    for sub_index, board in enumerate(game.boards):
        for cell_index, value in enumerate(board):
            if value == " ":
                continue
            row, col = _global_coords(sub_index, cell_index)
            if value == "X":
                planes[0, row, col] = 1.0
            elif value == "O":
                planes[1, row, col] = 1.0

    # Player to move (single plane; 1 if X to move, else 0)
    to_move = game.active_player()
    if to_move == "X":
        planes[2, :, :] = 1.0

    # Legal moves mask (per-cell)
    for action_index, allowed in enumerate(legal_mask):
        if allowed:
            sub_index, cell_index = divmod(action_index, 9)
            row, col = _global_coords(sub_index, cell_index)
            planes[3, row, col] = 1.0

    # Forced sub-board mask
    forced = game.forced_board_index()
    if forced is not None:
        _mark_sub_board(planes[4], forced, 1.0)
    else:
        for sub_index, macro_status in enumerate(game.macro_board):
            if macro_status == " ":
                _mark_sub_board(planes[4], sub_index, 1.0)

    # Macro board wins (upsampled to 9x9)
    for sub_index, macro_status in enumerate(game.macro_board):
        if macro_status == "X":
            _mark_sub_board(planes[5], sub_index, 1.0)
        elif macro_status == "O":
            _mark_sub_board(planes[6], sub_index, 1.0)

    # Last move one-hot
    if game.last_move is not None:
        row, col = _global_coords(*game.last_move)
        planes[7, row, col] = 1.0

    return planes, legal_mask.astype(bool)
