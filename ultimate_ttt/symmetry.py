"""Symmetry utilities for Ultimate Tic-Tac-Toe boards.

This module exposes the eight rotation/reflection symmetries of a 3x3 grid
and helpers for applying them to macro-board indices, local cell indices, and
flattened 9x9 feature planes.  The helpers are intentionally lightweight so
that they can be reused both by the classic reinforcement-learning agent and
by the AlphaZero-style training pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import numpy as np

MacroMapping = Tuple[int, ...]
CellMapping = Tuple[int, ...]
GlobalMapping = Tuple[int, ...]


def _transform_index(index: int, transform: Tuple[int, int, int, int]) -> int:
    """Apply an affine transform in the 3x3 grid space to an index."""

    row, col = divmod(index, 3)
    a, b, c, d = transform
    new_row = a * row + b * col
    new_col = c * row + d * col
    return new_row * 3 + new_col


def _build_macro_mapping(transform: Tuple[int, int, int, int]) -> MacroMapping:
    return tuple(_transform_index(index, transform) for index in range(9))


def _build_global_mapping(macro: MacroMapping, cell: CellMapping) -> GlobalMapping:
    mapping: list[int] = []
    for macro_index in range(9):
        macro_row, macro_col = divmod(macro_index, 3)
        for cell_index in range(9):
            cell_row, cell_col = divmod(cell_index, 3)
            row = macro_row * 3 + cell_row
            col = macro_col * 3 + cell_col
            new_macro = macro[macro_index]
            new_cell = cell[cell_index]
            new_macro_row, new_macro_col = divmod(new_macro, 3)
            new_cell_row, new_cell_col = divmod(new_cell, 3)
            new_row = new_macro_row * 3 + new_cell_row
            new_col = new_macro_col * 3 + new_cell_col
            mapping.append(new_row * 9 + new_col)
    return tuple(mapping)


@dataclass(frozen=True)
class Symmetry:
    """Represents a board symmetry over macro and cell indices."""

    name: str
    macro: MacroMapping
    cell: CellMapping
    global_: GlobalMapping

    def apply_move(self, move: Tuple[int, int]) -> Tuple[int, int]:
        """Transform a move under this symmetry."""

        sub, cell = move
        return self.macro[sub], self.cell[cell]

    def apply_macro_index(self, index: int) -> int:
        return self.macro[index]

    def apply_global_index(self, index: int) -> int:
        return self.global_[index]


# Affine transforms represented as (a, b, c, d) that act on (row, col)
_TRANSFORMS: Tuple[Tuple[int, int, int, int], ...] = (
    (1, 0, 0, 1),   # Identity
    (0, 1, -1, 0),  # Rotate 90°
    (-1, 0, 0, -1),  # Rotate 180°
    (0, -1, 1, 0),  # Rotate 270°
    (1, 0, 0, -1),  # Mirror vertical axis
    (-1, 0, 0, 1),  # Mirror horizontal axis
    (0, 1, 1, 0),   # Main diagonal reflection
    (0, -1, -1, 0),  # Anti-diagonal reflection
)


def _build_symmetries() -> Tuple[Symmetry, ...]:
    symmetries: list[Symmetry] = []
    for transform, name in zip(
        _TRANSFORMS,
        (
            "identity",
            "rot90",
            "rot180",
            "rot270",
            "mirror_v",
            "mirror_h",
            "diag_main",
            "diag_anti",
        ),
    ):
        macro = _build_macro_mapping(transform)
        cell = _build_macro_mapping(transform)
        global_map = _build_global_mapping(macro, cell)
        symmetries.append(Symmetry(name=name, macro=macro, cell=cell, global_=global_map))
    return tuple(symmetries)


SYMMETRIES: Tuple[Symmetry, ...] = _build_symmetries()
MACRO_MAPPINGS: Tuple[MacroMapping, ...] = tuple(sym.macro for sym in SYMMETRIES)
CELL_MAPPINGS: Tuple[CellMapping, ...] = tuple(sym.cell for sym in SYMMETRIES)
GLOBAL_MAPPINGS: Tuple[GlobalMapping, ...] = tuple(sym.global_ for sym in SYMMETRIES)


def apply_symmetry_to_planes(planes: np.ndarray, symmetry: Symmetry) -> np.ndarray:
    """Apply a symmetry to stacked (C, 9, 9) feature planes."""

    if planes.ndim != 3 or planes.shape[1:] != (9, 9):
        raise ValueError("planes must have shape (channels, 9, 9)")
    flattened = planes.reshape(planes.shape[0], 81)
    remapped = flattened[:, symmetry.global_]
    return remapped.reshape(planes.shape)


def apply_symmetry_to_policy(policy: Sequence[float], symmetry: Symmetry) -> np.ndarray:
    """Reorder a length-81 policy vector under the given symmetry."""

    arr = np.asarray(policy, dtype=np.float64)
    if arr.shape[-1] != 81:
        raise ValueError("policy vector must have length 81")
    return arr[..., symmetry.global_]


def invert_mapping(mapping: Sequence[int]) -> Tuple[int, ...]:
    """Return the inverse of a permutation mapping."""

    inverse = [0] * len(mapping)
    for source, target in enumerate(mapping):
        inverse[target] = source
    return tuple(inverse)


def canonical_transform_candidates() -> Iterable[Tuple[Symmetry, MacroMapping]]:
    """Expose candidates for canonicalization routines."""

    return ((sym, sym.macro) for sym in SYMMETRIES)

