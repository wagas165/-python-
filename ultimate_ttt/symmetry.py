"""Utilities for handling board symmetries in Ultimate Tic-Tac-Toe.

The module provides index mappings for the eight rotational and reflection
symmetries of the 3x3 macro board while keeping the forced-sub-board
constraint aligned with the corresponding micro boards.  The mappings operate
both on macro indices (0..8) and on the flattened 9x9 grid (0..80).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np

MacroMapping = Tuple[int, ...]
GlobalMapping = Tuple[int, ...]


def _build_macro_mappings() -> Tuple[MacroMapping, ...]:
    """Return the eight symmetry mappings for indices in a 3x3 grid."""

    def transform(func: Callable[[int, int], Tuple[int, int]]) -> MacroMapping:
        mapping = []
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


MACRO_MAPPINGS: Tuple[MacroMapping, ...] = _build_macro_mappings()


def _build_global_mappings() -> Tuple[GlobalMapping, ...]:
    """Create symmetry mappings for the flattened 9x9 grid (0..80)."""

    global_mappings: list[GlobalMapping] = []
    for macro_map in MACRO_MAPPINGS:
        mapping: list[int] = []
        for index in range(81):
            macro_index, cell_index = divmod(index, 9)
            new_macro = macro_map[macro_index]
            new_cell = macro_map[cell_index]
            mapping.append(new_macro * 9 + new_cell)
        global_mappings.append(tuple(mapping))
    return tuple(global_mappings)


GLOBAL_MAPPINGS: Tuple[GlobalMapping, ...] = _build_global_mappings()


def apply_to_macro(index: int, mapping: MacroMapping) -> int:
    """Apply a macro-level symmetry mapping to ``index``."""

    if not 0 <= index < 9:
        raise ValueError("macro index must be in range 0..8")
    return mapping[index]


def apply_to_global(index: int, mapping: GlobalMapping) -> int:
    """Apply a symmetry mapping to an index in the flattened 9x9 grid."""

    if not 0 <= index < 81:
        raise ValueError("global index must be in range 0..80")
    return mapping[index]


def apply_to_policy(policy: np.ndarray, mapping: GlobalMapping) -> np.ndarray:
    """Return a copy of ``policy`` after applying the symmetry mapping."""

    flat = np.asarray(policy, dtype=np.float32).reshape(-1)
    if flat.size != 81:
        raise ValueError("policy must have exactly 81 entries")
    transformed = flat[np.array(mapping, dtype=np.intp)]
    return transformed


def apply_to_planes(planes: np.ndarray, mapping: GlobalMapping) -> np.ndarray:
    """Apply the symmetry mapping to an array of shape ``(C, 9, 9)``."""

    array = np.asarray(planes)
    if array.ndim != 3 or array.shape[1:] != (9, 9):
        raise ValueError("planes must have shape (C, 9, 9)")
    flat = array.reshape(array.shape[0], 81)
    transformed = flat[:, np.array(mapping, dtype=np.intp)]
    return transformed.reshape(array.shape)


@dataclass(frozen=True)
class Symmetry:
    """Convenience wrapper bundling macro and global mappings."""

    macro: MacroMapping
    global_: GlobalMapping

    def apply_macro(self, index: int) -> int:
        return apply_to_macro(index, self.macro)

    def apply_global(self, index: int) -> int:
        return apply_to_global(index, self.global_)

    def apply_planes(self, planes: np.ndarray) -> np.ndarray:
        return apply_to_planes(planes, self.global_)

    def apply_policy(self, policy: np.ndarray) -> np.ndarray:
        return apply_to_policy(policy, self.global_)


SYMMETRIES: Tuple[Symmetry, ...] = tuple(
    Symmetry(macro=mapping, global_=GLOBAL_MAPPINGS[idx])
    for idx, mapping in enumerate(MACRO_MAPPINGS)
)
