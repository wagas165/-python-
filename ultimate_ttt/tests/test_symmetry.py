from __future__ import annotations

import numpy as np

from ultimate_ttt.game import UltimateTicTacToe
from ultimate_ttt.symmetry import SYMMETRIES, apply_symmetry_to_planes, apply_symmetry_to_policy


def test_symmetry_round_trip() -> None:
    game = UltimateTicTacToe()
    moves = [(0, 0), (4, 4), (8, 8), (3, 5)]
    players = ["X", "O", "X", "O"]
    for player, move in zip(players, moves):
        game.make_move(player, move)
    planes = np.random.rand(4, 9, 9).astype(np.float32)
    policy = np.linspace(0, 1, 81, dtype=np.float64)
    for symmetry in SYMMETRIES:
        transformed_planes = apply_symmetry_to_planes(planes, symmetry)
        restored_planes = apply_symmetry_to_planes(transformed_planes, symmetry)
        assert np.allclose(planes, restored_planes)
        transformed_policy = apply_symmetry_to_policy(policy, symmetry)
        restored_policy = apply_symmetry_to_policy(transformed_policy, symmetry)
        assert np.allclose(policy, restored_policy)
