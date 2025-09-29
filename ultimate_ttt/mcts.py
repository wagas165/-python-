"""Monte Carlo Tree Search with PUCT for Ultimate Tic-Tac-Toe."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .features import encode_state
from .game import UltimateTicTacToe, index_to_action
from .model import PolicyValueNet

__all__ = ["MCTS", "Node"]


@dataclass
class Node:
    state: UltimateTicTacToe
    prior: np.ndarray = field(init=False)
    visit_counts: np.ndarray = field(init=False)
    total_value: np.ndarray = field(init=False)
    children: Dict[int, "Node"] = field(default_factory=dict)
    expanded: bool = False

    def __post_init__(self) -> None:
        self.prior = np.zeros(81, dtype=np.float32)
        self.visit_counts = np.zeros(81, dtype=np.int32)
        self.total_value = np.zeros(81, dtype=np.float32)

    @property
    def to_move(self) -> str:
        return self.state.active_player()

    @property
    def legal_mask(self) -> np.ndarray:
        return self.state.legal_action_mask()

    def legal_indices(self) -> np.ndarray:
        return np.flatnonzero(self.legal_mask)

    def select(self, c_puct: float) -> int:
        legal = self.legal_indices()
        if legal.size == 0:
            raise RuntimeError("select called on terminal node")
        total = self.visit_counts[legal].sum()
        sqrt_total = math.sqrt(total + 1e-8)

        best_score = -float("inf")
        best_action = int(legal[0])
        for action in legal:
            action = int(action)
            visits = self.visit_counts[action]
            q_value = self.total_value[action] / visits if visits > 0 else 0.0
            u_value = c_puct * self.prior[action] * sqrt_total / (1 + visits)
            score = q_value + u_value
            if score > best_score:
                best_score = score
                best_action = action
        return best_action

    def add_child(self, action: int, child: "Node") -> None:
        self.children[action] = child


class MCTS:
    def __init__(
        self,
        model: PolicyValueNet,
        num_simulations: int = 400,
        c_puct: float = 2.0,
        dirichlet_epsilon: float = 0.25,
        dirichlet_alpha: float = 0.3,
        device: Optional[torch.device] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_epsilon = dirichlet_epsilon
        self.dirichlet_alpha = dirichlet_alpha
        self.device = device
        self.rng = rng or np.random.default_rng()

    def run(self, root_state: UltimateTicTacToe, add_noise: bool = True) -> Tuple[np.ndarray, Node]:
        root = Node(root_state.clone())
        if root_state.terminal:
            return np.zeros(81, dtype=np.float32), root
        self._evaluate(root)
        if add_noise:
            self._apply_dirichlet_noise(root)

        for _ in range(self.num_simulations):
            node = root
            scratch = root_state.clone()
            path: List[Tuple[Node, int]] = []

            while node.expanded and not scratch.terminal:
                action = node.select(self.c_puct)
                path.append((node, action))
                move = index_to_action(action)
                scratch.make_move(scratch.active_player(), move)
                child = node.children.get(action)
                if child is None:
                    child = Node(scratch.clone())
                    node.add_child(action, child)
                node = child

            if scratch.terminal:
                value_x = scratch.terminal_outcome_from_x_perspective()
                self._backup(path, value_x)
                continue

            _, value_x = self._evaluate(node)
            self._backup(path, value_x)

        pi = root.visit_counts.astype(np.float32)
        total = pi.sum()
        if total > 0:
            pi /= total
        return pi, root

    def _evaluate(self, node: Node) -> Tuple[np.ndarray, float]:
        planes, legal = encode_state(node.state)
        probs, value_x = self.model.inference(planes, legal, device=self.device)
        legal_indices = np.flatnonzero(legal)
        node.prior[:] = 0.0
        node.prior[legal_indices] = probs[legal_indices]
        prob_sum = node.prior.sum()
        if prob_sum > 0:
            node.prior /= prob_sum
        node.expanded = True
        return node.prior, value_x

    def _apply_dirichlet_noise(self, node: Node) -> None:
        legal = node.legal_indices()
        if legal.size == 0:
            return
        noise = self.rng.dirichlet([self.dirichlet_alpha] * legal.size)
        node.prior[legal] = (1 - self.dirichlet_epsilon) * node.prior[legal] + self.dirichlet_epsilon * noise
        node.prior[legal] /= node.prior[legal].sum()

    def _backup(self, path: Sequence[Tuple[Node, int]], value_x: float) -> None:
        for node, action in reversed(path):
            perspective = value_x if node.to_move == "X" else -value_x
            node.visit_counts[action] += 1
            node.total_value[action] += perspective
