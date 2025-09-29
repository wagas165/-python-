"""Monte Carlo Tree Search with PUCT exploration."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .features import EncodedState, encode_state
from .game import Player, UltimateTicTacToe, index_to_action
from .utils import dirichlet_noise, masked_softmax, opponent, safe_normalise


@dataclass
class MCTSConfig:
    num_simulations: int = 200
    c_puct: float = 2.0
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25


class Node:
    """A search node storing visit statistics for outgoing actions."""

    def __init__(self, state: EncodedState, to_move: Player) -> None:
        self.state = state
        self.to_move = to_move
        self.prior = np.zeros(81, dtype=np.float64)
        self.visit_counts = np.zeros(81, dtype=np.int32)
        self.value_sums = np.zeros(81, dtype=np.float64)
        self.children: Dict[int, Node] = {}
        self.legal_mask = state.legal_actions.astype(bool)
        self.total_visits = 0
        self.is_expanded = False

    def set_prior(self, policy: np.ndarray) -> None:
        clipped = np.zeros_like(self.prior)
        clipped[self.legal_mask] = policy[self.legal_mask]
        if clipped[self.legal_mask].sum() <= 0:
            count = int(self.legal_mask.sum())
            if count > 0:
                clipped[self.legal_mask] = 1.0 / count
        else:
            clipped = safe_normalise(clipped)
        self.prior = clipped
        self.is_expanded = True

    def select_action(self, c_puct: float) -> int:
        legal_indices = np.where(self.legal_mask)[0]
        if legal_indices.size == 0:
            raise RuntimeError("Node has no legal moves")
        sqrt_total = math.sqrt(self.total_visits + 1e-8)
        best_score = -np.inf
        best_action = legal_indices[0]
        for action in legal_indices:
            visits = self.visit_counts[action]
            q_value = 0.0 if visits == 0 else self.value_sums[action] / visits
            u_value = c_puct * self.prior[action] * sqrt_total / (1 + visits)
            score = q_value + u_value
            if score > best_score:
                best_score = score
                best_action = action
        return int(best_action)

    def child(self, action: int) -> Optional["Node"]:
        return self.children.get(action)

    def add_child(self, action: int, child: "Node") -> None:
        self.children[action] = child


class MCTS:
    """PUCT-based Monte Carlo Tree Search driven by a policy-value model."""

    def __init__(
        self,
        model: Optional[object],
        config: Optional[MCTSConfig] = None,
        device: Optional[object] = None,
    ) -> None:
        self.model = model
        self.config = config or MCTSConfig()
        self.device = device

    def run(self, game: UltimateTicTacToe, current_player: Player) -> np.ndarray:
        root_state = encode_state(game, current_player)
        root = Node(root_state, current_player)
        policy, _ = self._evaluate(root_state)
        root.set_prior(policy)
        self._apply_dirichlet_noise(root)

        for _ in range(self.config.num_simulations):
            self._simulate(game, root, current_player)

        counts = root.visit_counts.astype(np.float64)
        return safe_normalise(counts)

    def _simulate(
        self,
        game: UltimateTicTacToe,
        root: Node,
        root_player: Player,
    ) -> None:
        scratch = game.clone()
        node = root
        player = root_player
        path: List[Tuple[Node, int]] = []

        while True:
            if not node.is_expanded:
                policy, value_x = self._evaluate(node.state)
                node.set_prior(policy)
                self._backpropagate(path, value_x)
                return

            action = node.select_action(self.config.c_puct)
            path.append((node, action))
            move = index_to_action(action)
            scratch.make_move(player, move)
            player = opponent(player)

            child = node.child(action)
            if child is None:
                next_state = encode_state(scratch, player)
                child = Node(next_state, player)
                node.add_child(action, child)

            node = child

            if scratch.terminal:
                value_x = scratch.terminal_outcome_from_x_perspective()
                self._backpropagate(path, value_x)
                return

    def _backpropagate(self, path: Sequence[Tuple[Node, int]], value_x: float) -> None:
        for node, action in reversed(path):
            node.visit_counts[action] += 1
            node.total_visits += 1
            perspective = value_x if node.to_move == "X" else -value_x
            node.value_sums[action] += perspective

    def _apply_dirichlet_noise(self, node: Node) -> None:
        legal = np.where(node.legal_mask)[0]
        if legal.size == 0:
            return
        noise = dirichlet_noise(self.config.dirichlet_alpha, legal.size)
        node.prior[legal] = (
            (1 - self.config.dirichlet_epsilon) * node.prior[legal]
            + self.config.dirichlet_epsilon * noise
        )
        node.prior = safe_normalise(node.prior)

    def _evaluate(self, state: EncodedState) -> Tuple[np.ndarray, float]:
        if self.model is None:
            policy = np.ones(81, dtype=np.float64)
            policy[~state.legal_actions] = 0.0
            policy = safe_normalise(policy)
            return policy, 0.0

        if hasattr(self.model, "predict"):
            policy_logits, value = self.model.predict(
                state.planes,
                state.legal_actions.astype(np.float32),
                device=self.device,  # type: ignore[arg-type]
            )
        else:
            import torch  # pragma: no cover

            self.model.eval()
            with torch.no_grad():
                tensor = torch.from_numpy(state.planes).float().unsqueeze(0)
                mask = torch.from_numpy(state.legal_actions.astype(np.float32)).unsqueeze(0)
                if self.device is not None:
                    tensor = tensor.to(self.device)
                    mask = mask.to(self.device)
                    model = self.model.to(self.device)
                else:
                    model = self.model
                policy_logits, value_tensor = model(tensor, mask)
                policy_logits = policy_logits.squeeze(0).cpu().numpy()
                value = float(value_tensor.squeeze().cpu().numpy())

        policy = masked_softmax(policy_logits, state.legal_actions)
        return policy, float(value)


__all__ = ["MCTS", "MCTSConfig", "Node"]
