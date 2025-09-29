"""AlphaZero-inspired reinforcement learning opponents for Ultimate Tic-Tac-Toe."""
from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .game import Move, Player, UltimateTicTacToe

POLICY_SIZE = 81
INPUT_SIZE = 343


def move_to_index(move: Move) -> int:
    return move[0] * 9 + move[1]


def index_to_move(index: int) -> Move:
    return index // 9, index % 9


def relu(x: float) -> float:
    return x if x > 0.0 else 0.0


def relu_grad(x: float) -> float:
    return 1.0 if x > 0.0 else 0.0


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def softmax(logits: Sequence[float]) -> List[float]:
    if not logits:
        return []
    max_logit = max(logits)
    exp = [math.exp(value - max_logit) for value in logits]
    total = sum(exp)
    if total <= 0.0:
        return [1.0 / len(logits) for _ in logits]
    return [value / total for value in exp]


def dirichlet(alpha: Sequence[float]) -> List[float]:
    samples = [random.gammavariate(a, 1.0) for a in alpha]
    total = sum(samples)
    if total <= 0.0:
        return [1.0 / len(alpha) for _ in alpha]
    return [sample / total for sample in samples]


def encode_game_state(game: UltimateTicTacToe, player: Player) -> List[float]:
    """Encode the full game state into a flat list of floats."""

    current = [0.0] * POLICY_SIZE
    opponent = [0.0] * POLICY_SIZE
    empty = [0.0] * POLICY_SIZE
    legal = [0.0] * POLICY_SIZE
    macro = [0.0] * 9
    focus = [0.0] * 9

    opponent_player = "O" if player == "X" else "X"

    for sub_index, board in enumerate(game.boards):
        for cell_index, value in enumerate(board):
            idx = move_to_index((sub_index, cell_index))
            if value == player:
                current[idx] = 1.0
            elif value == opponent_player:
                opponent[idx] = 1.0
            else:
                empty[idx] = 1.0

    for move in game.available_moves():
        legal[move_to_index(move)] = 1.0

    for idx, owner in enumerate(game.macro_board):
        if owner == player:
            macro[idx] = 1.0
        elif owner == opponent_player:
            macro[idx] = -1.0
        elif owner == "T":
            macro[idx] = 0.5

    for idx in game.highlight_boards():
        focus[idx] = 1.0

    player_indicator = [1.0 if player == "X" else -1.0]

    return current + opponent + empty + legal + macro + focus + player_indicator


class AlphaZeroNetwork:
    """A small neural network approximating policy and value outputs."""

    def __init__(
        self,
        input_size: int = INPUT_SIZE,
        hidden_sizes: Tuple[int, int] = (256, 128),
        policy_size: int = POLICY_SIZE,
        seed: Optional[int] = None,
    ) -> None:
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.policy_size = policy_size
        self._rng = random.Random(seed)
        self._init_parameters()

    def _init_parameters(self) -> None:
        h1, h2 = self.hidden_sizes
        scale1 = 1.0 / math.sqrt(self.input_size)
        scale2 = 1.0 / math.sqrt(max(h1, 1))
        scale_head = 1.0 / math.sqrt(max(h2, 1))

        self.W1 = [
            [self._rng.gauss(0.0, scale1) for _ in range(self.input_size)] for _ in range(h1)
        ]
        self.b1 = [0.0] * h1

        self.W2 = [
            [self._rng.gauss(0.0, scale2) for _ in range(h1)] for _ in range(h2)
        ]
        self.b2 = [0.0] * h2

        self.Wp = [
            [self._rng.gauss(0.0, scale_head) for _ in range(h2)] for _ in range(self.policy_size)
        ]
        self.bp = [0.0] * self.policy_size

        self.Wv = [self._rng.gauss(0.0, scale_head) for _ in range(h2)]
        self.bv = 0.0

    # ------------------------------------------------------------------
    def forward(self, x: Sequence[float]) -> Tuple[List[float], float, Tuple]:
        h1, h2 = self.hidden_sizes

        z1 = [dot(self.W1[j], x) + self.b1[j] for j in range(h1)]
        a1 = [relu(val) for val in z1]

        z2 = [dot(self.W2[j], a1) + self.b2[j] for j in range(h2)]
        a2 = [relu(val) for val in z2]

        logits = [dot(self.Wp[j], a2) + self.bp[j] for j in range(self.policy_size)]
        value_pre = dot(self.Wv, a2) + self.bv
        value = math.tanh(value_pre)

        cache = (x, z1, a1, z2, a2, value_pre)
        return logits, value, cache

    def predict(self, x: Sequence[float]) -> Tuple[List[float], float]:
        logits, value, _ = self.forward(x)
        return logits, value

    # ------------------------------------------------------------------
    def train_step(
        self,
        states: Sequence[Sequence[float]],
        target_policies: Sequence[Sequence[float]],
        target_values: Sequence[float],
        learning_rate: float,
    ) -> float:
        batch_size = len(states)
        if batch_size == 0:
            return 0.0

        h1, h2 = self.hidden_sizes

        grad_W1 = [[0.0 for _ in range(self.input_size)] for _ in range(h1)]
        grad_b1 = [0.0 for _ in range(h1)]
        grad_W2 = [[0.0 for _ in range(h1)] for _ in range(h2)]
        grad_b2 = [0.0 for _ in range(h2)]
        grad_Wp = [[0.0 for _ in range(h2)] for _ in range(self.policy_size)]
        grad_bp = [0.0 for _ in range(self.policy_size)]
        grad_Wv = [0.0 for _ in range(h2)]
        grad_bv = 0.0

        total_loss = 0.0

        for state, target_policy, target_value in zip(states, target_policies, target_values):
            logits, value, cache = self.forward(state)
            x, z1, a1, z2, a2, value_pre = cache

            probs = softmax(logits)
            policy_loss = -sum(
                target * math.log(max(prob, 1e-10))
                for target, prob in zip(target_policy, probs)
            )
            value_loss = (value - target_value) ** 2
            total_loss += policy_loss + value_loss

            dlogits = [prob - target for prob, target in zip(probs, target_policy)]
            dvalue = 2.0 * (value - target_value) * (1.0 - value ** 2)

            hidden_grad = [0.0 for _ in range(h2)]
            for j in range(self.policy_size):
                for h in range(h2):
                    grad_Wp[j][h] += dlogits[j] * a2[h]
                    hidden_grad[h] += dlogits[j] * self.Wp[j][h]
                grad_bp[j] += dlogits[j]

            for h in range(h2):
                grad_Wv[h] += dvalue * a2[h]
                hidden_grad[h] += dvalue * self.Wv[h]
            grad_bv += dvalue

            dz2 = [hidden_grad[h] * relu_grad(z2[h]) for h in range(h2)]
            da1 = [0.0 for _ in range(h1)]
            for h in range(h2):
                for k in range(h1):
                    grad_W2[h][k] += dz2[h] * a1[k]
                    da1[k] += dz2[h] * self.W2[h][k]
                grad_b2[h] += dz2[h]

            dz1 = [da1[k] * relu_grad(z1[k]) for k in range(h1)]
            for k in range(h1):
                for i in range(self.input_size):
                    grad_W1[k][i] += dz1[k] * x[i]
                grad_b1[k] += dz1[k]

        scale = learning_rate / batch_size

        for j in range(self.policy_size):
            for h in range(h2):
                self.Wp[j][h] -= scale * grad_Wp[j][h]
            self.bp[j] -= scale * grad_bp[j]

        for h in range(h2):
            for k in range(h1):
                self.W2[h][k] -= scale * grad_W2[h][k]
            self.b2[h] -= scale * grad_b2[h]

        for k in range(h1):
            for i in range(self.input_size):
                self.W1[k][i] -= scale * grad_W1[k][i]
            self.b1[k] -= scale * grad_b1[k]

        for h in range(h2):
            self.Wv[h] -= scale * grad_Wv[h]
        self.bv -= scale * grad_bv

        return total_loss / batch_size

    def save(self, path: str) -> None:
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        data = {
            "input_size": self.input_size,
            "hidden_sizes": list(self.hidden_sizes),
            "policy_size": self.policy_size,
            "W1": self.W1,
            "b1": self.b1,
            "W2": self.W2,
            "b2": self.b2,
            "Wp": self.Wp,
            "bp": self.bp,
            "Wv": self.Wv,
            "bv": self.bv,
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh)

    @classmethod
    def load(cls, path: str) -> "AlphaZeroNetwork":
        if not os.path.exists(path):
            return cls()
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        network = cls(
            input_size=int(data.get("input_size", INPUT_SIZE)),
            hidden_sizes=tuple(data.get("hidden_sizes", (256, 128))),
            policy_size=int(data.get("policy_size", POLICY_SIZE)),
        )
        network.W1 = [[float(v) for v in row] for row in data.get("W1", network.W1)]
        network.b1 = [float(v) for v in data.get("b1", network.b1)]
        network.W2 = [[float(v) for v in row] for row in data.get("W2", network.W2)]
        network.b2 = [float(v) for v in data.get("b2", network.b2)]
        network.Wp = [[float(v) for v in row] for row in data.get("Wp", network.Wp)]
        network.bp = [float(v) for v in data.get("bp", network.bp)]
        network.Wv = [float(v) for v in data.get("Wv", network.Wv)]
        network.bv = float(data.get("bv", network.bv))
        return network


@dataclass
class MCTSNode:
    game: UltimateTicTacToe
    to_play: Player
    prior: float
    parent: Optional["MCTSNode"] = None
    children: Dict[Move, "MCTSNode"] = None
    visit_count: int = 0
    value_sum: float = 0.0

    def __post_init__(self) -> None:
        if self.children is None:
            self.children = {}

    def expanded(self) -> bool:
        return bool(self.children)

    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


class AlphaZeroMCTS:
    def __init__(
        self,
        network: AlphaZeroNetwork,
        num_simulations: int = 160,
        c_puct: float = 1.4,
        dirichlet_alpha: float = 0.3,
        dirichlet_fraction: float = 0.25,
    ) -> None:
        self.network = network
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_fraction = dirichlet_fraction

    def run(
        self,
        game: UltimateTicTacToe,
        player: Player,
        temperature: float = 0.0,
        add_noise: bool = False,
    ) -> Tuple[Move, List[float]]:
        root = MCTSNode(game.clone(), player, prior=1.0)
        value = self._expand(root, add_noise=add_noise)
        root.value_sum += value
        root.visit_count += 1

        for _ in range(self.num_simulations):
            node = root
            path = [node]
            while node.expanded() and not node.game.terminal:
                node = self._select_child(node)
                path.append(node)
            value = self._expand(node)
            self._backpropagate(path, value)

        policy = self._policy_from_visits(root, temperature)
        move = self._select_move(root, policy)
        return move, policy

    def _select_child(self, node: MCTSNode) -> MCTSNode:
        assert node.children, "select_child requires expanded node"
        best_score = -float("inf")
        best_child = None
        total = math.sqrt(node.visit_count)
        for move, child in node.children.items():
            exploration = self.c_puct * child.prior * total / (1 + child.visit_count)
            score = child.value() + exploration
            if score > best_score:
                best_score = score
                best_child = child
        assert best_child is not None
        return best_child

    def _expand(self, node: MCTSNode, add_noise: bool = False) -> float:
        if node.game.terminal:
            if node.game.winner is None:
                return 0.0
            return 1.0 if node.game.winner == node.to_play else -1.0

        state_vector = encode_game_state(node.game, node.to_play)
        logits, value = self.network.predict(state_vector)
        valid_moves = node.game.available_moves()
        if not valid_moves:
            return value

        policy = self._masked_softmax(logits, valid_moves)
        if add_noise and valid_moves:
            noise = dirichlet([self.dirichlet_alpha] * len(valid_moves))
            for idx, move in enumerate(valid_moves):
                move_index = move_to_index(move)
                policy_value = policy[move_index]
                policy[move_index] = (
                    (1.0 - self.dirichlet_fraction) * policy_value
                    + self.dirichlet_fraction * noise[idx]
                )

        total = sum(policy[move_to_index(move)] for move in valid_moves)
        if total <= 0.0:
            uniform = 1.0 / len(valid_moves)
            for move in valid_moves:
                policy[move_to_index(move)] = uniform
        else:
            for move in valid_moves:
                policy[move_to_index(move)] /= total

        for move in valid_moves:
            child_game = node.game.clone()
            child_game.make_move(node.to_play, move)
            next_player = "O" if node.to_play == "X" else "X"
            node.children[move] = MCTSNode(
                child_game,
                next_player,
                prior=policy[move_to_index(move)],
                parent=node,
            )

        return value

    def _masked_softmax(self, logits: Sequence[float], moves: Sequence[Move]) -> List[float]:
        scores = [-float("inf")] * POLICY_SIZE
        indices = [move_to_index(move) for move in moves]
        for idx in indices:
            scores[idx] = logits[idx]
        if not indices:
            return [0.0] * POLICY_SIZE
        max_val = max(scores[idx] for idx in indices)
        exp_scores = [0.0] * POLICY_SIZE
        total = 0.0
        for idx in indices:
            value = math.exp(scores[idx] - max_val)
            exp_scores[idx] = value
            total += value
        if total <= 0.0:
            return [0.0] * POLICY_SIZE
        return [value / total for value in exp_scores]

    def _backpropagate(self, path: List[MCTSNode], value: float) -> None:
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum += value
            value = -value

    def _policy_from_visits(self, root: MCTSNode, temperature: float) -> List[float]:
        if not root.children:
            return [0.0] * POLICY_SIZE
        visits = [0.0] * POLICY_SIZE
        for move, child in root.children.items():
            visits[move_to_index(move)] = float(child.visit_count)

        if temperature <= 1e-6:
            best = max(root.children.items(), key=lambda item: item[1].visit_count)[0]
            policy = [0.0] * POLICY_SIZE
            policy[move_to_index(best)] = 1.0
            return policy

        scale = 1.0 / max(temperature, 1e-6)
        scaled = [count ** scale if count > 0.0 else 0.0 for count in visits]
        total = sum(scaled)
        policy = [0.0] * POLICY_SIZE
        if total <= 0.0:
            if root.children:
                uniform = 1.0 / len(root.children)
                for move in root.children:
                    policy[move_to_index(move)] = uniform
            return policy
        for idx, value in enumerate(scaled):
            if value > 0.0:
                policy[idx] = value / total
        return policy

    def _select_move(self, root: MCTSNode, policy: Sequence[float]) -> Move:
        if root.children:
            best_move = max(root.children.items(), key=lambda item: item[1].visit_count)[0]
            best_index = move_to_index(best_move)
            if policy[best_index] > 0:
                return best_move
            return best_move
        return (0, 0)


class UltimateTTTRLAI:
    def __init__(
        self,
        network: Optional[AlphaZeroNetwork] = None,
        num_simulations: int = 160,
        model_path: Optional[str] = None,
    ) -> None:
        self.network = network or AlphaZeroNetwork()
        self.num_simulations = num_simulations
        self.model_path = model_path
        self._mcts = AlphaZeroMCTS(self.network, num_simulations=self.num_simulations)

    def set_num_simulations(self, num_simulations: int) -> None:
        self.num_simulations = num_simulations
        self._mcts = AlphaZeroMCTS(self.network, num_simulations=self.num_simulations)

    def select_move(
        self,
        game: UltimateTicTacToe,
        player: Player,
        temperature: float = 0.0,
        add_noise: bool = False,
    ) -> Move:
        move, _ = self._mcts.run(game, player, temperature=temperature, add_noise=add_noise)
        return move

    def policy(
        self,
        game: UltimateTicTacToe,
        player: Player,
        temperature: float = 1.0,
        add_noise: bool = False,
    ) -> Tuple[Move, List[float]]:
        return self._mcts.run(game, player, temperature=temperature, add_noise=add_noise)

    def save(self, path: Optional[str] = None) -> None:
        target = path or self.model_path
        if target is None:
            raise ValueError("No path specified to save the agent")
        self.network.save(target)

    @classmethod
    def load(
        cls, path: str, num_simulations: int = 160, seed: Optional[int] = None
    ) -> "UltimateTTTRLAI":
        if os.path.exists(path):
            network = AlphaZeroNetwork.load(path)
        else:
            network = AlphaZeroNetwork(seed=seed)
        agent = cls(network=network, num_simulations=num_simulations, model_path=path)
        return agent

    def to_serialisable(self) -> Dict[str, List[List[float]]]:
        return {
            "W1": self.network.W1,
            "b1": self.network.b1,
            "W2": self.network.W2,
            "b2": self.network.b2,
            "Wp": self.network.Wp,
            "bp": self.network.bp,
            "Wv": self.network.Wv,
            "bv": [self.network.bv],
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
