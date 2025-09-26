"""Reinforcement-learning powered opponents for Ultimate Tic-Tac-Toe."""
from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

try:  # pragma: no cover - optional dependency is validated at runtime
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - fallback when numpy is absent
    np = None  # type: ignore

from .game import Move, Player, UltimateTicTacToe, moves_to_indices

MoveKey = str


def _move_to_key(move: Move) -> MoveKey:
    return f"{move[0]}-{move[1]}"


def _key_to_move(key: MoveKey) -> Move:
    sub, cell = key.split("-")
    return int(sub), int(cell)


def _ensure_table(
    table: Dict[str, Dict[MoveKey, float]],
    state_key: str,
    moves: Sequence[Move],
    default: float,
) -> Dict[MoveKey, float]:
    entries = table.setdefault(state_key, {})
    for move in moves:
        key = _move_to_key(move)
        entries.setdefault(key, default)
    return entries


def _epsilon_greedy(
    table: Dict[MoveKey, float],
    moves: Sequence[Move],
    epsilon: float,
    default: float,
) -> Move:
    if not moves:
        raise ValueError("No legal moves available")
    if random.random() < epsilon:
        return random.choice(list(moves))
    best_value = -math.inf
    best: List[Move] = []
    for move in moves:
        value = table.get(_move_to_key(move), default)
        if value > best_value:
            best_value = value
            best = [move]
        elif value == best_value:
            best.append(move)
    return random.choice(best)


def _require_numpy() -> None:
    if np is None:
        raise ModuleNotFoundError(
            "numpy is required for AlphaZero-lite features. Install it with 'pip install numpy'."
        )


@dataclass
class DoubleQLearningAgent:
    """Double Q-learning agent reducing maximisation bias (idea #1)."""

    alpha: float = 0.4
    gamma: float = 0.95
    default_q: float = 0.0
    q1: Dict[str, Dict[MoveKey, float]] = field(default_factory=dict)
    q2: Dict[str, Dict[MoveKey, float]] = field(default_factory=dict)

    def _state_key(self, game: UltimateTicTacToe, player: Player) -> str:
        return game.serialize(player)

    def _average_table(self, state_key: str, moves: Sequence[Move]) -> Dict[MoveKey, float]:
        table1 = _ensure_table(self.q1, state_key, moves, self.default_q)
        table2 = _ensure_table(self.q2, state_key, moves, self.default_q)
        return {key: 0.5 * (table1[key] + table2[key]) for key in table1}

    def choose_action(
        self,
        state_key: str,
        moves: Sequence[Move],
        epsilon: float,
    ) -> Move:
        avg_table = self._average_table(state_key, moves)
        return _epsilon_greedy(avg_table, moves, epsilon, self.default_q)

    def select_move(
        self,
        game: UltimateTicTacToe,
        player: Player,
        epsilon: float = 0.0,
    ) -> Move:
        state_key = self._state_key(game, player)
        moves = game.available_moves()
        return self.choose_action(state_key, moves, epsilon)

    def update(
        self,
        state_key: str,
        move: Move,
        reward: float,
        next_state_key: Optional[str],
        next_moves: Sequence[Move],
    ) -> None:
        move_key = _move_to_key(move)
        table = self.q1 if random.random() < 0.5 else self.q2
        other = self.q2 if table is self.q1 else self.q1
        entries = _ensure_table(table, state_key, (move,), self.default_q)
        old_value = entries.get(move_key, self.default_q)
        if next_state_key is None:
            target = reward
        else:
            _ensure_table(table, next_state_key, next_moves, self.default_q)
            _ensure_table(other, next_state_key, next_moves, self.default_q)
            target_table = table[next_state_key]
            if next_moves:
                best_next_key = max(
                    (_move_to_key(next_move) for next_move in next_moves),
                    key=lambda k: target_table.get(k, self.default_q),
                )
                next_value = other[next_state_key].get(best_next_key, self.default_q)
            else:
                next_value = 0.0
            target = reward + self.gamma * next_value
        entries[move_key] = old_value + self.alpha * (target - old_value)

    def save(self, path: str) -> None:
        payload = {
            "algorithm": "double_q",
            "q1": self.q1,
            "q2": self.q2,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "default_q": self.default_q,
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)

    @classmethod
    def load(cls, path: str) -> "DoubleQLearningAgent":
        agent = cls()
        if not os.path.exists(path):
            return agent
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict) and "algorithm" in data:
            agent.alpha = float(data.get("alpha", agent.alpha))
            agent.gamma = float(data.get("gamma", agent.gamma))
            agent.default_q = float(data.get("default_q", agent.default_q))
            agent.q1 = {
                state: {move: float(value) for move, value in table.items()}
                for state, table in data.get("q1", {}).items()
            }
            agent.q2 = {
                state: {move: float(value) for move, value in table.items()}
                for state, table in data.get("q2", {}).items()
            }
        else:
            agent.q1 = {
                state: {move: float(value) for move, value in table.items()}
                for state, table in data.items()
            }
            agent.q2 = {}
        return agent


@dataclass
class SARSAgent:
    """On-policy SARSA agent (idea #2)."""

    alpha: float = 0.4
    gamma: float = 0.95
    default_q: float = 0.0
    q_values: Dict[str, Dict[MoveKey, float]] = field(default_factory=dict)

    def _state_key(self, game: UltimateTicTacToe, player: Player) -> str:
        return game.serialize(player)

    def choose_action(
        self,
        state_key: str,
        moves: Sequence[Move],
        epsilon: float,
    ) -> Move:
        table = _ensure_table(self.q_values, state_key, moves, self.default_q)
        return _epsilon_greedy(table, moves, epsilon, self.default_q)

    def select_move(
        self,
        game: UltimateTicTacToe,
        player: Player,
        epsilon: float = 0.0,
    ) -> Move:
        state_key = self._state_key(game, player)
        moves = game.available_moves()
        return self.choose_action(state_key, moves, epsilon)

    def update(
        self,
        state_key: str,
        move: Move,
        reward: float,
        next_state_key: Optional[str],
        next_moves: Sequence[Move],
        next_action: Optional[Move],
    ) -> None:
        move_key = _move_to_key(move)
        table = _ensure_table(self.q_values, state_key, (move,), self.default_q)
        old_value = table.get(move_key, self.default_q)
        if next_state_key is None or next_action is None:
            target = reward
        else:
            next_table = _ensure_table(
                self.q_values, next_state_key, next_moves, self.default_q
            )
            target = reward + self.gamma * next_table.get(
                _move_to_key(next_action), self.default_q
            )
        table[move_key] = old_value + self.alpha * (target - old_value)

    def save(self, path: str) -> None:
        payload = {
            "algorithm": "sarsa",
            "alpha": self.alpha,
            "gamma": self.gamma,
            "default_q": self.default_q,
            "q": self.q_values,
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)

    @classmethod
    def load(cls, path: str) -> "SARSAgent":
        agent = cls()
        if not os.path.exists(path):
            return agent
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        agent.alpha = float(data.get("alpha", agent.alpha))
        agent.gamma = float(data.get("gamma", agent.gamma))
        agent.default_q = float(data.get("default_q", agent.default_q))
        agent.q_values = {
            state: {move: float(value) for move, value in table.items()}
            for state, table in data.get("q", {}).items()
        }
        return agent


@dataclass
class DynaQAgent:
    """Model-based Dyna-Q agent combining learning and planning (idea #5)."""

    alpha: float = 0.4
    gamma: float = 0.95
    planning_steps: int = 20
    default_q: float = 0.0
    q_values: Dict[str, Dict[MoveKey, float]] = field(default_factory=dict)
    model: Dict[Tuple[str, MoveKey], Tuple[float, Optional[str], Tuple[MoveKey, ...]]] = field(
        default_factory=dict
    )

    def _state_key(self, game: UltimateTicTacToe, player: Player) -> str:
        return game.serialize(player)

    def choose_action(
        self,
        state_key: str,
        moves: Sequence[Move],
        epsilon: float,
    ) -> Move:
        table = _ensure_table(self.q_values, state_key, moves, self.default_q)
        return _epsilon_greedy(table, moves, epsilon, self.default_q)

    def select_move(
        self,
        game: UltimateTicTacToe,
        player: Player,
        epsilon: float = 0.0,
    ) -> Move:
        state_key = self._state_key(game, player)
        moves = game.available_moves()
        return self.choose_action(state_key, moves, epsilon)

    def _q_update(
        self,
        state_key: str,
        move_key: MoveKey,
        reward: float,
        next_state_key: Optional[str],
        next_moves: Sequence[Move],
    ) -> None:
        table = _ensure_table(self.q_values, state_key, (_key_to_move(move_key),), self.default_q)
        old_value = table.get(move_key, self.default_q)
        if next_state_key is None:
            target = reward
        else:
            next_table = _ensure_table(
                self.q_values, next_state_key, next_moves, self.default_q
            )
            best_next = max(next_table.values()) if next_table else 0.0
            target = reward + self.gamma * best_next
        table[move_key] = old_value + self.alpha * (target - old_value)

    def update(
        self,
        state_key: str,
        move: Move,
        reward: float,
        next_state_key: Optional[str],
        next_moves: Sequence[Move],
    ) -> None:
        move_key = _move_to_key(move)
        self.model[(state_key, move_key)] = (
            reward,
            next_state_key,
            tuple(_move_to_key(m) for m in next_moves),
        )
        self._q_update(state_key, move_key, reward, next_state_key, next_moves)
        for _ in range(self.planning_steps):
            if not self.model:
                break
            (s_key, a_key), (r, next_key, stored_moves) = random.choice(list(self.model.items()))
            simulated_moves = tuple(_key_to_move(k) for k in stored_moves)
            self._q_update(s_key, a_key, r, next_key, simulated_moves)

    def save(self, path: str) -> None:
        payload = {
            "algorithm": "dyna_q",
            "alpha": self.alpha,
            "gamma": self.gamma,
            "planning_steps": self.planning_steps,
            "default_q": self.default_q,
            "q": self.q_values,
            "model": {
                f"{state}|{move}": (reward, next_state, list(stored_moves))
                for (state, move), (reward, next_state, stored_moves) in self.model.items()
            },
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)

    @classmethod
    def load(cls, path: str) -> "DynaQAgent":
        agent = cls()
        if not os.path.exists(path):
            return agent
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        agent.alpha = float(data.get("alpha", agent.alpha))
        agent.gamma = float(data.get("gamma", agent.gamma))
        agent.default_q = float(data.get("default_q", agent.default_q))
        agent.planning_steps = int(data.get("planning_steps", agent.planning_steps))
        agent.q_values = {
            state: {move: float(value) for move, value in table.items()}
            for state, table in data.get("q", {}).items()
        }
        model: Dict[Tuple[str, MoveKey], Tuple[float, Optional[str], Tuple[MoveKey, ...]]] = {}
        for key, value in data.get("model", {}).items():
            state, move = key.split("|")
            reward, next_state, stored = value
            model[(state, move)] = (
                float(reward),
                next_state,
                tuple(str(m) for m in stored),
            )
        agent.model = model
        return agent


def _encode_player_cell(cell: str, player: Player) -> Tuple[int, int]:
    if cell == " ":
        return 0, 0
    if cell == player:
        return 1, 0
    return 0, 1


def encode_state(game: UltimateTicTacToe, player: Player) -> np.ndarray:
    """Encode the board for the lightweight AlphaZero-style agent."""

    _require_numpy()
    features: List[float] = []
    for sub_board in game.boards:
        for cell in sub_board:
            mine, theirs = _encode_player_cell(cell, player)
            features.extend([mine, theirs])

    for status in game.macro_board:
        mine, theirs = _encode_player_cell(status, player)
        features.extend([mine, theirs])

    forced = game.highlight_boards()
    forced_vec = [0.0] * 10
    if len(forced) == 1:
        forced_vec[forced[0]] = 1.0
    else:
        forced_vec[-1] = 1.0
    features.extend(forced_vec)
    return np.array(features, dtype=np.float32)


class PolicyValueNet:
    """Small fully-connected policy/value network implemented with NumPy."""

    def __init__(self, input_dim: int, hidden_dim: int = 128) -> None:
        _require_numpy()
        rng = np.random.default_rng()
        self.W1 = rng.normal(0, 0.1, size=(hidden_dim, input_dim)).astype(np.float32)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.Wp = rng.normal(0, 0.1, size=(81, hidden_dim)).astype(np.float32)
        self.bp = np.zeros(81, dtype=np.float32)
        self.Wv = rng.normal(0, 0.1, size=(hidden_dim,)).astype(np.float32)
        self.bv = np.zeros(1, dtype=np.float32)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, float, Dict[str, np.ndarray]]:
        _require_numpy()
        z1 = np.tanh(self.W1 @ x + self.b1)
        logits = self.Wp @ z1 + self.bp
        value = float(np.tanh(self.Wv @ z1 + self.bv))
        cache = {"x": x, "z1": z1, "logits": logits, "value": value}
        return logits, value, cache

    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, float]:
        _require_numpy()
        logits, value, _ = self.forward(x)
        return logits, value

    def train_batch(
        self,
        batch_inputs: np.ndarray,
        target_policies: np.ndarray,
        target_values: np.ndarray,
        lr: float = 1e-2,
        l2: float = 1e-4,
    ) -> float:
        _require_numpy()
        batch_size = batch_inputs.shape[0]
        grads = {
            "W1": np.zeros_like(self.W1),
            "b1": np.zeros_like(self.b1),
            "Wp": np.zeros_like(self.Wp),
            "bp": np.zeros_like(self.bp),
            "Wv": np.zeros_like(self.Wv),
            "bv": np.zeros_like(self.bv),
        }
        loss_total = 0.0
        for i in range(batch_size):
            x = batch_inputs[i]
            logits, value, cache = self.forward(x)
            logits_shifted = logits - np.max(logits)
            exp_logits = np.exp(logits_shifted)
            probs = exp_logits / np.sum(exp_logits)
            target_pi = target_policies[i]
            policy_loss = -float(np.sum(target_pi * np.log(probs + 1e-12)))
            dlogits = probs - target_pi
            target_v = float(target_values[i])
            value_loss = (value - target_v) ** 2
            dvalue = 2 * (value - target_v) * (1 - value**2)
            grads["Wp"] += np.outer(dlogits, cache["z1"]) + l2 * self.Wp
            grads["bp"] += dlogits
            grads["Wv"] += cache["z1"] * dvalue + l2 * self.Wv
            grads["bv"] += dvalue
            dz1 = (self.Wp.T @ dlogits) + self.Wv * dvalue
            dz1 *= (1 - cache["z1"] ** 2)
            grads["W1"] += np.outer(dz1, cache["x"]) + l2 * self.W1
            grads["b1"] += dz1
            loss_total += policy_loss + value_loss

        for key, grad in grads.items():
            grads[key] = grad / batch_size

        self.W1 -= lr * grads["W1"]
        self.b1 -= lr * grads["b1"]
        self.Wp -= lr * grads["Wp"]
        self.bp -= lr * grads["bp"]
        self.Wv -= lr * grads["Wv"]
        self.bv -= lr * grads["bv"]
        return loss_total / batch_size

    def to_dict(self) -> Dict[str, List[float]]:
        _require_numpy()
        return {
            "W1": self.W1.tolist(),
            "b1": self.b1.tolist(),
            "Wp": self.Wp.tolist(),
            "bp": self.bp.tolist(),
            "Wv": self.Wv.tolist(),
            "bv": self.bv.tolist(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, List[float]]) -> "PolicyValueNet":
        _require_numpy()
        input_dim = len(data["W1"][0])
        hidden_dim = len(data["W1"])
        net = cls(input_dim, hidden_dim)
        net.W1 = np.array(data["W1"], dtype=np.float32)
        net.b1 = np.array(data["b1"], dtype=np.float32)
        net.Wp = np.array(data["Wp"], dtype=np.float32)
        net.bp = np.array(data["bp"], dtype=np.float32)
        net.Wv = np.array(data["Wv"], dtype=np.float32)
        net.bv = np.array(data["bv"], dtype=np.float32)
        return net


def _opponent(player: Player) -> Player:
    return "O" if player == "X" else "X"


class _MCTSNode:
    def __init__(self, prior: float) -> None:
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: Dict[Move, "_MCTSNode"] = {}

    def expanded(self) -> bool:
        return bool(self.children)

    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def select_child(self, cpuct: float) -> Tuple[Move, "_MCTSNode"]:
        best_move: Optional[Move] = None
        best_score = -math.inf
        best_child: Optional[_MCTSNode] = None
        for move, child in self.children.items():
            u = cpuct * child.prior * math.sqrt(self.visit_count + 1) / (child.visit_count + 1)
            score = child.value() + u
            if score > best_score:
                best_score = score
                best_move = move
                best_child = child
        if best_move is None or best_child is None:
            raise RuntimeError("MCTS child selection failed")
        return best_move, best_child

    def expand(self, moves: Sequence[Move], priors: Sequence[float]) -> None:
        for move, prior in zip(moves, priors):
            self.children[move] = _MCTSNode(prior)

    def backup(self, value: float) -> None:
        self.visit_count += 1
        self.value_sum += value


@dataclass
class AlphaZeroLiteAgent:
    """Small AlphaZero-inspired agent with shared policy/value network (idea #9)."""

    network: PolicyValueNet
    num_simulations: int = 64
    cpuct: float = 1.4

    def select_move(
        self,
        game: UltimateTicTacToe,
        player: Player,
        temperature: float = 0.05,
    ) -> Move:
        _require_numpy()
        moves, visit_probs = self.policy(game, player, temperature)
        if temperature <= 1e-6:
            return moves[int(np.argmax(visit_probs))]
        index = int(np.random.choice(len(moves), p=visit_probs))
        return moves[index]

    def policy(
        self,
        game: UltimateTicTacToe,
        player: Player,
        temperature: float = 1.0,
    ) -> Tuple[List[Move], np.ndarray]:
        _require_numpy()
        root = _MCTSNode(1.0)
        state_vec = encode_state(game, player)
        logits, _, _ = self.network.forward(state_vec)
        moves = game.available_moves()
        priors = self._masked_softmax(logits, moves)
        root.expand(moves, priors)

        for _ in range(self.num_simulations):
            self._simulate(game, player, root)

        visits = np.array([root.children[move].visit_count for move in moves], dtype=np.float32)
        if temperature <= 1e-6:
            probs = np.zeros_like(visits)
            probs[int(np.argmax(visits))] = 1.0
            return list(moves), probs
        visit_probs = visits ** (1.0 / max(temperature, 1e-3))
        visit_probs = visit_probs / np.sum(visit_probs)
        return list(moves), visit_probs

    def _simulate(
        self,
        game: UltimateTicTacToe,
        player: Player,
        root: _MCTSNode,
    ) -> None:
        path: List[Tuple[_MCTSNode, Player]] = []
        _require_numpy()
        node = root
        sim_game = game.clone()
        current_player = player

        while node.expanded() and not sim_game.terminal:
            move, node = node.select_child(self.cpuct)
            sim_game.make_move(current_player, move)
            path.append((node, current_player))
            current_player = _opponent(current_player)

        if sim_game.terminal:
            if sim_game.winner is None:
                value = 0.0
            elif sim_game.winner == player:
                value = 1.0
            else:
                value = -1.0
        else:
            state_vec = encode_state(sim_game, current_player)
            logits, value, _ = self.network.forward(state_vec)
            moves = sim_game.available_moves()
            priors = self._masked_softmax(logits, moves)
            node.expand(moves, priors)
            if current_player != player:
                value = -value

        for node_step, node_player in path:
            node_value = value if node_player == player else -value
            node_step.backup(node_value)
        root.backup(value)

    @staticmethod
    def _masked_softmax(logits: np.ndarray, moves: Sequence[Move]) -> List[float]:
        _require_numpy()
        mask = np.zeros(81, dtype=np.float32)
        indices = moves_to_indices(moves)
        mask[list(indices)] = 1.0
        logits = logits - np.max(logits)
        exp_logits = np.exp(logits) * mask
        if exp_logits.sum() <= 0:
            return [1.0 / len(moves)] * len(moves)
        probs = exp_logits / np.sum(exp_logits)
        return [float(probs[idx]) for idx in indices]

    def save(self, path: str) -> None:
        _require_numpy()
        payload = {
            "algorithm": "alpha_zero_lite",
            "num_simulations": self.num_simulations,
            "cpuct": self.cpuct,
            "network": self.network.to_dict(),
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)

    @classmethod
    def load(cls, path: str, input_dim: int) -> "AlphaZeroLiteAgent":
        _require_numpy()
        if not os.path.exists(path):
            return cls(network=PolicyValueNet(input_dim))
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        network = PolicyValueNet.from_dict(data["network"])
        agent = cls(network=network)
        agent.num_simulations = int(data.get("num_simulations", agent.num_simulations))
        agent.cpuct = float(data.get("cpuct", agent.cpuct))
        return agent


UltimateTTTRLAI = DoubleQLearningAgent


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
