"""Reinforcement-learning powered opponents for Ultimate Tic-Tac-Toe."""
from __future__ import annotations

import importlib.util
import json
import math
import os
import random
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

if importlib.util.find_spec("numpy") is not None:  # pragma: no cover - optional dependency
    import numpy as np  # type: ignore
else:  # pragma: no cover - optional dependency missing
    np = None  # type: ignore

from .game import Move, Player, UltimateTicTacToe

MoveKey = str


def _move_to_key(move: Move) -> MoveKey:
    return f"{move[0]}-{move[1]}"


def _key_to_move(key: MoveKey) -> Move:
    a, b = key.split("-")
    return int(a), int(b)


@dataclass
class UltimateTTTRLAI:
    """Baseline tabular Q-learning agent for Ultimate Tic-Tac-Toe."""

    alpha: float = 0.4
    gamma: float = 0.95
    default_q: float = 0.0
    q_values: Dict[str, Dict[MoveKey, float]] = field(default_factory=dict)
    algorithm: str = "q_learning"

    def _state_key(self, game: UltimateTicTacToe, player: Player) -> str:
        return game.serialize(player)

    def _ensure_state(
        self,
        state_key: str,
        moves: Sequence[Move],
        table: Optional[Dict[str, Dict[MoveKey, float]]] = None,
    ) -> Dict[MoveKey, float]:
        store = self.q_values if table is None else table
        state_table = store.setdefault(state_key, {})
        for move in moves:
            key = _move_to_key(move)
            if key not in state_table:
                state_table[key] = self.default_q
        return state_table

    def choose_action(self, state_key: str, moves: Sequence[Move], epsilon: float) -> Move:
        self._ensure_state(state_key, moves)
        if not moves:
            raise ValueError("No available moves to choose from")
        if random.random() < epsilon:
            return random.choice(list(moves))
        table = self.q_values[state_key]
        best_value = -math.inf
        best_moves: List[Move] = []
        for move in moves:
            value = table.get(_move_to_key(move), self.default_q)
            if value > best_value:
                best_value = value
                best_moves = [move]
            elif value == best_value:
                best_moves.append(move)
        return random.choice(best_moves)

    def best_value(self, state_key: str, moves: Sequence[Move]) -> float:
        if not moves:
            return 0.0
        table = self._ensure_state(state_key, moves)
        return max(table.get(_move_to_key(move), self.default_q) for move in moves)

    def update(
        self,
        state_key: str,
        move: Move,
        reward: float,
        next_state_key: Optional[str],
        next_moves: Sequence[Move],
        next_action: Optional[Move] = None,
        *,
        model_based: bool = False,
    ) -> None:
        move_key = _move_to_key(move)
        table = self._ensure_state(state_key, (move,))
        old_value = table.get(move_key, self.default_q)
        if next_state_key is None:
            target = reward
        else:
            opponent_best = self.best_value(next_state_key, next_moves)
            target = reward - self.gamma * opponent_best
        table[move_key] = old_value + self.alpha * (target - old_value)

    def select_move(
        self,
        game: UltimateTicTacToe,
        player: Player,
        epsilon: float = 0.0,
    ) -> Move:
        state_key = self._state_key(game, player)
        moves = game.available_moves()
        return self.choose_action(state_key, moves, epsilon)

    def save(self, path: str) -> None:
        payload = {"algorithm": self.algorithm, "tables": self.to_serialisable()}
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)

    @classmethod
    def load(cls, path: str, **kwargs) -> "UltimateTTTRLAI":
        agent = cls(**kwargs)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict) and "algorithm" in data:
                agent.load_from_dict(data.get("tables", {}))
            else:
                agent.load_from_dict(data)
        return agent

    def to_serialisable(self) -> Dict[str, Dict[MoveKey, float]]:
        return self.q_values

    def load_from_dict(self, data: Dict[str, Dict[MoveKey, float]]) -> None:
        if "primary" in data and isinstance(data["primary"], dict):
            tables = data["primary"]  # type: ignore[assignment]
        else:
            tables = data
        self.q_values = {
            state: {move: float(value) for move, value in table.items()}
            for state, table in tables.items()
        }


@dataclass
class DoubleQLearningAgent(UltimateTTTRLAI):
    """Double Q-learning variant that mitigates maximisation bias."""

    q_values_b: Dict[str, Dict[MoveKey, float]] = field(default_factory=dict)
    algorithm: str = "double_q"

    def _ensure_state(
        self,
        state_key: str,
        moves: Sequence[Move],
        table: Optional[Dict[str, Dict[MoveKey, float]]] = None,
    ) -> Dict[MoveKey, float]:
        if table is None:
            return super()._ensure_state(state_key, moves)
        state_table = table.setdefault(state_key, {})
        for move in moves:
            move_key = _move_to_key(move)
            if move_key not in state_table:
                state_table[move_key] = self.default_q
        return state_table

    def choose_action(self, state_key: str, moves: Sequence[Move], epsilon: float) -> Move:
        self._ensure_state(state_key, moves)
        self._ensure_state(state_key, moves, self.q_values_b)
        if not moves:
            raise ValueError("No available moves to choose from")
        if random.random() < epsilon:
            return random.choice(list(moves))
        table_a = self.q_values[state_key]
        table_b = self.q_values_b[state_key]
        best_value = -math.inf
        best_moves: List[Move] = []
        for move in moves:
            move_key = _move_to_key(move)
            value = table_a.get(move_key, self.default_q) + table_b.get(
                move_key, self.default_q
            )
            if value > best_value:
                best_value = value
                best_moves = [move]
            elif value == best_value:
                best_moves.append(move)
        return random.choice(best_moves)

    def best_value(self, state_key: str, moves: Sequence[Move]) -> float:
        if not moves:
            return 0.0
        table_a = self._ensure_state(state_key, moves)
        table_b = self._ensure_state(state_key, moves, self.q_values_b)
        return max(
            table_a.get(_move_to_key(move), self.default_q)
            + table_b.get(_move_to_key(move), self.default_q)
            for move in moves
        ) / 2.0

    def update(
        self,
        state_key: str,
        move: Move,
        reward: float,
        next_state_key: Optional[str],
        next_moves: Sequence[Move],
        next_action: Optional[Move] = None,
        *,
        model_based: bool = False,
    ) -> None:
        tables = [self.q_values, self.q_values_b]
        idx = random.randint(0, 1)
        primary = tables[idx]
        secondary = tables[1 - idx]
        self._ensure_state(state_key, (move,), primary)
        self._ensure_state(state_key, (move,), secondary)
        move_key = _move_to_key(move)
        old_value = primary[state_key][move_key]
        if next_state_key is None:
            target = reward
        else:
            self._ensure_state(next_state_key, next_moves, primary)
            self._ensure_state(next_state_key, next_moves, secondary)
            best_move = max(
                next_moves,
                key=lambda mv: primary[next_state_key][_move_to_key(mv)],
            )
            target = reward + self.gamma * secondary[next_state_key][
                _move_to_key(best_move)
            ]
        primary[state_key][move_key] = old_value + self.alpha * (target - old_value)

    def to_serialisable(self) -> Dict[str, Dict[str, Dict[MoveKey, float]]]:
        return {"primary": self.q_values, "secondary": self.q_values_b}

    def load_from_dict(self, data: Dict[str, Dict[str, Dict[MoveKey, float]]]) -> None:
        primary = data.get("primary", {})
        secondary = data.get("secondary", {})
        super().load_from_dict(primary)
        self.q_values_b = {
            state: {move: float(value) for move, value in table.items()}
            for state, table in secondary.items()
        }


@dataclass
class OnPolicySARSAAgent(UltimateTTTRLAI):
    """On-policy SARSA variant using the actually selected next action."""

    algorithm: str = "sarsa"

    def update(
        self,
        state_key: str,
        move: Move,
        reward: float,
        next_state_key: Optional[str],
        next_moves: Sequence[Move],
        next_action: Optional[Move] = None,
        *,
        model_based: bool = False,
    ) -> None:
        move_key = _move_to_key(move)
        table = self._ensure_state(state_key, (move,))
        old_value = table.get(move_key, self.default_q)
        if next_state_key is None or next_action is None:
            target = reward
        else:
            next_table = self._ensure_state(next_state_key, (next_action,))
            next_key = _move_to_key(next_action)
            target = reward + self.gamma * next_table.get(next_key, self.default_q)
        table[move_key] = old_value + self.alpha * (target - old_value)


@dataclass
class DynaQAgent(UltimateTTTRLAI):
    """Dyna-Q agent that augments learning with planning updates."""

    planning_steps: int = 10
    model: Dict[Tuple[str, MoveKey], Tuple[float, Optional[str], Tuple[Move, ...]]] = field(
        default_factory=dict
    )
    algorithm: str = "dyna_q"

    def update(
        self,
        state_key: str,
        move: Move,
        reward: float,
        next_state_key: Optional[str],
        next_moves: Sequence[Move],
        next_action: Optional[Move] = None,
        *,
        model_based: bool = False,
    ) -> None:
        super().update(
            state_key,
            move,
            reward,
            next_state_key,
            next_moves,
            next_action=next_action,
            model_based=model_based,
        )
        if model_based:
            return
        move_key = _move_to_key(move)
        self.model[(state_key, move_key)] = (
            reward,
            next_state_key,
            tuple(next_moves),
        )
        self._planning_updates()

    def _planning_updates(self) -> None:
        if not self.model or self.planning_steps <= 0:
            return
        items = list(self.model.items())
        for _ in range(self.planning_steps):
            (state_key, move_key), (reward, next_state_key, next_moves) = random.choice(items)
            move = _key_to_move(move_key)
            self.update(
                state_key,
                move,
                reward,
                next_state_key,
                next_moves,
                model_based=True,
            )

    @classmethod
    def load(cls, path: str, **kwargs) -> "DynaQAgent":
        planning_steps = kwargs.get("planning_steps")
        agent = cls(planning_steps=planning_steps or cls.planning_steps)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict) and "algorithm" in data:
                agent.load_from_dict(data.get("tables", {}))
            else:
                agent.load_from_dict(data)
        return agent


if np is not None:  # pragma: no cover - optional dependency
    INPUT_SIZE = 81 * 2 + 9 * 3 + 10


    def _softmax(logits: np.ndarray) -> np.ndarray:
        logits = logits - np.max(logits)
        exps = np.exp(logits)
        total = np.sum(exps)
        if total == 0.0:
            return np.full_like(logits, 1.0 / logits.size)
        return exps / total


    def encode_state(game: UltimateTicTacToe, player: Player) -> np.ndarray:
        opponent = "O" if player == "X" else "X"
        own_cells: List[float] = []
        opp_cells: List[float] = []
        for board in game.boards:
            for cell in board:
                own_cells.append(1.0 if cell == player else 0.0)
                opp_cells.append(1.0 if cell == opponent else 0.0)
        macro_owner = game.macro_board
        macro_own = [1.0 if cell == player else 0.0 for cell in macro_owner]
        macro_opp = [1.0 if cell == opponent else 0.0 for cell in macro_owner]
        macro_tie = [1.0 if cell == "T" else 0.0 for cell in macro_owner]
        forced = game._forced_board_index()
        forced_vec = [0.0] * 10
        if forced is None:
            forced_vec[-1] = 1.0
        else:
            forced_vec[forced] = 1.0
        return np.array(
            own_cells + opp_cells + macro_own + macro_opp + macro_tie + forced_vec,
            dtype=np.float32,
        )


    @dataclass
    class TinyPolicyValueNet:
        """Minimal neural network used by the AlphaZero-like agent."""

        input_size: int
        hidden_size: int = 128
        seed: Optional[int] = None

        def __post_init__(self) -> None:
            rng = np.random.default_rng(self.seed)
            scale1 = 1.0 / math.sqrt(self.input_size)
            scale2 = 1.0 / math.sqrt(self.hidden_size)
            self.W1 = rng.normal(scale=scale1, size=(self.input_size, self.hidden_size))
            self.b1 = np.zeros(self.hidden_size)
            self.Wp = rng.normal(scale=scale2, size=(self.hidden_size, 81))
            self.bp = np.zeros(81)
            self.Wv = rng.normal(scale=scale2, size=(self.hidden_size, 1))
            self.bv = np.zeros(1)

        def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
            h = np.tanh(x @ self.W1 + self.b1)
            logits = h @ self.Wp + self.bp
            value_raw = float(h @ self.Wv + self.bv)
            value = math.tanh(value_raw)
            return h, logits, value

        def predict(self, x: np.ndarray, legal_moves: Sequence[Move]) -> Tuple[np.ndarray, float]:
            _, logits, value = self.forward(x)
            if not legal_moves:
                return np.zeros(81, dtype=np.float32), value
            indices = [move[0] * 9 + move[1] for move in legal_moves]
            legal_logits = logits[indices]
            legal_probs = _softmax(legal_logits)
            policy = np.zeros(81, dtype=np.float32)
            policy[indices] = legal_probs.astype(np.float32)
            return policy, value

        def train_batch(
            self, states: np.ndarray, policies: np.ndarray, values: np.ndarray, lr: float
        ) -> None:
            batch_size = states.shape[0]
            for idx in range(batch_size):
                x = states[idx]
                target_policy = policies[idx]
                target_value = values[idx]
                h, logits, value = self.forward(x)
                policy = _softmax(logits)
                dlogits = policy - target_policy
                dWp = np.outer(h, dlogits)
                dbp = dlogits
                dvalue = value - target_value
                value_grad = dvalue * (1.0 - value**2)
                dWv = np.outer(h, value_grad)
                dbv = np.array([value_grad])
                hidden_grad = (dlogits @ self.Wp.T) + (value_grad * self.Wv.flatten())
                hidden_grad *= (1.0 - h**2)
                dW1 = np.outer(x, hidden_grad)
                db1 = hidden_grad
                self.Wp -= lr * dWp
                self.bp -= lr * dbp
                self.Wv -= lr * dWv.reshape(self.Wv.shape)
                self.bv -= lr * dbv
                self.W1 -= lr * dW1
                self.b1 -= lr * db1

        def to_dict(self) -> Dict[str, List[List[float]]]:
            return {
                "W1": self.W1.tolist(),
                "b1": self.b1.tolist(),
                "Wp": self.Wp.tolist(),
                "bp": self.bp.tolist(),
                "Wv": self.Wv.tolist(),
                "bv": self.bv.tolist(),
            }

        def load_dict(self, data: Dict[str, List[List[float]]]) -> None:
            self.W1 = np.array(data["W1"], dtype=np.float64)
            self.b1 = np.array(data["b1"], dtype=np.float64)
            self.Wp = np.array(data["Wp"], dtype=np.float64)
            self.bp = np.array(data["bp"], dtype=np.float64)
            self.Wv = np.array(data["Wv"], dtype=np.float64)
            self.bv = np.array(data["bv"], dtype=np.float64)


    @dataclass
    class MCTSNode:
        game: UltimateTicTacToe
        player: Player
        parent: Optional["MCTSNode"] = None
        prior: float = 0.0
        children: Dict[Move, "MCTSNode"] = field(default_factory=dict)
        visit_count: int = 0
        value_sum: float = 0.0

        def expanded(self) -> bool:
            return bool(self.children)

        def value(self) -> float:
            if self.visit_count == 0:
                return 0.0
            return self.value_sum / self.visit_count

        def select_child(self, c_puct: float) -> Tuple[Move, "MCTSNode"]:
            total_visits = sum(child.visit_count for child in self.children.values())
            best_score = -math.inf
            best_move: Optional[Move] = None
            best_child: Optional[MCTSNode] = None
            for move, child in self.children.items():
                prior = child.prior
                q_value = child.value()
                u_value = c_puct * prior * math.sqrt(total_visits + 1) / (child.visit_count + 1)
                score = q_value + u_value
                if score > best_score:
                    best_score = score
                    best_move = move
                    best_child = child
            if best_move is None or best_child is None:
                raise RuntimeError("MCTS selection failed to find a child")
            return best_move, best_child

        def expand(
            self,
            policy: np.ndarray,
            next_player: Player,
        ) -> None:
            legal_moves = self.game.available_moves()
            for move in legal_moves:
                idx = move[0] * 9 + move[1]
                cloned = self.game.clone()
                cloned.make_move(self.player, move)
                child = MCTSNode(cloned, next_player, parent=self, prior=float(policy[idx]))
                self.children[move] = child

        def backup(self, value: float, leaf_player: Player) -> None:
            node: Optional[MCTSNode] = self
            current_value = value
            current_player = leaf_player
            while node is not None:
                node.visit_count += 1
                if node.player == current_player:
                    node.value_sum += current_value
                else:
                    node.value_sum -= current_value
                current_value = -current_value
                current_player = "O" if current_player == "X" else "X"
                node = node.parent


    @dataclass
    class AlphaZeroAgent:
        """Simplified AlphaZero-style agent with tiny policy/value network and MCTS."""

        network: TinyPolicyValueNet
        num_simulations: int = 25
        c_puct: float = 1.4
        learning_rate: float = 0.01

        @classmethod
        def load(
            cls,
            path: str,
            *,
            simulations: Optional[int] = None,
            learning_rate: Optional[float] = None,
            seed: Optional[int] = None,
        ) -> "AlphaZeroAgent":
            network = TinyPolicyValueNet(INPUT_SIZE, seed=seed)
            agent = cls(
                network=network,
                num_simulations=simulations or cls.num_simulations,
                learning_rate=learning_rate or 0.01,
            )
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                net_data = data.get("network")
                if net_data:
                    agent.network.load_dict(net_data)
                agent.num_simulations = data.get("num_simulations", agent.num_simulations)
                agent.learning_rate = data.get("learning_rate", agent.learning_rate)
            return agent

        def save(self, path: str) -> None:
            payload = {
                "algorithm": "alphazero_like",
                "num_simulations": self.num_simulations,
                "learning_rate": self.learning_rate,
                "network": self.network.to_dict(),
            }
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh)

        def plan(self, game: UltimateTicTacToe, player: Player) -> np.ndarray:
            root = MCTSNode(game.clone(), player)
            legal_moves = game.available_moves()
            if not legal_moves:
                return np.zeros(81, dtype=np.float32)
            state_vec = encode_state(game, player)
            policy, _ = self.network.predict(state_vec, legal_moves)
            root.expand(policy, "O" if player == "X" else "X")
            for _ in range(self.num_simulations):
                node = root
                current_player = player
                search_path = [node]
                while node.expanded() and not node.game.terminal:
                    move, node = node.select_child(self.c_puct)
                    current_player = "O" if current_player == "X" else "X"
                    search_path.append(node)
                if node.game.terminal:
                    if node.game.winner is None:
                        value = 0.0
                    else:
                        value = 1.0 if node.game.winner == current_player else -1.0
                else:
                    next_player = "O" if current_player == "X" else "X"
                    state_vec = encode_state(node.game, current_player)
                    policy, value = self.network.predict(
                        state_vec, node.game.available_moves()
                    )
                    node.expand(policy, next_player)
                search_path[-1].backup(value, current_player)
            visits = np.zeros(81, dtype=np.float32)
            for move, child in root.children.items():
                idx = move[0] * 9 + move[1]
                visits[idx] = child.visit_count
            if np.sum(visits) == 0:
                visits = np.ones_like(visits)
            return visits / np.sum(visits)

        def select_move(
            self, game: UltimateTicTacToe, player: Player, temperature: float = 1.0
        ) -> Move:
            policy = self.plan(game, player)
            legal_moves = game.available_moves()
            if not legal_moves:
                raise ValueError("No legal moves available")
            if temperature <= 1e-6:
                best = max(legal_moves, key=lambda mv: policy[mv[0] * 9 + mv[1]])
                return best
            probs = np.array(
                [policy[mv[0] * 9 + mv[1]] for mv in legal_moves], dtype=np.float64
            )
            probs = probs ** (1.0 / max(temperature, 1e-6))
            probs /= probs.sum()
            choice = random.choices(legal_moves, weights=probs, k=1)[0]
            return choice

        def self_play_episode(
            self, temperature: float
        ) -> Tuple[List[Tuple[np.ndarray, np.ndarray, float]], Optional[Player], int]:
            game = UltimateTicTacToe()
            player = "X"
            history: List[Tuple[Player, np.ndarray, np.ndarray]] = []
            move_count = 0
            while not game.terminal:
                state_vec = encode_state(game, player)
                visit_probs = self.plan(game, player)
                legal_moves = game.available_moves()
                if not legal_moves:
                    break
                policy = np.zeros(81, dtype=np.float32)
                for move in legal_moves:
                    idx = move[0] * 9 + move[1]
                    policy[idx] = visit_probs[idx]
                policy_sum = float(policy.sum())
                if policy_sum > 0:
                    policy /= policy_sum
                move = self.select_move(game, player, temperature)
                history.append((player, state_vec, policy))
                game.make_move(player, move)
                move_count += 1
                player = "O" if player == "X" else "X"
            if game.winner is None:
                value_lookup = {"X": 0.0, "O": 0.0}
            else:
                value_lookup = {
                    "X": 1.0 if game.winner == "X" else -1.0,
                    "O": 1.0 if game.winner == "O" else -1.0,
                }
            samples: List[Tuple[np.ndarray, np.ndarray, float]] = []
            for player_id, state_vec, policy in history:
                samples.append((state_vec, policy, value_lookup[player_id]))
            return samples, game.winner, move_count

        def train_on_samples(self, samples: List[Tuple[np.ndarray, np.ndarray, float]]) -> None:
            if not samples:
                return
            states = np.stack([sample[0] for sample in samples])
            policies = np.stack([sample[1] for sample in samples])
            values = np.array([sample[2] for sample in samples])
            self.network.train_batch(states, policies, values, self.learning_rate)


else:  # pragma: no cover - optional dependency missing

    class AlphaZeroAgent:  # type: ignore[misc]
        """Placeholder when NumPy is unavailable."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            raise ImportError(
                "AlphaZeroAgent requires the optional dependency 'numpy'. "
                "Please install numpy to enable this architecture."
            )

        @classmethod
        def load(cls, *args: object, **kwargs: object) -> "AlphaZeroAgent":
            raise ImportError(
                "AlphaZeroAgent requires the optional dependency 'numpy'. "
                "Please install numpy to enable this architecture."
            )


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


__all__ = [
    "UltimateTTTRLAI",
    "DoubleQLearningAgent",
    "OnPolicySARSAAgent",
    "DynaQAgent",
    "AlphaZeroAgent",
    "immediate_winning_move",
    "block_opponent_move",
]
