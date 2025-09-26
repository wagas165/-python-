"""Training utilities for multiple Ultimate Tic-Tac-Toe RL agents."""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

try:  # pragma: no cover - optional dependency validated at runtime
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - fallback when numpy is absent
    np = None  # type: ignore

from .ai import AlphaZeroLiteAgent, DoubleQLearningAgent, DynaQAgent, SARSAgent, encode_state
from .game import Move, UltimateTicTacToe


def _opponent(player: str) -> str:
    return "O" if player == "X" else "X"


def _reward(outcome: Optional[str], player: str) -> float:
    if outcome is None:
        return 0.0
    if outcome == player:
        return 1.0
    if outcome == "draw":
        return 0.2
    return -1.0


def play_episode_double(agent: DoubleQLearningAgent, epsilon: float) -> Tuple[str, int]:
    game = UltimateTicTacToe()
    current_player = "X"
    move_count = 0

    while not game.terminal:
        state_key = game.serialize(current_player)
        moves = game.available_moves()
        move = agent.choose_action(state_key, moves, epsilon)
        game.make_move(current_player, move)
        move_count += 1

        if game.terminal:
            outcome = game.winner if game.winner else ("draw" if game.is_draw else None)
            reward = _reward(outcome, current_player)
            agent.update(state_key, move, reward, None, [])
            return outcome or _opponent(current_player), move_count

        next_player = _opponent(current_player)
        next_state_key = game.serialize(next_player)
        next_moves = game.available_moves()
        agent.update(state_key, move, 0.0, next_state_key, next_moves)
        current_player = next_player

    return "draw", move_count


def play_episode_sarsa(agent: SARSAgent, epsilon: float) -> Tuple[str, int]:
    game = UltimateTicTacToe()
    current_player = "X"
    state_key = game.serialize(current_player)
    moves = game.available_moves()
    action = agent.choose_action(state_key, moves, epsilon)
    move_count = 0

    while True:
        game.make_move(current_player, action)
        move_count += 1
        next_player = _opponent(current_player)
        if game.terminal:
            outcome = game.winner if game.winner else ("draw" if game.is_draw else None)
            reward = _reward(outcome, current_player)
            agent.update(state_key, action, reward, None, [], None)
            return outcome or next_player, move_count

        next_state_key = game.serialize(next_player)
        next_moves = game.available_moves()
        next_action = agent.choose_action(next_state_key, next_moves, epsilon)
        agent.update(state_key, action, 0.0, next_state_key, next_moves, next_action)
        state_key = next_state_key
        action = next_action
        current_player = next_player


def play_episode_dyna(agent: DynaQAgent, epsilon: float) -> Tuple[str, int]:
    game = UltimateTicTacToe()
    current_player = "X"
    move_count = 0

    while not game.terminal:
        state_key = game.serialize(current_player)
        moves = game.available_moves()
        move = agent.choose_action(state_key, moves, epsilon)
        game.make_move(current_player, move)
        move_count += 1

        if game.terminal:
            outcome = game.winner if game.winner else ("draw" if game.is_draw else None)
            reward = _reward(outcome, current_player)
            agent.update(state_key, move, reward, None, [])
            return outcome or _opponent(current_player), move_count

        next_player = _opponent(current_player)
        next_state_key = game.serialize(next_player)
        next_moves = game.available_moves()
        agent.update(state_key, move, 0.0, next_state_key, next_moves)
        current_player = next_player

    return "draw", move_count


def _policy_vector(moves: Sequence[Move], probs: Sequence[float]) -> np.ndarray:
    if np is None:
        raise ModuleNotFoundError(
            "numpy is required for AlphaZero-lite training. Install it with 'pip install numpy'."
        )
    vec = np.zeros(81, dtype=np.float32)
    for move, prob in zip(moves, probs):
        index = move[0] * 9 + move[1]
        vec[index] = float(prob)
    return vec


def alphazero_self_play(
    agent: AlphaZeroLiteAgent,
    temperature_schedule: Callable[[int], float],
) -> Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
    if np is None:
        raise ModuleNotFoundError(
            "numpy is required for AlphaZero-lite training. Install it with 'pip install numpy'."
        )
    game = UltimateTicTacToe()
    current_player = "X"
    trajectory: List[Tuple[np.ndarray, np.ndarray, str]] = []
    turn = 0

    while not game.terminal:
        temp = temperature_schedule(turn)
        moves, probs = agent.policy(game, current_player, temperature=temp)
        policy_vec = _policy_vector(moves, probs)
        state_vec = encode_state(game, current_player)
        chosen_index = int(np.random.choice(len(moves), p=probs))
        move = moves[chosen_index]
        trajectory.append((state_vec, policy_vec, current_player))
        game.make_move(current_player, move)
        current_player = _opponent(current_player)
        turn += 1

    results_inputs: List[np.ndarray] = []
    results_policies: List[np.ndarray] = []
    results_values: List[float] = []
    if game.winner is None:
        outcome = "draw"
    else:
        outcome = game.winner
    for state_vec, policy_vec, player in trajectory:
        if outcome == "draw":
            value = 0.0
        elif outcome == player:
            value = 1.0
        else:
            value = -1.0
        results_inputs.append(state_vec)
        results_policies.append(policy_vec)
        results_values.append(value)
    return results_inputs, results_policies, results_values


def train_double_q(
    episodes: int,
    model_path: Path,
    epsilon_start: float,
    epsilon_end: float,
    seed: Optional[int],
) -> DoubleQLearningAgent:
    if seed is not None:
        random.seed(seed)
    agent = DoubleQLearningAgent.load(str(model_path))
    stats = {"X": 0, "O": 0, "draw": 0}

    for episode in range(1, episodes + 1):
        t = (episode - 1) / max(1, episodes - 1)
        epsilon = epsilon_start + (epsilon_end - epsilon_start) * t
        winner, moves = play_episode_double(agent, epsilon)
        stats[winner] = stats.get(winner, 0) + 1
        if episode % max(1, episodes // 10) == 0:
            print(
                f"[double-q] Episode {episode}/{episodes}: winner={winner} moves={moves} epsilon={epsilon:.3f}"
            )

    agent.save(str(model_path))
    print("Double Q-learning training completed. Totals:", stats)
    return agent


def train_sarsa(
    episodes: int,
    model_path: Path,
    epsilon_start: float,
    epsilon_end: float,
    seed: Optional[int],
) -> SARSAgent:
    if seed is not None:
        random.seed(seed)
    agent = SARSAgent.load(str(model_path))
    stats = {"X": 0, "O": 0, "draw": 0}

    for episode in range(1, episodes + 1):
        t = (episode - 1) / max(1, episodes - 1)
        epsilon = epsilon_start + (epsilon_end - epsilon_start) * t
        winner, moves = play_episode_sarsa(agent, epsilon)
        stats[winner] = stats.get(winner, 0) + 1
        if episode % max(1, episodes // 10) == 0:
            print(
                f"[sarsa] Episode {episode}/{episodes}: winner={winner} moves={moves} epsilon={epsilon:.3f}"
            )

    agent.save(str(model_path))
    print("SARSA training completed. Totals:", stats)
    return agent


def train_dyna_q(
    episodes: int,
    model_path: Path,
    epsilon_start: float,
    epsilon_end: float,
    planning_steps: int,
    seed: Optional[int],
) -> DynaQAgent:
    if seed is not None:
        random.seed(seed)
    agent = DynaQAgent.load(str(model_path))
    agent.planning_steps = planning_steps
    stats = {"X": 0, "O": 0, "draw": 0}

    for episode in range(1, episodes + 1):
        t = (episode - 1) / max(1, episodes - 1)
        epsilon = epsilon_start + (epsilon_end - epsilon_start) * t
        winner, moves = play_episode_dyna(agent, epsilon)
        stats[winner] = stats.get(winner, 0) + 1
        if episode % max(1, episodes // 10) == 0:
            print(
                f"[dyna-q] Episode {episode}/{episodes}: winner={winner} moves={moves} epsilon={epsilon:.3f}"
            )

    agent.save(str(model_path))
    print("Dyna-Q training completed. Totals:", stats)
    return agent


def train_alpha_zero_lite(
    episodes: int,
    model_path: Path,
    simulations: int,
    batch_size: int,
    lr: float,
    seed: Optional[int],
) -> AlphaZeroLiteAgent:
    if np is None:
        raise ModuleNotFoundError(
            "numpy is required for AlphaZero-lite training. Install it with 'pip install numpy'."
        )
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    input_dim = len(encode_state(UltimateTicTacToe(), "X"))
    agent = AlphaZeroLiteAgent.load(str(model_path), input_dim)
    agent.num_simulations = simulations

    dataset_inputs: List[np.ndarray] = []
    dataset_policies: List[np.ndarray] = []
    dataset_values: List[float] = []

    def temperature_schedule(turn: int) -> float:
        return 1.0 if turn < 10 else 0.1

    for episode in range(1, episodes + 1):
        inputs, policies, values = alphazero_self_play(agent, temperature_schedule)
        dataset_inputs.extend(inputs)
        dataset_policies.extend(policies)
        dataset_values.extend(values)

        if len(dataset_inputs) >= batch_size:
            idx = np.random.choice(len(dataset_inputs), size=batch_size, replace=False)
            batch_inputs = np.stack([dataset_inputs[i] for i in idx])
            batch_policies = np.stack([dataset_policies[i] for i in idx])
            batch_values = np.array([dataset_values[i] for i in idx], dtype=np.float32)
            loss = agent.network.train_batch(batch_inputs, batch_policies, batch_values, lr=lr)
            if episode % max(1, episodes // 10) == 0:
                print(
                    f"[alpha-zero-lite] Episode {episode}/{episodes}: batch_loss={loss:.4f} "
                    f"buffer={len(dataset_inputs)}"
                )

    agent.save(str(model_path))
    print("AlphaZero-lite training completed. Samples:", len(dataset_inputs))
    return agent


def train_agent(
    episodes: int,
    model_path: Path,
    algorithm: str = "double_q",
    epsilon_start: float = 0.4,
    epsilon_end: float = 0.05,
    planning_steps: int = 20,
    simulations: int = 64,
    batch_size: int = 64,
    lr: float = 1e-2,
    seed: Optional[int] = None,
):
    if algorithm == "double_q":
        return train_double_q(episodes, model_path, epsilon_start, epsilon_end, seed)
    if algorithm == "sarsa":
        return train_sarsa(episodes, model_path, epsilon_start, epsilon_end, seed)
    if algorithm == "dyna_q":
        return train_dyna_q(
            episodes,
            model_path,
            epsilon_start,
            epsilon_end,
            planning_steps,
            seed,
        )
    if algorithm == "alpha_zero_lite":
        return train_alpha_zero_lite(
            episodes,
            model_path,
            simulations,
            batch_size,
            lr,
            seed,
        )
    raise ValueError(f"Unknown algorithm '{algorithm}'")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Ultimate Tic-Tac-Toe agents across multiple algorithms"
    )
    parser.add_argument("--episodes", type=int, default=1000, help="Number of training episodes")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(__file__).resolve().parent / "models" / "ultimate_ttt_q.json",
        help="Where to store the learned parameters",
    )
    parser.add_argument(
        "--algorithm",
        choices=["double_q", "sarsa", "dyna_q", "alpha_zero_lite"],
        default="double_q",
        help="Training algorithm to use",
    )
    parser.add_argument("--epsilon-start", type=float, default=0.5, help="Initial exploration rate")
    parser.add_argument("--epsilon-end", type=float, default=0.05, help="Final exploration rate")
    parser.add_argument(
        "--planning-steps", type=int, default=20, help="Number of model updates per real step"
    )
    parser.add_argument(
        "--simulations", type=int, default=64, help="MCTS simulations for AlphaZero-lite"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size for AlphaZero-lite updates"
    )
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate for AlphaZero-lite")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_agent(
        episodes=args.episodes,
        model_path=args.model_path,
        algorithm=args.algorithm,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        planning_steps=args.planning_steps,
        simulations=args.simulations,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
