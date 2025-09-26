"""Training utilities for the Ultimate Tic-Tac-Toe reinforcement-learning agents."""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, Optional, Tuple, Type

from .ai import (
    AlphaZeroAgent,
    DynaQAgent,
    DoubleQLearningAgent,
    OnPolicySARSAAgent,
    UltimateTTTRLAI,
)
from .game import Player, UltimateTicTacToe

TabularAgentType = Type[UltimateTTTRLAI]


def play_episode_value(agent: UltimateTTTRLAI, epsilon: float) -> Tuple[str, int]:
    game = UltimateTicTacToe()
    current_player: Player = "X"
    move_count = 0

    while not game.terminal:
        state_key = game.serialize(current_player)
        moves = game.available_moves()
        move = agent.choose_action(state_key, moves, epsilon)
        game.make_move(current_player, move)
        move_count += 1

        if game.terminal:
            if game.winner == current_player:
                agent.update(state_key, move, 1.0, None, [], next_action=None)
                return current_player, move_count
            if game.is_draw:
                agent.update(state_key, move, 0.2, None, [], next_action=None)
                return "draw", move_count
            agent.update(state_key, move, -1.0, None, [], next_action=None)
            return ("O" if current_player == "X" else "X"), move_count

        next_player = "O" if current_player == "X" else "X"
        next_state_key = game.serialize(next_player)
        next_moves = game.available_moves()
        agent.update(
            state_key,
            move,
            0.0,
            next_state_key,
            next_moves,
            next_action=None,
        )
        current_player = next_player

    return "draw", move_count


def play_episode_sarsa(agent: OnPolicySARSAAgent, epsilon: float) -> Tuple[str, int]:
    game = UltimateTicTacToe()
    current_player: Player = "X"
    pending: Dict[Player, Tuple[str, Move]] = {}
    move_count = 0

    while not game.terminal:
        state_key = game.serialize(current_player)
        moves = game.available_moves()
        move = agent.choose_action(state_key, moves, epsilon)
        if current_player in pending:
            prev_state, prev_move = pending[current_player]
            agent.update(
                prev_state,
                prev_move,
                0.0,
                state_key,
                moves,
                next_action=move,
            )
        pending[current_player] = (state_key, move)
        game.make_move(current_player, move)
        move_count += 1
        if game.terminal:
            break
        current_player = "O" if current_player == "X" else "X"

    winner = game.winner
    if winner is None:
        rewards = {"X": 0.0, "O": 0.0}
    else:
        rewards = {
            "X": 1.0 if winner == "X" else -1.0,
            "O": 1.0 if winner == "O" else -1.0,
        }
    for player, (state_key, move) in pending.items():
        agent.update(state_key, move, rewards[player], None, [], next_action=None)

    return (winner if winner is not None else "draw"), move_count


def linear_decay(start: float, end: float, step: float) -> float:
    return start + (end - start) * step


def train_tabular_agent(
    agent_cls: TabularAgentType,
    episodes: int,
    model_path: Path,
    epsilon_start: float,
    epsilon_end: float,
    seed: Optional[int] = None,
    **kwargs,
) -> UltimateTTTRLAI:
    if seed is not None:
        random.seed(seed)

    agent = agent_cls.load(str(model_path), **kwargs)
    stats = {"X": 0, "O": 0, "draw": 0}

    for episode in range(1, episodes + 1):
        t = (episode - 1) / max(1, episodes - 1)
        epsilon = linear_decay(epsilon_start, epsilon_end, t)
        winner, moves = play_episode_value(agent, epsilon)
        stats[winner] += 1
        if episode % max(1, episodes // 10) == 0:
            print(
                f"Episode {episode}/{episodes}: winner={winner} moves={moves} "
                f"epsilon={epsilon:.3f}"
            )

    agent.save(str(model_path))
    print("Training completed. Totals:", stats)
    return agent


def train_agent(
    episodes: int,
    model_path: Path,
    epsilon_start: float = 0.4,
    epsilon_end: float = 0.05,
    seed: Optional[int] = None,
) -> UltimateTTTRLAI:
    """Backward-compatible wrapper for the default Q-learning agent."""

    return train_tabular_agent(
        UltimateTTTRLAI,
        episodes=episodes,
        model_path=model_path,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        seed=seed,
    )


def train_sarsa_agent(
    episodes: int,
    model_path: Path,
    epsilon_start: float,
    epsilon_end: float,
    seed: Optional[int] = None,
) -> OnPolicySARSAAgent:
    if seed is not None:
        random.seed(seed)

    agent = OnPolicySARSAAgent.load(str(model_path))
    stats = {"X": 0, "O": 0, "draw": 0}

    for episode in range(1, episodes + 1):
        t = (episode - 1) / max(1, episodes - 1)
        epsilon = linear_decay(epsilon_start, epsilon_end, t)
        winner, moves = play_episode_sarsa(agent, epsilon)
        stats[winner] += 1
        if episode % max(1, episodes // 10) == 0:
            print(
                f"Episode {episode}/{episodes}: winner={winner} moves={moves} "
                f"epsilon={epsilon:.3f}"
            )

    agent.save(str(model_path))
    print("Training completed. Totals:", stats)
    return agent


def train_alphazero_agent(
    episodes: int,
    model_path: Path,
    simulations: int,
    learning_rate: float,
    temperature: float,
    seed: Optional[int] = None,
) -> AlphaZeroAgent:
    agent = AlphaZeroAgent.load(
        str(model_path), simulations=simulations, learning_rate=learning_rate, seed=seed
    )
    stats = {"X": 0, "O": 0, "draw": 0}

    for episode in range(1, episodes + 1):
        samples, winner, moves = agent.self_play_episode(temperature)
        agent.train_on_samples(samples)
        result_key = winner if winner is not None else "draw"
        stats[result_key] += 1
        if episode % max(1, episodes // 10) == 0:
            print(
                f"Episode {episode}/{episodes}: winner={result_key} moves={moves} "
                f"simulations={agent.num_simulations}"
            )

    agent.save(str(model_path))
    print("Training completed. Totals:", stats)
    return agent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the Ultimate Tic-Tac-Toe reinforcement-learning agents"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1000,
        help="Number of self-play episodes",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(__file__).resolve().parent / "models" / "ultimate_ttt_q.json",
        help="Where to store the learned parameters",
    )
    parser.add_argument(
        "--epsilon-start",
        type=float,
        default=0.5,
        help="Initial exploration rate (tabular agents)",
    )
    parser.add_argument(
        "--epsilon-end",
        type=float,
        default=0.05,
        help="Final exploration rate (tabular agents)",
    )
    parser.add_argument(
        "--architecture",
        choices=["q_learning", "double_q", "sarsa", "dyna_q", "alphazero"],
        default="q_learning",
        help="Select which learning architecture to train",
    )
    parser.add_argument(
        "--planning-steps",
        type=int,
        default=10,
        help="Planning updates per step for Dyna-Q",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=25,
        help="MCTS simulations per move for the AlphaZero-like agent",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
        help="Learning rate for the AlphaZero-like network",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature during AlphaZero self-play",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.architecture == "q_learning":
        train_tabular_agent(
            UltimateTTTRLAI,
            episodes=args.episodes,
            model_path=args.model_path,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
            seed=args.seed,
        )
    elif args.architecture == "double_q":
        train_tabular_agent(
            DoubleQLearningAgent,
            episodes=args.episodes,
            model_path=args.model_path,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
            seed=args.seed,
        )
    elif args.architecture == "dyna_q":
        train_tabular_agent(
            DynaQAgent,
            episodes=args.episodes,
            model_path=args.model_path,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
            seed=args.seed,
            planning_steps=args.planning_steps,
        )
    elif args.architecture == "sarsa":
        train_sarsa_agent(
            episodes=args.episodes,
            model_path=args.model_path,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
            seed=args.seed,
        )
    else:
        train_alphazero_agent(
            episodes=args.episodes,
            model_path=args.model_path,
            simulations=args.simulations,
            learning_rate=args.learning_rate,
            temperature=args.temperature,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
