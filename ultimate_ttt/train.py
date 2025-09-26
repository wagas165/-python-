"""Training utilities for the Ultimate Tic-Tac-Toe reinforcement-learning agent."""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Optional, Tuple

from .ai import UltimateTTTRLAI
from .game import UltimateTicTacToe


def play_episode(agent: UltimateTTTRLAI, epsilon: float) -> Tuple[str, int]:
    game = UltimateTicTacToe()
    current_player = "X"
    move_count = 0

    while not game.terminal:
        state_key = agent._state_key(game, current_player)
        moves = game.available_moves()
        move = agent.choose_action(state_key, moves, epsilon)
        game.make_move(current_player, move)
        move_count += 1

        if game.terminal:
            if game.winner == current_player:
                reward = 1.0
                agent.update(state_key, move, reward, None, [])
                return current_player, move_count
            if game.is_draw:
                agent.update(state_key, move, 0.2, None, [])
                return "draw", move_count
            agent.update(state_key, move, -1.0, None, [])
            return (
                ("O" if current_player == "X" else "X"),
                move_count,
            )

        next_player = "O" if current_player == "X" else "X"
        next_state_key = agent._state_key(game, next_player)
        next_moves = game.available_moves()
        agent.update(state_key, move, 0.0, next_state_key, next_moves)
        current_player = next_player

    return "draw", move_count


def linear_decay(start: float, end: float, step: float) -> float:
    return start + (end - start) * step


def train_agent(
    episodes: int,
    model_path: Path,
    epsilon_start: float = 0.4,
    epsilon_end: float = 0.05,
    seed: Optional[int] = None,
) -> UltimateTTTRLAI:
    if seed is not None:
        random.seed(seed)

    agent = UltimateTTTRLAI.load(str(model_path))
    stats = {"X": 0, "O": 0, "draw": 0}

    for episode in range(1, episodes + 1):
        t = (episode - 1) / max(1, episodes - 1)
        epsilon = linear_decay(epsilon_start, epsilon_end, t)
        winner, moves = play_episode(agent, epsilon)
        stats[winner] += 1
        if episode % max(1, episodes // 10) == 0:
            print(
                f"Episode {episode}/{episodes}: winner={winner} moves={moves} "
                f"epsilon={epsilon:.3f}"
            )

    agent.save(str(model_path))
    print("Training completed. Totals:", stats)
    return agent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Ultimate TTT RL agent")
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
        help="Where to store the learned Q-values",
    )
    parser.add_argument(
        "--epsilon-start",
        type=float,
        default=0.5,
        help="Initial exploration rate",
    )
    parser.add_argument(
        "--epsilon-end",
        type=float,
        default=0.05,
        help="Final exploration rate",
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
    train_agent(
        episodes=args.episodes,
        model_path=args.model_path,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
