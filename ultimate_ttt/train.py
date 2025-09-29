"""Training utilities for the AlphaZero-style Ultimate Tic-Tac-Toe agent."""
from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from .ai import UltimateTTTRLAI, encode_game_state
from .game import Player, UltimateTicTacToe


@dataclass
class ReplaySample:
    state: List[float]
    policy: List[float]
    value: float


class ReplayBuffer:
    """Simple FIFO replay buffer for self-play experiences."""

    def __init__(self, capacity: int = 5000) -> None:
        self.capacity = capacity
        self._samples: List[ReplaySample] = []

    def add_game(self, samples: Sequence[ReplaySample]) -> None:
        self._samples.extend(samples)
        overflow = len(self._samples) - self.capacity
        if overflow > 0:
            self._samples = self._samples[overflow:]

    def __len__(self) -> int:
        return len(self._samples)

    def sample(self, batch_size: int) -> Tuple[List[List[float]], List[List[float]], List[float]]:
        if not self._samples:
            raise ValueError("Replay buffer is empty")
        count = len(self._samples)
        if count >= batch_size:
            indices = random.sample(range(count), batch_size)
        else:
            indices = [random.randrange(count) for _ in range(batch_size)]
        states = [self._samples[i].state[:] for i in indices]
        policies = [self._samples[i].policy[:] for i in indices]
        values = [self._samples[i].value for i in indices]
        return states, policies, values


def self_play_episode(
    agent: UltimateTTTRLAI,
    simulations: int,
    temperature_moves: int = 8,
    base_temperature: float = 1.0,
) -> Tuple[List[ReplaySample], Optional[Player]]:
    """Run a single self-play game using MCTS guidance."""

    game = UltimateTicTacToe()
    player: Player = "X"
    move_count = 0
    history: List[Tuple[List[float], List[float], Player]] = []

    while not game.terminal:
        agent.set_num_simulations(simulations)
        temperature = base_temperature if move_count < temperature_moves else 0.0
        add_noise = move_count == 0
        state_vec = encode_game_state(game, player)
        move, policy = agent.policy(
            game,
            player,
            temperature=max(temperature, 0.0),
            add_noise=add_noise,
        )
        history.append((state_vec, policy[:], player))
        game.make_move(player, move)
        player = "O" if player == "X" else "X"
        move_count += 1

    samples: List[ReplaySample] = []
    for state_vec, policy, perspective in history:
        if game.winner is None:
            outcome = 0.0
        elif game.winner == perspective:
            outcome = 1.0
        else:
            outcome = -1.0
        samples.append(ReplaySample(state_vec, policy, outcome))

    return samples, game.winner


def train_agent(
    episodes: int,
    model_path: Path,
    simulations: int = 160,
    replay_size: int = 5000,
    batch_size: int = 64,
    learning_rate: float = 0.01,
    training_steps: int = 4,
    temperature: float = 1.0,
    temperature_moves: int = 8,
    seed: Optional[int] = None,
) -> UltimateTTTRLAI:
    """Train the AlphaZero-style agent via self-play."""

    if seed is not None:
        random.seed(seed)

    agent = UltimateTTTRLAI.load(str(model_path), num_simulations=simulations, seed=seed)
    buffer = ReplayBuffer(capacity=replay_size)

    stats = {"X": 0, "O": 0, "draw": 0}

    for episode in range(1, episodes + 1):
        samples, winner = self_play_episode(
            agent,
            simulations=simulations,
            temperature_moves=temperature_moves,
            base_temperature=temperature,
        )
        buffer.add_game(samples)

        if winner is None:
            stats["draw"] += 1
        else:
            stats[winner] += 1

        if len(buffer) >= batch_size:
            losses: List[float] = []
            for _ in range(training_steps):
                states, policies, values = buffer.sample(batch_size)
                loss = agent.network.train_step(states, policies, values, learning_rate)
                losses.append(loss)
            avg_loss = sum(losses) / len(losses)
        else:
            avg_loss = float("nan")

        print(
            f"Episode {episode}/{episodes} - winner: {winner or 'draw'} "
            f"buffer={len(buffer)} loss={avg_loss:.4f}"
        )

    agent.save(str(model_path))
    print("Training completed. Totals:", stats)
    return agent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the AlphaZero Ultimate TTT agent")
    parser.add_argument("--episodes", type=int, default=100, help="Number of self-play episodes")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(__file__).resolve().parent / "models" / "ultimate_ttt_alpha.json",
        help="Where to store the learned network weights",
    )
    parser.add_argument("--simulations", type=int, default=160, help="MCTS simulations per move")
    parser.add_argument("--replay-size", type=int, default=5000, help="Replay buffer capacity")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
        help="Learning rate for the network",
    )
    parser.add_argument(
        "--training-steps",
        type=int,
        default=4,
        help="Gradient steps per episode",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Initial sampling temperature for move selection",
    )
    parser.add_argument(
        "--temperature-moves",
        type=int,
        default=8,
        help="Number of moves to apply temperature before going greedy",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_agent(
        episodes=args.episodes,
        model_path=args.model_path,
        simulations=args.simulations,
        replay_size=args.replay_size,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        training_steps=args.training_steps,
        temperature=args.temperature,
        temperature_moves=args.temperature_moves,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
