"""AlphaZero-style training loop for Ultimate Tic-Tac-Toe."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np

from .model import NetworkConfig, PolicyValueNet
from .selfplay import SelfPlayConfig, play_game
from .utils import ReplayBuffer, ReplaySample, set_random_seed

try:  # pragma: no cover - optional torch dependency
    import torch
except Exception:  # pragma: no cover
    torch = None


def _stack_samples(samples: list[ReplaySample]):
    planes = np.stack([sample.planes for sample in samples]).astype(np.float32)
    policies = np.stack([sample.policy for sample in samples]).astype(np.float32)
    masks = np.stack([sample.legal_mask for sample in samples]).astype(np.float32)
    values = np.array([sample.value for sample in samples], dtype=np.float32)
    return planes, policies, masks, values


def train_alphazero(
    episodes: int,
    model_path: Path,
    buffer_capacity: int = 200_000,
    batch_size: int = 256,
    learning_rate: float = 3e-4,
    device: str = "cpu",
    seed: Optional[int] = None,
) -> PolicyValueNet:
    if torch is None:
        raise ImportError("PyTorch is required for AlphaZero training")
    if seed is not None:
        set_random_seed(seed)
    dev = torch.device(device)
    model = PolicyValueNet(config=NetworkConfig())
    model.to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    replay = ReplayBuffer(buffer_capacity)
    selfplay_config = SelfPlayConfig()

    for episode in range(1, episodes + 1):
        samples = play_game(model, selfplay_config)
        for sample in samples:
            replay.append(sample)

        if len(replay) >= batch_size:
            batch = replay.sample_batch(batch_size)
            planes, policies, masks, values = _stack_samples(batch)
            planes_t = torch.from_numpy(planes).to(dev)
            masks_t = torch.from_numpy(masks).to(dev)
            policies_t = torch.from_numpy(policies).to(dev)
            values_t = torch.from_numpy(values).to(dev)

            logits, predicted = model(planes_t, masks_t)
            log_probs = torch.log_softmax(logits, dim=1)
            policy_loss = -(policies_t * log_probs).sum(dim=1).mean()
            value_loss = torch.mean((predicted.squeeze(-1) - values_t) ** 2)
            loss = policy_loss + value_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if episode % 10 == 0:
            save_checkpoint(model, optimizer, model_path)
            print(f"Episode {episode}: buffer_size={len(replay)}")

    save_checkpoint(model, optimizer, model_path)
    return model


def save_checkpoint(model: PolicyValueNet, optimizer, path: Path) -> None:
    if torch is None:  # pragma: no cover
        raise ImportError("PyTorch is required to save checkpoints")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AlphaZero training for Ultimate TTT")
    parser.add_argument("--episodes", type=int, default=100, help="Number of self-play episodes")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(__file__).resolve().parent / "models" / "alphazero_policy.pt",
        help="Checkpoint destination",
    )
    parser.add_argument("--buffer", type=int, default=200_000, help="Replay buffer capacity")
    parser.add_argument("--batch", type=int, default=256, help="Training batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_alphazero(
        episodes=args.episodes,
        model_path=args.model_path,
        buffer_capacity=args.buffer,
        batch_size=args.batch,
        learning_rate=args.lr,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
