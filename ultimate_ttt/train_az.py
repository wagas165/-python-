"""AlphaZero-style training loop for Ultimate Tic-Tac-Toe."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
import yaml
from torch import nn

from .arena import Arena
from .model import PolicyValueNet
from .mcts import MCTS
from .selfplay import SelfPlaySample, augment_samples, play_game
from .utils import save_checkpoint, select_device, set_random_seeds

__all__ = ["main", "train"]


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._data: List[SelfPlaySample] = []

    def extend(self, samples: Iterable[SelfPlaySample]) -> None:
        self._data.extend(samples)
        if len(self._data) > self.capacity:
            self._data = self._data[-self.capacity :]

    def sample(self, batch_size: int) -> List[SelfPlaySample]:
        if len(self._data) < batch_size:
            raise ValueError("Not enough samples in replay buffer")
        indices = np.random.choice(len(self._data), size=batch_size, replace=False)
        return [self._data[int(i)] for i in indices]

    def __len__(self) -> int:
        return len(self._data)


def _loss_fn(
    model: PolicyValueNet,
    batch: Sequence[SelfPlaySample],
    device: torch.device,
) -> torch.Tensor:
    planes = torch.from_numpy(np.stack([sample.planes for sample in batch])).to(device)
    legal_mask = torch.from_numpy(
        np.stack([sample.legal_mask for sample in batch]).astype(np.bool_)
    ).to(device)
    target_policy = torch.from_numpy(np.stack([sample.policy for sample in batch])).to(device)
    target_value = torch.from_numpy(np.array([sample.value_x for sample in batch], dtype=np.float32)).to(device)

    logits, value = model(planes, legal_mask=legal_mask)
    log_probs = torch.log_softmax(logits, dim=-1)
    policy_loss = -(target_policy * log_probs).sum(dim=-1).mean()
    value_loss = nn.functional.mse_loss(value.squeeze(-1), target_value)
    return policy_loss + value_loss


def train(config: Dict[str, Dict[str, float]]) -> None:
    training_cfg = config.get("training", {})
    selfplay_cfg = config.get("selfplay", {})
    replay_cfg = config.get("replay", {})
    eval_cfg = config.get("evaluation", {})

    seed = int(training_cfg.get("seed", 0))
    set_random_seeds(seed)
    device = select_device(training_cfg.get("use_gpu", True))

    model = PolicyValueNet(
        channels=int(training_cfg.get("channels", 64)),
        num_blocks=int(training_cfg.get("res_blocks", 8)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg.get("learning_rate", 3e-4)),
        weight_decay=float(training_cfg.get("weight_decay", 1e-4)),
    )

    baseline = PolicyValueNet(
        channels=int(training_cfg.get("channels", 64)),
        num_blocks=int(training_cfg.get("res_blocks", 8)),
    ).to(device)
    baseline.load_state_dict(model.state_dict())

    buffer = ReplayBuffer(int(replay_cfg.get("capacity", 200_000)))
    cycles = int(training_cfg.get("cycles", 1))
    games_per_cycle = int(selfplay_cfg.get("games_per_cycle", 50))
    batch_size = int(training_cfg.get("batch_size", 256))
    train_steps = int(training_cfg.get("steps_per_cycle", 1000))

    checkpoint_path = Path(training_cfg.get("checkpoint", "ultimate_ttt/models/az_checkpoint.pt"))

    for cycle in range(cycles):
        for _ in range(games_per_cycle):
            mcts = MCTS(
                model=model,
                num_simulations=int(selfplay_cfg.get("simulations", 400)),
                c_puct=float(selfplay_cfg.get("c_puct", 2.0)),
                dirichlet_epsilon=float(selfplay_cfg.get("dirichlet_epsilon", 0.25)),
                dirichlet_alpha=float(selfplay_cfg.get("dirichlet_alpha", 0.3)),
                device=device,
            )
            samples = play_game(
                mcts,
                temperature_moves=int(selfplay_cfg.get("temperature_moves", 16)),
            )
            if selfplay_cfg.get("augment", True):
                samples = augment_samples(samples)
            buffer.extend(samples)

        if len(buffer) < batch_size:
            continue

        for _ in range(train_steps):
            batch = buffer.sample(batch_size)
            optimizer.zero_grad()
            loss = _loss_fn(
                model,
                batch,
                device,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(training_cfg.get("grad_clip", 1.0)))
            optimizer.step()

        if eval_cfg:
            interval = int(eval_cfg.get("interval", 1))
            if (cycle + 1) % interval == 0:
                arena = Arena(
                    challenger=model,
                    baseline=baseline,
                    mcts_simulations=int(eval_cfg.get("simulations", 200)),
                    c_puct=float(eval_cfg.get("c_puct", 2.0)),
                )
                result = arena.play_matches(int(eval_cfg.get("matches", 200)))
                if result.win_rate >= float(eval_cfg.get("threshold", 0.55)):
                    baseline.load_state_dict(model.state_dict())

        save_checkpoint(checkpoint_path, model, optimizer, step=cycle + 1)

    torch.save(model.state_dict(), checkpoint_path.with_suffix(".weights.pt"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an AlphaZero-style agent for Ultimate Tic-Tac-Toe")
    parser.add_argument("--config", type=Path, default=Path("ultimate_ttt/config.yaml"))
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    train(config)


if __name__ == "__main__":
    main()
