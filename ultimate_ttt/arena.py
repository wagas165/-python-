"""Evaluation harness to pit trained Ultimate Tic-Tac-Toe agents against each other."""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

from .ai import (
    AlphaZeroAgent,
    DynaQAgent,
    DoubleQLearningAgent,
    OnPolicySARSAAgent,
    UltimateTTTRLAI,
)
from .game import UltimateTicTacToe

AgentType = UltimateTTTRLAI

AGENT_LOADERS = {
    "q_learning": UltimateTTTRLAI.load,
    "double_q": DoubleQLearningAgent.load,
    "sarsa": OnPolicySARSAAgent.load,
    "dyna_q": DynaQAgent.load,
    "alphazero": AlphaZeroAgent.load,
}


@dataclass(frozen=True)
class AgentSpec:
    label: str
    architecture: Optional[str]
    model_path: Path


def parse_agent_spec(arg: str) -> AgentSpec:
    try:
        left, path_str = arg.split("=", 1)
    except ValueError as exc:  # pragma: no cover - defensive
        raise argparse.ArgumentTypeError(
            "Agent definitions must follow 'name[:architecture]=/path/to/model.json'"
        ) from exc
    if ":" in left:
        label, architecture = left.split(":", 1)
        architecture = architecture or None
    else:
        label, architecture = left, None
    if not label:
        raise argparse.ArgumentTypeError("Agent label cannot be empty")
    return AgentSpec(label=label, architecture=architecture, model_path=Path(path_str))


def infer_architecture(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        algo = data.get("algorithm")
        if isinstance(algo, str):
            return algo
        if "primary" in data and "secondary" in data:
            return "double_q"
    return "q_learning"


def load_agent(spec: AgentSpec) -> AgentType:
    architecture = spec.architecture or infer_architecture(spec.model_path)
    loader = AGENT_LOADERS.get(architecture)
    if loader is None:
        raise ValueError(f"Unknown architecture '{architecture}' for agent '{spec.label}'")
    kwargs = {}
    if architecture == "dyna_q":
        kwargs["planning_steps"] = DynaQAgent.planning_steps
    return loader(str(spec.model_path), **kwargs)


def play_game(agent_x: AgentType, agent_o: AgentType, epsilon: float) -> str:
    game = UltimateTicTacToe()
    current = "X"
    while not game.terminal:
        agent = agent_x if current == "X" else agent_o
        move = agent.select_move(game, current, epsilon=epsilon)
        game.make_move(current, move)
        current = "O" if current == "X" else "X"
    if game.winner is None:
        return "draw"
    return game.winner


def run_round_robin(
    specs: Sequence[AgentSpec],
    games_per_pair: int,
    epsilon: float,
    seed: Optional[int],
) -> Tuple[Dict[str, Dict[str, int]], Dict[Tuple[str, str], Dict[str, int]]]:
    if seed is not None:
        random.seed(seed)
    agents = {spec.label: load_agent(spec) for spec in specs}
    totals: Dict[str, Dict[str, int]] = {
        spec.label: {"wins": 0, "losses": 0, "draws": 0} for spec in specs
    }
    pair_results: Dict[Tuple[str, str], Dict[str, int]] = {}

    for i, spec_a in enumerate(specs):
        for spec_b in specs[i + 1 :]:
            record = {spec_a.label: 0, spec_b.label: 0, "draw": 0}
            for game_index in range(games_per_pair):
                first_x = game_index % 2 == 0
                agent_x = agents[spec_a.label] if first_x else agents[spec_b.label]
                agent_o = agents[spec_b.label] if first_x else agents[spec_a.label]
                result = play_game(agent_x, agent_o, epsilon)
                if result == "draw":
                    totals[spec_a.label]["draws"] += 1
                    totals[spec_b.label]["draws"] += 1
                    record["draw"] += 1
                elif result == "X":
                    winner = spec_a if first_x else spec_b
                    loser = spec_b if first_x else spec_a
                    totals[winner.label]["wins"] += 1
                    totals[loser.label]["losses"] += 1
                    record[winner.label] += 1
                else:  # result == "O"
                    winner = spec_b if first_x else spec_a
                    loser = spec_a if first_x else spec_b
                    totals[winner.label]["wins"] += 1
                    totals[loser.label]["losses"] += 1
                    record[winner.label] += 1
            pair_results[(spec_a.label, spec_b.label)] = record
    return totals, pair_results


def format_totals(totals: Dict[str, Dict[str, int]]) -> str:
    lines = ["Overall results:"]
    for label, stats in totals.items():
        total_games = stats["wins"] + stats["losses"] + stats["draws"]
        win_rate = stats["wins"] / total_games if total_games else 0.0
        lines.append(
            f"  {label}: {stats['wins']}W/{stats['losses']}L/{stats['draws']}D "
            f"(win rate {win_rate:.2%})"
        )
    return "\n".join(lines)


def format_pair_results(pair_results: Dict[Tuple[str, str], Dict[str, int]]) -> str:
    lines = ["Pairwise breakdown:"]
    for (label_a, label_b), record in pair_results.items():
        lines.append(
            f"  {label_a} vs {label_b}: {record[label_a]}-{record[label_b]} "
            f"with {record['draw']} draws"
        )
    return "\n".join(lines)


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a round-robin tournament between trained Ultimate Tic-Tac-Toe agents."
        )
    )
    parser.add_argument(
        "--agent",
        action="append",
        type=parse_agent_spec,
        required=True,
        help="Agent specification: name[:architecture]=/path/to/model.json",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=20,
        help="Number of games per pairing (half played as X, half as O)",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.0,
        help="Exploration rate to use during evaluation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed to control move tie-breaking",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    totals, pair_results = run_round_robin(args.agent, args.games, args.epsilon, args.seed)
    print(format_totals(totals))
    print()
    print(format_pair_results(pair_results))


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
