import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultimate_ttt.ai import UltimateTTTRLAI, _move_to_key
from ultimate_ttt.game import UltimateTicTacToe, apply_mapping_to_move
from ultimate_ttt.train import train_agent


def build_asymmetric_game() -> UltimateTicTacToe:
    game = UltimateTicTacToe()
    game.boards[0][0] = "X"
    game.boards[0][8] = "O"
    game.boards[1][4] = "X"
    game.macro_board[1] = "X"
    game.last_move = (1, 4)
    return game


def test_update_maps_moves_into_canonical_orientation():
    agent = UltimateTTTRLAI()
    game = build_asymmetric_game()
    state = agent._state_key(game, "X")
    move = game.available_moves()[0]

    agent.update(state, move, reward=1.0, next_state=None, next_moves=[])

    state_key, mapping = state
    canonical_move = apply_mapping_to_move(move, mapping)
    move_key = _move_to_key(canonical_move)

    assert move_key in agent.q_values[state_key]
    assert agent.q_values[state_key][move_key] != agent.default_q


def test_choose_action_returns_original_orientation_after_update():
    agent = UltimateTTTRLAI()
    game = build_asymmetric_game()
    state = agent._state_key(game, "X")
    move = game.available_moves()[0]

    agent.update(state, move, reward=1.0, next_state=None, next_moves=[])

    moves = game.available_moves()
    chosen_move = agent.choose_action(state, moves, epsilon=0.0)

    assert chosen_move == move


def test_train_agent_updates_q_table(tmp_path):
    model_path = tmp_path / "ultimate_ttt_q.json"
    agent = train_agent(
        episodes=5,
        model_path=model_path,
        epsilon_start=1.0,
        epsilon_end=1.0,
        seed=123,
    )

    assert model_path.exists()
    with open(model_path, "r", encoding="utf-8") as fh:
        saved = json.load(fh)

    q_values = saved.get("q_values") if isinstance(saved, dict) else None
    assert q_values

    has_non_default = False
    for table in q_values.values():
        for value in table.values():
            if value != agent.default_q:
                has_non_default = True
                break
        if has_non_default:
            break

    assert has_non_default
