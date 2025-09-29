import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultimate_ttt.ai import INPUT_SIZE, UltimateTTTRLAI, encode_game_state
from ultimate_ttt.game import UltimateTicTacToe
from ultimate_ttt.train import train_agent


def test_encode_game_state_initial_features():
    game = UltimateTicTacToe()
    encoding = encode_game_state(game, "X")
    assert len(encoding) == INPUT_SIZE

    empty_plane = encoding[81 * 2 : 81 * 3]
    legal_plane = encoding[81 * 3 : 81 * 4]
    focus_plane = encoding[81 * 4 + 9 : 81 * 4 + 18]

    assert all(math.isclose(value, 1.0, rel_tol=1e-6) for value in empty_plane)
    assert all(math.isclose(value, 1.0, rel_tol=1e-6) for value in legal_plane)
    assert all(math.isclose(value, 1.0, rel_tol=1e-6) for value in focus_plane)


def test_select_move_is_legal():
    agent = UltimateTTTRLAI(num_simulations=20)
    game = UltimateTicTacToe()
    move = agent.select_move(game, "X", temperature=0.0)
    assert move in game.available_moves()


def test_train_agent_saves_model(tmp_path):
    model_path = tmp_path / "ultimate_ttt_alpha.json"
    agent = train_agent(
        episodes=2,
        model_path=model_path,
        simulations=20,
        replay_size=200,
        batch_size=16,
        learning_rate=0.05,
        training_steps=1,
        temperature=0.8,
        temperature_moves=3,
        seed=123,
    )

    assert model_path.exists()
    with open(model_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    assert "W1" in data and len(data["W1"][0]) == INPUT_SIZE
    assert isinstance(agent, UltimateTTTRLAI)
