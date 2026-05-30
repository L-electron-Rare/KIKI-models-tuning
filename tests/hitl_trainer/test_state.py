from pathlib import Path

from hitl_trainer.state import State, load_state, save_state


def test_default_when_missing(tmp_path):
    s = load_state(tmp_path / "missing.json")
    assert s.last_trained_key is None


def test_round_trip(tmp_path):
    p = tmp_path / "state.json"
    save_state(State(last_trained_key="hardware/dpo/x.jsonl"), p)
    fresh = load_state(p)
    assert fresh.last_trained_key == "hardware/dpo/x.jsonl"


def test_atomic_write(tmp_path):
    p = tmp_path / "state.json"
    save_state(State(last_trained_key="a"), p)
    save_state(State(last_trained_key="b"), p)
    assert load_state(p).last_trained_key == "b"
    # No leftover .tmp
    assert not (tmp_path / "state.json.tmp").exists()
