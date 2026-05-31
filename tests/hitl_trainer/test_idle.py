from datetime import datetime, timedelta
from unittest.mock import patch

from hitl_trainer.idle import is_idle


def _at(hour):
    return datetime(2026, 5, 30, hour, 0, 0)


def test_outside_hours_not_idle(tmp_path):
    with patch("hitl_trainer.idle.datetime") as dt:
        dt.now.return_value = _at(12)
        with patch("hitl_trainer.idle.os.getloadavg", return_value=(0.5, 0, 0)):
            assert not is_idle(
                load_threshold=4.0, allowed_hours={23, 0, 7},
                idle_min_seconds=1800, state_path=tmp_path / "i.json",
            )


def test_high_load_resets_idle(tmp_path):
    with patch("hitl_trainer.idle.datetime") as dt, \
         patch("hitl_trainer.idle.os.getloadavg", return_value=(10.0, 0, 0)):
        dt.now.return_value = _at(2)
        assert not is_idle(
            load_threshold=4.0, allowed_hours={2},
            idle_min_seconds=1800, state_path=tmp_path / "i.json",
        )


def test_low_load_under_threshold_not_yet(tmp_path):
    # First call records idle_since but returns False.
    with patch("hitl_trainer.idle.datetime") as dt, \
         patch("hitl_trainer.idle.os.getloadavg", return_value=(0.5, 0, 0)):
        t0 = _at(2)
        dt.now.return_value = t0
        dt.fromisoformat.side_effect = lambda s: datetime.fromisoformat(s)
        assert not is_idle(
            load_threshold=4.0, allowed_hours={2},
            idle_min_seconds=1800, state_path=tmp_path / "i.json",
        )
        # Second call 10 min later still false.
        dt.now.return_value = t0 + timedelta(minutes=10)
        assert not is_idle(
            load_threshold=4.0, allowed_hours={2},
            idle_min_seconds=1800, state_path=tmp_path / "i.json",
        )


def test_low_load_after_idle_min_returns_true(tmp_path):
    with patch("hitl_trainer.idle.datetime") as dt, \
         patch("hitl_trainer.idle.os.getloadavg", return_value=(0.5, 0, 0)):
        t0 = _at(2)
        dt.now.return_value = t0
        dt.fromisoformat.side_effect = lambda s: datetime.fromisoformat(s)
        is_idle(
            load_threshold=4.0, allowed_hours={2},
            idle_min_seconds=1800, state_path=tmp_path / "i.json",
        )
        dt.now.return_value = t0 + timedelta(minutes=35)
        assert is_idle(
            load_threshold=4.0, allowed_hours={2},
            idle_min_seconds=1800, state_path=tmp_path / "i.json",
        )
