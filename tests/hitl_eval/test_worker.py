from types import SimpleNamespace
from unittest.mock import patch

from hitl_eval.worker import tick


def test_tick_processes_pending(tmp_path):
    cfg = SimpleNamespace(
        base_model_path=tmp_path,
        curriculum_link=tmp_path / "cur",
        hitl_output_root=tmp_path,
        target_lora="x",
        golden_datasets_dir=tmp_path,
        domain="kicad",
        token="tk",
        life_core_url="https://api",
        poll_seconds=60,
        mlx_lm_generate_bin="mlx_lm.generate",
        max_tokens=128,
    )
    with patch("hitl_eval.worker.webhook") as wh:
        wh.get_pending_evals.return_value = []
        wh.get_pending_promotions.return_value = []
        tick(cfg)
        assert wh.get_pending_evals.called
        assert wh.get_pending_promotions.called
