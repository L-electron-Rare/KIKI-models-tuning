from unittest.mock import MagicMock, patch

import pytest

from hitl_trainer import webhook


class _Cfg:
    token = "tk"
    life_core_url = "https://api"
    base_model_path = "/base"
    target_lora = "qwen36-kicad-hitl"
    curriculum_lora_path = "/cur"
    shard_format = "dpo"
    epochs = 2
    lr_reduced = 1e-5


def _client_mock():
    """Build a MagicMock that mimics httpx.Client used as a context manager."""
    fake = MagicMock()
    # `with httpx.Client(...) as client:` → client is __enter__'s return.
    fake.__enter__.return_value = fake
    fake.__exit__.return_value = False
    return fake


def test_post_create_returns_id():
    fake = _client_mock()
    fake.post.return_value.status_code = 200
    fake.post.return_value.json.return_value = {"id": "abc"}
    with patch("hitl_trainer.webhook.httpx.Client", return_value=fake):
        tid = webhook.post_create(
            _Cfg(), dataset_key="x", output_path="/o", n_samples=3,
        )
    assert tid == "abc"


def test_post_create_retries_then_succeeds():
    fake = _client_mock()
    resp_ok = MagicMock(status_code=200)
    resp_ok.json.return_value = {"id": "x"}
    resp_fail = MagicMock(status_code=503)
    fake.post.side_effect = [resp_fail, resp_fail, resp_ok]
    with patch("hitl_trainer.webhook.httpx.Client", return_value=fake), \
         patch("hitl_trainer.webhook.time.sleep"):
        tid = webhook.post_create(
            _Cfg(), dataset_key="x", output_path="/o", n_samples=3,
        )
    assert tid == "x"
    assert fake.post.call_count == 3


def test_patch_finish_calls_patch():
    fake = _client_mock()
    fake.patch.return_value.status_code = 200
    with patch("hitl_trainer.webhook.httpx.Client", return_value=fake):
        webhook.patch_finish(
            _Cfg(), "abc",
            status="success", loss_initial=0.42, loss_final=0.31,
        )
    fake.patch.assert_called_once()
