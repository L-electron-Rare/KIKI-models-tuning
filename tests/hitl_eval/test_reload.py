"""reload_qwen36_adapter cascade: admin endpoint → launchctl → SIGTERM."""
from unittest.mock import MagicMock, patch

from hitl_eval import swap


def test_reload_uses_admin_endpoint_when_available():
    """Returns early if the admin endpoint replies 200."""
    fake_resp = MagicMock(status_code=200)
    fake_client = MagicMock()
    fake_client.__enter__.return_value = fake_client
    fake_client.post.return_value = fake_resp
    with patch("hitl_eval.swap.httpx.Client", return_value=fake_client), \
         patch("hitl_eval.swap.subprocess.run") as run, \
         patch("hitl_eval.swap._kill_qwen36_process") as kill:
        swap.reload_qwen36_adapter()
    run.assert_not_called()
    kill.assert_not_called()


def test_reload_falls_back_to_launchctl_then_sigterm():
    """Admin 404 → launchctl fails → SIGTERM is the last resort."""
    fake_resp = MagicMock(status_code=404)
    fake_client = MagicMock()
    fake_client.__enter__.return_value = fake_client
    fake_client.post.return_value = fake_resp
    rc = MagicMock(returncode=125)  # launchctl Domain does not support
    with patch("hitl_eval.swap.httpx.Client", return_value=fake_client), \
         patch("hitl_eval.swap.subprocess.run", return_value=rc), \
         patch("hitl_eval.swap._kill_qwen36_process", return_value=True) as kill:
        swap.reload_qwen36_adapter()
    kill.assert_called_once_with(9360)


def test_reload_short_circuits_on_launchctl_success():
    """When launchctl succeeds, we don't kill the process."""
    fake_resp = MagicMock(status_code=404)
    fake_client = MagicMock()
    fake_client.__enter__.return_value = fake_client
    fake_client.post.return_value = fake_resp
    rc = MagicMock(returncode=0)
    with patch("hitl_eval.swap.httpx.Client", return_value=fake_client), \
         patch("hitl_eval.swap.subprocess.run", return_value=rc), \
         patch("hitl_eval.swap._kill_qwen36_process") as kill:
        swap.reload_qwen36_adapter()
    kill.assert_not_called()
