"""Regression: promoting a *directory* (not a pre-existing symlink) must
record `previous_path` as the `.prev` backup, so that rollback restores
the original directory rather than creating a self-link."""
from unittest.mock import patch

from hitl_eval.swap import promote, rollback


def _mk_lora(d, files=("adapters.safetensors", "adapter_config.json")):
    d.mkdir(parents=True, exist_ok=True)
    for f in files:
        (d / f).write_text("x")


def test_promote_with_directory_curriculum_records_prev(tmp_path):
    """When curriculum_link is a real directory, previous_path must point
    to the `.prev` backup (not to the curriculum_link itself)."""
    new = tmp_path / "hitl"
    _mk_lora(new)
    link = tmp_path / "curriculum"
    _mk_lora(link)  # real directory, not a symlink

    with patch("hitl_eval.swap.reload_qwen36_adapter"):
        result = promote(new_lora_path=new, curriculum_link=link)

    backup = tmp_path / "curriculum.prev"
    assert backup.is_dir(), "original directory should be backed up to .prev"
    assert link.is_symlink() and link.resolve() == new
    assert result["previous_path"] == str(backup)
    # And the backup is NOT a path that resolves to curriculum_link.
    assert backup.resolve() != link.resolve()


def test_rollback_after_directory_promote_restores_original(tmp_path):
    """End-to-end: promote a directory, then rollback should bring the
    original directory back at the curriculum_link path."""
    new = tmp_path / "hitl"
    _mk_lora(new)
    link = tmp_path / "curriculum"
    _mk_lora(link, files=("adapters.safetensors", "original_marker"))

    with patch("hitl_eval.swap.reload_qwen36_adapter"):
        result = promote(new_lora_path=new, curriculum_link=link)
        prev_path = result["previous_path"]
        assert prev_path is not None
        rollback(curriculum_link=link, previous_path=type(link)(prev_path))

    # link must point to the previous content (which still has original_marker).
    resolved = link.resolve()
    assert (resolved / "original_marker").exists(), (
        "rollback should restore the original directory, not self-link"
    )


def test_rollback_refuses_self_link(tmp_path):
    """Defensive: if for any reason the previous_path resolves to the
    same path as curriculum_link, rollback must refuse rather than
    create a self-link."""
    link = tmp_path / "curriculum"
    _mk_lora(link)
    # Pass curriculum_link as its own previous_path -> resolves equal.
    import pytest
    with pytest.raises(RuntimeError, match="self-rollback"):
        with patch("hitl_eval.swap.reload_qwen36_adapter"):
            rollback(curriculum_link=link, previous_path=link)
