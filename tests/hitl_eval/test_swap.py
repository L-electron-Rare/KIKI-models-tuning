from unittest.mock import patch

from hitl_eval.swap import promote, rollback


def test_promote_creates_symlink(tmp_path):
    new = tmp_path / "new_lora"
    new.mkdir()
    link = tmp_path / "curriculum"
    link.mkdir()
    with patch("hitl_eval.swap.reload_qwen36_adapter"):
        result = promote(new_lora_path=new, curriculum_link=link)
    assert link.is_symlink()
    assert link.resolve() == new
    assert result["promoted_path"] == str(new)


def test_rollback_restores_previous(tmp_path):
    new = tmp_path / "new"
    new.mkdir()
    prev = tmp_path / "prev"
    prev.mkdir()
    link = tmp_path / "curriculum"
    link.symlink_to(new)
    with patch("hitl_eval.swap.reload_qwen36_adapter"):
        rollback(curriculum_link=link, previous_path=prev)
    assert link.resolve() == prev
