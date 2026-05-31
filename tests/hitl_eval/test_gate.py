import pytest
from hitl_eval.gate import decide


@pytest.mark.parametrize(
    "score_prod,score_hitl,reg,expected",
    [
        (0.5, 0.75, [{"ok": True}], "go"),
        (0.5, 0.5, [{"ok": True}], "go"),
        (0.5, 0.4, [{"ok": True}], "no_go"),
        (0.5, 0.75, [{"ok": False}], "no_go"),
    ],
)
def test_decide(score_prod, score_hitl, reg, expected):
    assert decide(
        score_prod=score_prod,
        score_hitl=score_hitl,
        regression_check=reg,
    ) == expected
