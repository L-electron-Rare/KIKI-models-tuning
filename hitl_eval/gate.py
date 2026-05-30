"""Gate decision: go / no_go."""
from __future__ import annotations


def decide(
    *,
    score_prod: float,
    score_hitl: float,
    regression_check: list[dict],
) -> str:
    if any(not r.get("ok") for r in regression_check):
        return "no_go"
    if score_hitl + 1e-9 < score_prod:
        return "no_go"
    return "go"
