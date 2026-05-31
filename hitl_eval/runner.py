"""Eval runner: golden prompts, score, regression check."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

REGRESSION_SAMPLES = {
    "spice": {
        "prompt": "Write a SPICE netlist for a 1k+1uF low-pass RC filter."
    },
    "firmware": {
        "prompt": "Write a minimal Arduino sketch that blinks an LED on pin 13."
    },
}


def _safe_validate(validate_fn: Callable, output: str) -> dict:
    try:
        return validate_fn(output)
    except Exception:
        return {"validated_pass": False, "violations": [{"type": "validator_error"}]}


def run_eval(
    *,
    golden_path: Path,
    prod_lora: Path,
    hitl_lora: Path,
    base: Path,
    gen_fn: Callable,
    validate_fn: Callable,
    max_tokens: int = 2048,
) -> dict:
    prompts = [
        json.loads(line)
        for line in golden_path.read_text().splitlines()
        if line.strip()
    ]
    per_prompt = []
    n_prod_pass = n_hitl_pass = 0
    for p in prompts:
        try:
            prod_out = gen_fn(prod_lora, base, p["prompt"], max_tokens)
        except Exception as exc:
            prod_out = f"ERROR: {exc}"
        try:
            hitl_out = gen_fn(hitl_lora, base, p["prompt"], max_tokens)
        except Exception as exc:
            hitl_out = f"ERROR: {exc}"
        prod_v = _safe_validate(validate_fn, prod_out)
        hitl_v = _safe_validate(validate_fn, hitl_out)
        n_prod_pass += int(prod_v.get("validated_pass", False))
        n_hitl_pass += int(hitl_v.get("validated_pass", False))
        per_prompt.append(
            {
                "id": p["id"],
                "prod_pass": prod_v.get("validated_pass", False),
                "hitl_pass": hitl_v.get("validated_pass", False),
                "prod_violations": prod_v.get("violations", []),
                "hitl_violations": hitl_v.get("violations", []),
            }
        )
    n = len(prompts) or 1
    return {
        "n_prompts": len(prompts),
        "score_prod": n_prod_pass / n,
        "score_hitl": n_hitl_pass / n,
        "delta": (n_hitl_pass - n_prod_pass) / n,
        "per_prompt": per_prompt,
    }


def regression_check(
    *,
    hitl_lora: Path,
    base: Path,
    gen_fn: Callable,
    max_tokens: int = 512,
) -> list[dict]:
    out = []
    for domain, sample in REGRESSION_SAMPLES.items():
        try:
            gen_fn(hitl_lora, base, sample["prompt"], max_tokens)
            ok = True
        except Exception:
            ok = False
        out.append({"domain": domain, "ok": ok})
    return out
