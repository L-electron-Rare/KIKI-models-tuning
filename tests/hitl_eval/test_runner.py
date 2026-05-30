import json

from hitl_eval.runner import run_eval


def test_score_calculation(tmp_path):
    golden = tmp_path / "g.jsonl"
    golden.write_text(
        json.dumps({"id": "a", "prompt": "x"}) + "\n"
        + json.dumps({"id": "b", "prompt": "y"}) + "\n"
    )

    def gen(adapter, base, prompt, max_tokens):
        return f"out-{prompt}"

    def vald(out):
        return {"validated_pass": "y" in out, "violations": []}

    result = run_eval(
        golden_path=golden,
        prod_lora=tmp_path,
        hitl_lora=tmp_path,
        base=tmp_path,
        gen_fn=gen,
        validate_fn=vald,
    )
    assert result["n_prompts"] == 2
    assert result["score_prod"] == 0.5
    assert result["score_hitl"] == 0.5
    assert result["delta"] == 0.0
