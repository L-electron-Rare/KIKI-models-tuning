import json
from pathlib import Path

from hitl_trainer import trainer


def test_convert_dpo_to_sft(tmp_path):
    src = tmp_path / "in.jsonl"
    src.write_text(json.dumps({"prompt": "p", "chosen": "c", "rejected": "r"}) + "\n")
    dst = tmp_path / "out.jsonl"
    n = trainer._convert_jsonl_for_sft(src, dst)
    assert n == 1
    out = [json.loads(line) for line in dst.read_text().splitlines()]
    assert out[0] == {"prompt": "p", "completion": "c"}


def test_convert_passes_sft_through(tmp_path):
    src = tmp_path / "in.jsonl"
    src.write_text(json.dumps({"prompt": "p", "completion": "x"}) + "\n")
    dst = tmp_path / "out.jsonl"
    trainer._convert_jsonl_for_sft(src, dst)
    out = [json.loads(line) for line in dst.read_text().splitlines()]
    assert out[0]["completion"] == "x"


def test_build_argv():
    argv = trainer._build_argv(
        base=Path("/m/base"),
        resume_from=Path("/m/cur"),
        data_dir=Path("/tmp/d"),
        iters=200,
        lr=1e-5,
    )
    assert argv[0] == "mlx_lm.lora"
    assert "/m/base" in argv
    assert "--resume-adapter-file" in argv
    i = argv.index("--iters")
    assert argv[i + 1] == "200"
    i = argv.index("--learning-rate")
    assert argv[i + 1] == "1e-05"
