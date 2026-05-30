"""mlx_lm.lora subprocess wrapper."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

_LOSS_RE = re.compile(r"Iter\s+(\d+):\s+(?:Train\s+)?[lL]oss\s+([0-9.]+)")


def _convert_jsonl_for_sft(src: Path, dst: Path) -> int:
    n = 0
    with src.open() as fin, dst.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "chosen" in row:
                fout.write(json.dumps({
                    "prompt": row.get("prompt", ""),
                    "completion": row["chosen"],
                }) + "\n")
            elif "completion" in row:
                fout.write(json.dumps({
                    "prompt": row.get("prompt", ""),
                    "completion": row["completion"],
                }) + "\n")
            n += 1
    return n


def _build_argv(*, base: Path, resume_from: Path, data_dir: Path, iters: int, lr: float) -> list[str]:
    return [
        "mlx_lm.lora",
        "--model", str(base),
        "--adapter-path", str(data_dir),
        "--resume-adapter-file", str(resume_from / "adapters.safetensors"),
        "--train",
        "--data", str(data_dir),
        "--iters", str(iters),
        "--learning-rate", f"{lr:.0e}",
        "--batch-size", "1",
        "--lora-layers", "16",
    ]


def make_output_dir(cfg, dataset_key: str) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    name = Path(dataset_key).stem
    return cfg.output_root / f"{ts}_{name}"


def run(*, base: Path, resume_from: Path, dataset: Path, epochs: int, lr: float,
        output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    sft_jsonl = output_dir / "train.jsonl"
    n_samples = _convert_jsonl_for_sft(dataset, sft_jsonl)
    iters = max(epochs * n_samples, 100)
    argv = _build_argv(
        base=base, resume_from=resume_from, data_dir=output_dir,
        iters=iters, lr=lr,
    )
    loss_curve: list[dict] = []
    started_at = datetime.now(timezone.utc).isoformat()
    proc = subprocess.Popen(
        argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        m = _LOSS_RE.search(line)
        if m:
            loss_curve.append({"iter": int(m.group(1)), "loss": float(m.group(2))})
    rc = proc.wait()
    completed_at = datetime.now(timezone.utc).isoformat()
    if rc != 0:
        raise RuntimeError(f"mlx_lm.lora exited {rc}")
    log = {
        "started_at": started_at,
        "completed_at": completed_at,
        "epochs": epochs,
        "lr": lr,
        "iters": iters,
        "loss_curve": loss_curve,
        "dataset": dataset.name,
    }
    (output_dir / "training_log.json").write_text(json.dumps(log, indent=2))
    return {
        "loss_initial": loss_curve[0]["loss"] if loss_curve else None,
        "loss_final": loss_curve[-1]["loss"] if loss_curve else None,
        "loss_curve": loss_curve,
        "output_path": str(output_dir),
        "n_samples": n_samples,
    }


def update_latest_symlink(cfg, output_dir: Path) -> None:
    latest = cfg.output_root / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(output_dir.name)
