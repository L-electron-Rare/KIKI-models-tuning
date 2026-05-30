"""Main daemon loop: poll pending evals + promotions, execute, PATCH."""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from hitl_eval import gate, generator, runner, swap, validator, webhook
from hitl_eval.config import EvalConfig

logger = logging.getLogger("hitl_eval")


def _resolve_latest_hitl(cfg: EvalConfig) -> Path:
    latest = cfg.hitl_output_root / "latest"
    if latest.is_symlink() or latest.exists():
        return latest.resolve()
    subdirs = sorted(p for p in cfg.hitl_output_root.iterdir() if p.is_dir())
    if not subdirs:
        raise RuntimeError("no HITL LoRA dir found")
    return subdirs[-1]


def _resolve_prod(cfg: EvalConfig) -> Path:
    if cfg.curriculum_link.exists() or cfg.curriculum_link.is_symlink():
        return cfg.curriculum_link.resolve()
    return cfg.curriculum_link


def _run_eval_job(cfg: EvalConfig, eval_item: dict) -> None:
    eval_id = eval_item["id"]
    golden_name = eval_item["golden_dataset"]
    golden_path = cfg.golden_datasets_dir / f"{golden_name}.jsonl"

    prod_lora = _resolve_prod(cfg)
    hitl_lora = _resolve_latest_hitl(cfg)

    webhook.claim_eval(
        cfg,
        eval_id,
        prod_lora_path=str(prod_lora),
        hitl_lora_path=str(hitl_lora),
        n_prompts=sum(1 for _ in golden_path.open()),
    )

    def gen(adapter, base, prompt, max_tokens):
        return generator.generate(
            adapter,
            base,
            prompt,
            max_tokens=max_tokens,
            bin_path=cfg.mlx_lm_generate_bin,
        )

    def vald(out):
        return validator.validate(
            out, life_core_url=cfg.life_core_url, token=cfg.token
        )

    try:
        result = runner.run_eval(
            golden_path=golden_path,
            prod_lora=prod_lora,
            hitl_lora=hitl_lora,
            base=cfg.base_model_path,
            gen_fn=gen,
            validate_fn=vald,
            max_tokens=cfg.max_tokens,
        )
        regression = runner.regression_check(
            hitl_lora=hitl_lora,
            base=cfg.base_model_path,
            gen_fn=gen,
            max_tokens=512,
        )
        decision = gate.decide(
            score_prod=result["score_prod"],
            score_hitl=result["score_hitl"],
            regression_check=regression,
        )
        webhook.patch_eval_final(
            cfg,
            eval_id,
            status="success",
            completed_at=datetime.now(timezone.utc).isoformat(),
            n_prompts=result["n_prompts"],
            score_prod=result["score_prod"],
            score_hitl=result["score_hitl"],
            delta=result["delta"],
            regression_check=regression,
            gate_decision=decision,
            per_prompt=result["per_prompt"],
        )
    except Exception as exc:
        logger.exception("eval failed")
        webhook.patch_eval_final(
            cfg,
            eval_id,
            status="failed",
            completed_at=datetime.now(timezone.utc).isoformat(),
            error=str(exc),
        )


def _run_promo_job(cfg: EvalConfig, promo: dict) -> None:
    promo_id = promo["id"]
    promoted_path_str = promo.get("promoted_path") or ""
    previous_path_str = promo.get("previous_path") or ""
    try:
        # Rollback path: queued promotion with previous_path pointing to the
        # old active (path we want to come back to).
        if previous_path_str and Path(promoted_path_str).exists():
            swap.rollback(
                curriculum_link=cfg.curriculum_link,
                previous_path=Path(promoted_path_str),
            )
            webhook.patch_promotion(
                cfg,
                promo_id,
                status="active",
                promoted_path=promoted_path_str,
                previous_path=previous_path_str,
            )
        else:
            hitl_lora = _resolve_latest_hitl(cfg)
            result = swap.promote(
                new_lora_path=hitl_lora,
                curriculum_link=cfg.curriculum_link,
            )
            webhook.patch_promotion(
                cfg,
                promo_id,
                status="active",
                promoted_path=result["promoted_path"],
                previous_path=result["previous_path"],
            )
    except Exception as exc:
        logger.exception("promo failed")
        webhook.patch_promotion(cfg, promo_id, status="failed", error=str(exc))


def tick(cfg: EvalConfig) -> None:
    for ev in webhook.get_pending_evals(cfg):
        _run_eval_job(cfg, ev)
    for pr in webhook.get_pending_promotions(cfg):
        _run_promo_job(cfg, pr)


def main_loop(cfg: EvalConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    while True:
        try:
            tick(cfg)
        except Exception:
            logger.exception("tick failed")
        time.sleep(cfg.poll_seconds)
