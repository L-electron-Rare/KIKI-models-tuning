"""Main loop: poll MinIO + idle + train + webhook."""
from __future__ import annotations

import argparse
import logging
import sys
import time

from hitl_trainer import idle, minio_pull, trainer, webhook
from hitl_trainer.config import load_config
from hitl_trainer.state import load_state, save_state

logger = logging.getLogger("hitl_trainer")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _count_lines(path) -> int:
    return sum(1 for _ in open(path))


def _cycle(cfg, state) -> bool:
    """Returns True if a training was launched."""
    client = minio_pull.make_client(
        endpoint=cfg.minio_endpoint,
        access_key=cfg.minio_access_key,
        secret_key=cfg.minio_secret_key,
    )
    latest = minio_pull.find_latest_unprocessed(
        client, bucket=cfg.minio_bucket,
        shard_type=cfg.shard_type, shard_format=cfg.shard_format,
        since_key=state.last_trained_key,
    )
    if latest is None:
        logger.info("no new dataset")
        return False
    if not idle.is_idle(
        load_threshold=cfg.load_threshold,
        allowed_hours=cfg.allowed_hours,
        idle_min_seconds=cfg.idle_min_seconds,
        state_path=cfg.idle_state_path,
    ):
        logger.info("not idle")
        return False
    jsonl = minio_pull.fetch(client, bucket=cfg.minio_bucket, key=latest,
                             tmp_dir=cfg.tmp_dir)
    output_dir = trainer.make_output_dir(cfg, latest)
    tr_id = webhook.post_create(
        cfg, dataset_key=latest, output_path=str(output_dir),
        n_samples=_count_lines(jsonl),
    )
    try:
        result = trainer.run(
            base=cfg.base_model_path,
            resume_from=cfg.curriculum_lora_path,
            dataset=jsonl,
            epochs=cfg.epochs,
            lr=cfg.lr_reduced,
            output_dir=output_dir,
        )
        webhook.patch_finish(cfg, tr_id, status="success", **{
            k: v for k, v in result.items()
            if k in ("loss_initial", "loss_final", "loss_curve", "output_path")
        })
        trainer.update_latest_symlink(cfg, output_dir)
        state.last_trained_key = latest
        save_state(state, cfg.state_path)
    except Exception as exc:
        logger.exception("training failed")
        webhook.patch_finish(cfg, tr_id, status="failed", error=str(exc))
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="single cycle, no loop")
    args = parser.parse_args()

    cfg = load_config()
    state = load_state(cfg.state_path)

    if args.once:
        _cycle(cfg, state)
        return

    while True:
        try:
            _cycle(cfg, state)
        except Exception:
            logger.exception("cycle failed")
        time.sleep(cfg.poll_seconds)


if __name__ == "__main__":
    main()
