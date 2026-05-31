"""Life-core webhook (POST /trainings, PATCH /trainings/<id>) with retry."""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

import httpx

logger = logging.getLogger(__name__)

_MAX_TRIES = 3


def _retry_call(method, url, **kwargs):
    last = None
    for attempt in range(_MAX_TRIES):
        try:
            resp = method(url, **kwargs)
            if resp.status_code < 500:
                return resp
            last = resp
        except httpx.HTTPError as exc:
            last = exc
        time.sleep(2 ** attempt)
    if isinstance(last, httpx.Response):
        return last
    raise last  # type: ignore[misc]


def post_create(cfg, *, dataset_key: str, output_path: str, n_samples: int) -> str:
    body = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "base_model": str(cfg.base_model_path),
        "target_lora": cfg.target_lora,
        "resume_from": str(cfg.curriculum_lora_path),
        "dataset_key": dataset_key,
        "dataset_format": cfg.shard_format,
        "n_samples": n_samples,
        "epochs": cfg.epochs,
        "learning_rate": cfg.lr_reduced,
        "output_path": output_path,
        "host": "macstudio",
    }
    headers = {"Authorization": f"Bearer {cfg.token}"}
    with httpx.Client(timeout=30) as client:
        resp = _retry_call(client.post, f"{cfg.life_core_url}/trainings",
                           json=body, headers=headers)
    resp.raise_for_status()
    return resp.json()["id"]


def patch_finish(cfg, training_id: str, *, status: str, **fields) -> None:
    body = {
        "status": status,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        **fields,
    }
    headers = {"Authorization": f"Bearer {cfg.token}"}
    with httpx.Client(timeout=30) as client:
        _retry_call(client.patch, f"{cfg.life_core_url}/trainings/{training_id}",
                    json=body, headers=headers)
