"""Eval daemon webhook (claim, finalize eval / promote / rollback)."""
from __future__ import annotations

import time

import httpx


def _retry(method, url, **kw):
    last = None
    for i in range(3):
        try:
            r = method(url, **kw)
            if r.status_code < 500:
                return r
            last = r
        except httpx.HTTPError as e:
            last = e
        time.sleep(2 ** i)
    if isinstance(last, httpx.Response):
        return last
    raise last  # type: ignore[misc]


def _headers(cfg) -> dict:
    return {"Authorization": f"Bearer {cfg.token}"}


def claim_eval(cfg, eval_id, *, prod_lora_path, hitl_lora_path, n_prompts):
    body = {
        "prod_lora_path": prod_lora_path,
        "hitl_lora_path": hitl_lora_path,
        "n_prompts": n_prompts,
    }
    with httpx.Client(timeout=30) as c:
        return _retry(
            c.post,
            f"{cfg.life_core_url}/eval_runs/{eval_id}/claim",
            json=body,
            headers=_headers(cfg),
        )


def patch_eval_final(cfg, eval_id, **fields):
    with httpx.Client(timeout=30) as c:
        return _retry(
            c.patch,
            f"{cfg.life_core_url}/eval_runs/{eval_id}",
            json=fields,
            headers=_headers(cfg),
        )


def patch_promotion(cfg, promo_id, **fields):
    with httpx.Client(timeout=30) as c:
        return _retry(
            c.patch,
            f"{cfg.life_core_url}/promotions/{promo_id}",
            json=fields,
            headers=_headers(cfg),
        )


def get_pending_evals(cfg) -> list[dict]:
    with httpx.Client(timeout=30) as c:
        r = c.get(
            f"{cfg.life_core_url}/eval_runs/pending", headers=_headers(cfg)
        )
    r.raise_for_status()
    return r.json().get("items", [])


def get_pending_promotions(cfg) -> list[dict]:
    with httpx.Client(timeout=30) as c:
        r = c.get(
            f"{cfg.life_core_url}/promotions/pending", headers=_headers(cfg)
        )
    r.raise_for_status()
    return r.json().get("items", [])
