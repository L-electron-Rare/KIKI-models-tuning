"""Validator: POST to life-core /internal/kicad_validate."""
from __future__ import annotations

import httpx


def validate(json_output: str, *, life_core_url: str, token: str) -> dict:
    headers = {"Authorization": f"Bearer {token}"}
    body = {"json_output": json_output}
    with httpx.Client(timeout=120) as c:
        r = c.post(
            f"{life_core_url}/internal/kicad_validate",
            json=body,
            headers=headers,
        )
    r.raise_for_status()
    return r.json()
