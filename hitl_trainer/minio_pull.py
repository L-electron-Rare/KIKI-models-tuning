"""MinIO list + fetch helpers."""
from __future__ import annotations

from pathlib import Path


def find_latest_unprocessed(
    client, *, bucket: str, shard_type: str, shard_format: str, since_key: str | None,
) -> str | None:
    prefix = f"{shard_type}/{shard_format}/"
    paginator = client.get_paginator("list_objects_v2")
    keys: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []) or []:
            keys.append(obj["Key"])
    keys.sort(reverse=True)
    for k in keys:
        if since_key is None or k > since_key:
            return k
    return None


def fetch(client, *, bucket: str, key: str, tmp_dir: Path) -> Path:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out = tmp_dir / Path(key).name
    client.download_file(bucket, key, str(out))
    return out


def make_client(*, endpoint: str, access_key: str, secret_key: str):
    import boto3
    from botocore.client import Config
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(signature_version="s3v4"),
    )
