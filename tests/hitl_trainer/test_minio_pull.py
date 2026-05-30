from unittest.mock import MagicMock

from hitl_trainer import minio_pull


def test_find_latest_unprocessed_picks_newest():
    client = MagicMock()
    client.get_paginator.return_value.paginate.return_value = [{
        "Contents": [
            {"Key": "hardware/dpo/2026-05-30T01-00-00_n1.jsonl"},
            {"Key": "hardware/dpo/2026-05-30T03-00-00_n5.jsonl"},
        ]
    }]
    out = minio_pull.find_latest_unprocessed(
        client, bucket="x", shard_type="hardware", shard_format="dpo",
        since_key=None,
    )
    assert out == "hardware/dpo/2026-05-30T03-00-00_n5.jsonl"


def test_find_latest_unprocessed_respects_watermark():
    client = MagicMock()
    client.get_paginator.return_value.paginate.return_value = [{
        "Contents": [
            {"Key": "hardware/dpo/2026-05-30T01-00-00_n1.jsonl"},
            {"Key": "hardware/dpo/2026-05-30T03-00-00_n5.jsonl"},
        ]
    }]
    out = minio_pull.find_latest_unprocessed(
        client, bucket="x", shard_type="hardware", shard_format="dpo",
        since_key="hardware/dpo/2026-05-30T03-00-00_n5.jsonl",
    )
    assert out is None


def test_find_latest_unprocessed_empty():
    client = MagicMock()
    client.get_paginator.return_value.paginate.return_value = [{"Contents": []}]
    assert minio_pull.find_latest_unprocessed(
        client, bucket="x", shard_type="hardware", shard_format="dpo",
        since_key=None,
    ) is None
