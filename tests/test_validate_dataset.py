"""Tests for dataset validator."""

import json
import tempfile
from pathlib import Path

from scripts.validate_dataset import validate_entry, validate_file


def test_valid_entry():
    entry = {"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]}
    errors = validate_entry(entry, 1)
    assert errors == []


def test_missing_messages():
    errors = validate_entry({}, 1)
    assert len(errors) == 1
    assert "missing 'messages'" in errors[0]


def test_invalid_role():
    entry = {"messages": [{"role": "invalid", "content": "Hello"}]}
    errors = validate_entry(entry, 1)
    assert len(errors) == 1
    assert "invalid role" in errors[0]


def test_empty_content():
    entry = {"messages": [{"role": "user", "content": ""}]}
    errors = validate_entry(entry, 1)
    assert len(errors) == 1


def test_validate_file():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps({"messages": [{"role": "user", "content": "test"}, {"role": "assistant", "content": "ok"}]}) + "\n")
        f.write(json.dumps({"messages": [{"role": "user", "content": "q2"}, {"role": "assistant", "content": "a2"}]}) + "\n")
        f.flush()
        total, valid, errors = validate_file(Path(f.name))
    assert total == 2
    assert valid == 2
    assert errors == []
