#!/usr/bin/env python3
"""Validate JSONL dataset format for training."""

import json
import sys
from pathlib import Path


def validate_entry(entry: dict, idx: int) -> list[str]:
    """Validate a single dataset entry."""
    errors = []

    if "messages" not in entry:
        errors.append(f"Line {idx}: missing 'messages' field")
        return errors

    messages = entry["messages"]
    if not isinstance(messages, list) or len(messages) == 0:
        errors.append(f"Line {idx}: 'messages' must be a non-empty list")
        return errors

    for i, msg in enumerate(messages):
        if "role" not in msg:
            errors.append(f"Line {idx}, message {i}: missing 'role'")
        elif msg["role"] not in ("system", "user", "assistant"):
            errors.append(f"Line {idx}, message {i}: invalid role '{msg['role']}'")

        if "content" not in msg:
            errors.append(f"Line {idx}, message {i}: missing 'content'")
        elif not isinstance(msg["content"], str) or len(msg["content"]) == 0:
            errors.append(f"Line {idx}, message {i}: 'content' must be a non-empty string")

    return errors


def validate_file(path: Path) -> tuple[int, int, list[str]]:
    """Validate a JSONL file. Returns (total, valid, errors)."""
    total = 0
    valid = 0
    all_errors = []

    with open(path) as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            total += 1
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                all_errors.append(f"Line {idx}: invalid JSON: {e}")
                continue

            errors = validate_entry(entry, idx)
            if errors:
                all_errors.extend(errors)
            else:
                valid += 1

    return total, valid, all_errors


def main():
    if len(sys.argv) < 2:
        print("Usage: validate_dataset.py <path.jsonl> [path2.jsonl ...]")
        sys.exit(1)

    total_files = 0
    total_errors = 0

    for filepath in sys.argv[1:]:
        path = Path(filepath)
        if not path.exists():
            print(f"SKIP {path} — file not found")
            continue

        total, valid, errors = validate_file(path)
        total_files += 1

        if errors:
            total_errors += len(errors)
            print(f"FAIL {path} — {valid}/{total} valid, {len(errors)} errors:")
            for err in errors[:10]:
                print(f"  {err}")
            if len(errors) > 10:
                print(f"  ... and {len(errors) - 10} more")
        else:
            print(f"PASS {path} — {valid}/{total} valid")

    if total_errors > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
