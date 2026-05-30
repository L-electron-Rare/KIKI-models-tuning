"""CLI entrypoint for hitl_eval daemon."""
from __future__ import annotations

import argparse
import logging

from hitl_eval.config import load_config
from hitl_eval.worker import main_loop, tick


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--once",
        action="store_true",
        help="run a single tick (debug)",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    cfg = load_config()
    if args.once:
        tick(cfg)
        return
    main_loop(cfg)


if __name__ == "__main__":
    main()
