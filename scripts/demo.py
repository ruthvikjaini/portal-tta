#!/usr/bin/env python3
"""Print one mixed-arrivals stream under the portal controller."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from portal_tta.cli import demo  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-batches", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    demo(n_batches=args.n_batches, seed=args.seed)


if __name__ == "__main__":
    main()
