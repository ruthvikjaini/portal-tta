#!/usr/bin/env python3
"""Run the full experiment grid and write results files."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from portal_tta.cli import benchmark  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-out", default="results/experiment.json")
    parser.add_argument("--measured-out", default="results/measured.json")
    args = parser.parse_args()
    benchmark(args.experiment_out, args.measured_out)


if __name__ == "__main__":
    main()
