#!/usr/bin/env python3
"""Check that every headline number in the README matches measured.json.

Usage: python scripts/crossfoot.py
Exits non-zero if any headline value, formatted to three decimals, is
absent from README.md. Run after regenerating results.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    measured = json.loads((ROOT / "results" / "measured.json").read_text())
    readme = (ROOT / "README.md").read_text()
    missing = []
    for key, value in measured["headline"].items():
        text = f"{value:.3f}"
        if text not in readme:
            missing.append(f"{key} = {text}")
    if missing:
        print("README is missing measured values:")
        for line in missing:
            print("  " + line)
        return 1
    print(f"cross-foot OK: {len(measured['headline'])} headline values present in README")
    return 0


if __name__ == "__main__":
    sys.exit(main())
