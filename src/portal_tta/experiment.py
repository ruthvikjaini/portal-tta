"""Headline experiment: aggregate the grid into measured results."""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from .benchmark import run_grid
from .policies import default_policies

SCENARIOS = [
    "clean",
    "corrupt_mixed",
    "corrupt_block",
    "clean_block",
    "mixed_arrivals",
    "drift",
]
SEEDS = list(range(8))


def aggregate(rows: list[dict]) -> dict:
    """Mean over seeds for each (scenario, policy), plus headline diffs."""
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["scenario"], row["policy"])].append(row)

    table: dict[str, dict[str, dict[str, float]]] = {}
    for (scenario, policy), group in sorted(grouped.items()):
        table.setdefault(scenario, {})[policy] = {
            key: float(np.mean([r[key] for r in group]))
            for key in (
                "accuracy",
                "mean_harm",
                "worst_batch_accuracy",
                "labels_per_batch",
                "total_labels",
                "regret_vs_oracle",
            )
        }

    def get(scenario: str, policy: str, key: str) -> float:
        return table[scenario][policy][key]

    headline = {
        "mixed_arrivals_portal_accuracy": get("mixed_arrivals", "portal", "accuracy"),
        "mixed_arrivals_always_batch_norm_accuracy": get(
            "mixed_arrivals", "always_batch_norm", "accuracy"
        ),
        "mixed_arrivals_portal_mean_harm": get("mixed_arrivals", "portal", "mean_harm"),
        "mixed_arrivals_always_batch_norm_mean_harm": get(
            "mixed_arrivals", "always_batch_norm", "mean_harm"
        ),
        "mixed_arrivals_portal_worst_batch": get(
            "mixed_arrivals", "portal", "worst_batch_accuracy"
        ),
        "mixed_arrivals_portal_no_refresh_accuracy": get(
            "mixed_arrivals", "portal_no_refresh", "accuracy"
        ),
        "mixed_arrivals_always_batch_norm_worst_batch": get(
            "mixed_arrivals", "always_batch_norm", "worst_batch_accuracy"
        ),
        "corrupt_mixed_portal_accuracy": get("corrupt_mixed", "portal", "accuracy"),
        "corrupt_mixed_always_base_accuracy": get(
            "corrupt_mixed", "always_base", "accuracy"
        ),
        "corrupt_block_batch_norm_accuracy": get(
            "corrupt_block", "always_batch_norm", "accuracy"
        ),
        "corrupt_block_batch_norm_mean_harm": get(
            "corrupt_block", "always_batch_norm", "mean_harm"
        ),
        "corrupt_block_portal_accuracy": get("corrupt_block", "portal", "accuracy"),
        "corrupt_block_portal_mean_harm": get("corrupt_block", "portal", "mean_harm"),
        "corrupt_block_base_accuracy": get("corrupt_block", "always_base", "accuracy"),
        "clean_portal_labels_per_batch": get("clean", "portal", "labels_per_batch"),
        "clean_naive_labels_per_batch": get("clean", "naive", "labels_per_batch"),
        "clean_portal_accuracy": get("clean", "portal", "accuracy"),
        "clean_always_batch_norm_accuracy": get(
            "clean", "always_batch_norm", "accuracy"
        ),
        "drift_portal_accuracy": get("drift", "portal", "accuracy"),
        "drift_portal_no_refresh_accuracy": get(
            "drift", "portal_no_refresh", "accuracy"
        ),
        "drift_portal_labels_per_batch": get("drift", "portal", "labels_per_batch"),
        "expanded_portal_accuracy_mixed_arrivals": get(
            "mixed_arrivals", "portal_expanded", "accuracy"
        ),
        "expanded_naive_accuracy_mixed_arrivals": get(
            "mixed_arrivals", "naive_expanded", "accuracy"
        ),
        "core_naive_accuracy_mixed_arrivals": get(
            "mixed_arrivals", "naive", "accuracy"
        ),
    }
    return {
        "scenarios": SCENARIOS,
        "seeds": SEEDS,
        "n_batches": 12,
        "batch_size": 64,
        "table": table,
        "headline": headline,
    }


def run_experiment() -> tuple[list[dict], dict]:
    rows = run_grid(SCENARIOS, default_policies(), SEEDS)
    return rows, aggregate(rows)


def write_results(rows: list[dict], measured: dict, experiment_path: str, measured_path: str) -> None:
    with open(experiment_path, "w", encoding="utf-8") as fh:
        json.dump({"rows": rows}, fh, indent=2, sort_keys=True)
        fh.write("\n")
    with open(measured_path, "w", encoding="utf-8") as fh:
        json.dump(measured, fh, indent=2, sort_keys=True)
        fh.write("\n")
