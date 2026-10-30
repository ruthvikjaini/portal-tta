"""Deployment metrics.

``harm`` is the paper's central diagnostic, restated for this testbed:
for one batch, harm is how much accuracy the deployed rule loses
relative to deploying no intervention (``base``), floored at zero.
Average accuracy alone hides harm: a rule can win on average while
destroying individual batches. Reporting both the mean and the
worst-batch accuracy, plus mean harm, keeps that failure visible.
"""

from __future__ import annotations

import numpy as np


def accuracy(predicted: np.ndarray, truth: np.ndarray) -> float:
    return float((np.asarray(predicted) == np.asarray(truth)).mean())


def batch_harm(deployed_acc: float, base_acc: float) -> float:
    """Accuracy lost to the deployed rule relative to no intervention."""
    return max(0.0, base_acc - deployed_acc)


def worst_batch(accuracies: list[float]) -> float:
    return float(min(accuracies)) if accuracies else 0.0


def summarise(
    batch_accuracies: list[float],
    batch_harms: list[float],
    labels_used: list[int],
    oracle_accuracies: list[float],
) -> dict[str, float]:
    acc = float(np.mean(batch_accuracies)) if batch_accuracies else 0.0
    oracle = float(np.mean(oracle_accuracies)) if oracle_accuracies else 0.0
    return {
        "accuracy": acc,
        "mean_harm": float(np.mean(batch_harms)) if batch_harms else 0.0,
        "worst_batch_accuracy": worst_batch(batch_accuracies),
        "labels_per_batch": float(np.mean(labels_used)) if labels_used else 0.0,
        "total_labels": float(sum(labels_used)),
        "regret_vs_oracle": oracle - acc,
    }
