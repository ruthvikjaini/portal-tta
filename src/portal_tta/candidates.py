"""Candidate prediction rules.

Each candidate is a prediction rule whose predictions for a batch are
fixed before any label for that batch arrives. That freezing is the
property the controller relies on: one acquired label then evaluates
every candidate on the same input, so comparisons are paired, not
confounded by which inputs each rule happened to see.

Candidates
----------
base
    Nearest prototype in the raw feature space. No adaptation.
batch_norm
    Recentre and rescale each batch by its own mean and standard
    deviation, then compare against source statistics. This is the
    batch-statistics analogue of updating normalisation statistics at
    test time. It rescues shifted mixed batches and fails when a batch
    is mostly one class, because the batch statistics then encode the
    class.
shrink
    Batch statistics shrunk halfway toward the source statistics.
    A deliberately conservative intervention: part of the rescue, much
    less of the failure.
label_adapted
    Nearest prototype with prototypes moved toward the class means of
    labels acquired so far (exponential moving average). It only ever
    learns from labels the controller actually paid for, and its
    prototypes are frozen within a batch.
batch_norm_first8
    Batch statistics estimated from the first eight examples of the
    batch only. A weak, noisy candidate included only for the
    portfolio-expansion experiment: a controller that is not protected
    against extra candidates should get worse when it is added.
"""

from __future__ import annotations

import numpy as np

from .data import N_CLASSES, default_prototypes, train_stats


class Candidate:
    name: str

    def predict(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def update(self, x: np.ndarray, y: np.ndarray) -> None:
        """Absorb acquired labels. No-op for rules that do not learn."""


def _nearest(x: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    dist = ((x[:, None, :] - prototypes[None, :, :]) ** 2).sum(axis=-1)
    return dist.argmin(axis=1)


class BaseCandidate(Candidate):
    name = "base"

    def __init__(self) -> None:
        self.prototypes = default_prototypes()

    def predict(self, x: np.ndarray) -> np.ndarray:
        return _nearest(x, self.prototypes)


class BatchNormCandidate(Candidate):
    name = "batch_norm"

    def __init__(self) -> None:
        self.prototypes = default_prototypes()
        self.train_mean, self.train_std = train_stats(self.prototypes)

    def _standardise(self, x: np.ndarray) -> np.ndarray:
        batch_mean = x.mean(axis=0)
        batch_std = x.std(axis=0)
        batch_std = np.maximum(batch_std, 1e-6)
        return (x - batch_mean) / batch_std * self.train_std + self.train_mean

    def predict(self, x: np.ndarray) -> np.ndarray:
        return _nearest(self._standardise(x), self.prototypes)


class ShrinkCandidate(BatchNormCandidate):
    name = "shrink"

    def __init__(self, weight: float = 0.5) -> None:
        super().__init__()
        if not 0.0 <= weight <= 1.0:
            raise ValueError("weight must be in [0, 1]")
        self.weight = weight

    def _standardise(self, x: np.ndarray) -> np.ndarray:
        batch_mean = x.mean(axis=0)
        batch_std = np.maximum(x.std(axis=0), 1e-6)
        mean = self.weight * batch_mean + (1.0 - self.weight) * self.train_mean
        std = self.weight * batch_std + (1.0 - self.weight) * self.train_std
        return (x - mean) / std * self.train_std + self.train_mean


class LabelAdaptedCandidate(Candidate):
    name = "label_adapted"

    def __init__(self, alpha: float = 0.5) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self.prototypes = default_prototypes()
        self._seen = np.zeros(N_CLASSES, dtype=int)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return _nearest(x, self.prototypes)

    def update(self, x: np.ndarray, y: np.ndarray) -> None:
        if len(x) == 0:
            return
        for k in range(N_CLASSES):
            mask = y == k
            if not mask.any():
                continue
            class_mean = x[mask].mean(axis=0)
            self.prototypes[k] = (
                (1.0 - self.alpha) * self.prototypes[k] + self.alpha * class_mean
            )
            self._seen[k] += int(mask.sum())


class BatchNormFirst8Candidate(BatchNormCandidate):
    """Weak candidate: batch statistics from the first 8 examples only."""

    name = "batch_norm_first8"

    def _standardise(self, x: np.ndarray) -> np.ndarray:
        head = x[:8]
        batch_mean = head.mean(axis=0)
        batch_std = np.maximum(head.std(axis=0), 1e-6)
        return (x - batch_mean) / batch_std * self.train_std + self.train_mean


def make_candidates(portfolio: str = "core") -> list[Candidate]:
    """Fresh candidate instances. ``portfolio`` is ``core`` or ``expanded``."""
    candidates: list[Candidate] = [
        BaseCandidate(),
        BatchNormCandidate(),
        ShrinkCandidate(),
        LabelAdaptedCandidate(),
    ]
    if portfolio == "expanded":
        candidates.append(BatchNormFirst8Candidate())
    elif portfolio != "core":
        raise ValueError(f"unknown portfolio: {portfolio!r}")
    return candidates


def candidate_names(portfolio: str = "core") -> list[str]:
    return [c.name for c in make_candidates(portfolio)]
