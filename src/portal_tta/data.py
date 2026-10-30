"""Synthetic deployment streams for test-time intervention control.

The testbed is deliberately small and fully specified: class-conditional
Gaussian blobs in eight dimensions, classified by nearest prototype. A
"domain" is an affine corruption (per-feature gain plus shift) applied to
every example in a stream. Arrival order matters because batch-statistic
interventions are computed from the batch itself: a mixed batch carries
the domain shift in its statistics, while a class-block batch carries
mostly one class, so its statistics carry the class signal instead.

Nothing here claims to be image data. The testbed is an analogue of the
batch-statistics failure mode: it reproduces the qualitative pattern --
adaptation rescues shifted mixed batches and damages class-block batches
-- with numbers that are measured on this testbed only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

N_CLASSES = 4
DIM = 8
SIGMA = 0.7

# Prototypes are fixed across every stream and seed so that candidate
# comparisons are comparisons of rules, not of draws.
_PROTOTYPE_SEED = 20261030


def default_prototypes() -> np.ndarray:
    rng = np.random.default_rng(_PROTOTYPE_SEED)
    mu = rng.normal(size=(N_CLASSES, DIM))
    mu = mu / np.linalg.norm(mu, axis=1, keepdims=True) * 2.0
    return mu


def train_stats(prototypes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and per-feature std of a large clean sample (the "source" stats)."""
    rng = np.random.default_rng(999)
    n = 20000
    y = rng.integers(0, N_CLASSES, n)
    x = prototypes[y] + rng.normal(scale=SIGMA, size=(n, DIM))
    return x.mean(axis=0), x.std(axis=0)


@dataclass(frozen=True)
class Batch:
    x: np.ndarray
    y: np.ndarray
    index: int
    arrival: str  # "mixed" or "block"
    domain: str  # "clean" or "corrupt"


@dataclass
class StreamConfig:
    scenario: str = "corrupt_mixed"
    n_batches: int = 12
    batch_size: int = 64
    seed: int = 0
    drift_batch: int = 6

    def __post_init__(self) -> None:
        known = {
            "clean",
            "corrupt_mixed",
            "corrupt_block",
            "clean_block",
            "mixed_arrivals",
            "drift",
        }
        if self.scenario not in known:
            raise ValueError(f"unknown scenario: {self.scenario!r}")
        if self.n_batches < 1 or self.batch_size < 8:
            raise ValueError("n_batches >= 1 and batch_size >= 8 required")
        if not 0 < self.drift_batch < self.n_batches and self.scenario == "drift":
            raise ValueError("drift_batch must lie inside the stream")


def _draw_corruption(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    gain = np.exp(rng.normal(scale=0.9, size=DIM))
    shift = rng.normal(scale=1.0, size=DIM)
    return gain, shift


def generate_stream(config: StreamConfig) -> list[Batch]:
    """Generate one stream deterministically from ``config``.

    The corruption (gain, shift) is drawn once per stream from the stream
    seed, so two runs with the same config produce identical streams.
    """
    rng = np.random.default_rng(config.seed)
    prototypes = default_prototypes()
    gain, shift = _draw_corruption(rng)
    gain2, shift2 = _draw_corruption(rng)
    batches: list[Batch] = []
    for i in range(config.n_batches):
        if config.scenario == "clean":
            g, s, arrival, domain = np.ones(DIM), np.zeros(DIM), "mixed", "clean"
        elif config.scenario == "corrupt_mixed":
            g, s, arrival, domain = gain, shift, "mixed", "corrupt"
        elif config.scenario == "corrupt_block":
            g, s, arrival, domain = gain, shift, "block", "corrupt"
        elif config.scenario == "clean_block":
            g, s, arrival, domain = np.ones(DIM), np.zeros(DIM), "block", "clean"
        elif config.scenario == "mixed_arrivals":
            arrival = "block" if i % 3 == 2 else "mixed"
            g, s, domain = gain, shift, "corrupt"
        elif config.scenario == "drift":
            if i < config.drift_batch:
                g, s, domain = np.ones(DIM), np.zeros(DIM), "clean"
            else:
                g, s, domain = gain2, shift2, "corrupt"
            arrival = "mixed"
        else:  # pragma: no cover - guarded by StreamConfig
            raise ValueError(config.scenario)

        batch_rng = np.random.default_rng(1000 * config.seed + i)
        if arrival == "block":
            # Class-block arrival: the whole batch is one class, so the
            # batch statistics carry that class, not the domain. The
            # class rotates from batch to batch.
            y = np.full(config.batch_size, i % N_CLASSES, dtype=int)
        else:
            y = batch_rng.integers(0, N_CLASSES, config.batch_size)
        x = prototypes[y] + batch_rng.normal(
            scale=SIGMA, size=(config.batch_size, DIM)
        )
        x = x * g + s
        batches.append(Batch(x=x, y=y, index=i, arrival=arrival, domain=domain))
    return batches


@dataclass
class StreamSpec:
    """A named stream plus the corruption it used, for reporting."""

    config: StreamConfig
    batches: list[Batch] = field(default_factory=list)
