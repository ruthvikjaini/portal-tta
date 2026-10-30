"""Run one policy over one stream, honestly accounting for labels.

Protocol per batch:

1.  Compute every candidate's predictions once, before any label.
2.  The policy chooses a candidate, querying labels only through
    ``labels_for``. The labels it acquires are the only labels the
    learning candidate may train on.
3.  The chosen candidate's frozen predictions are scored on the whole
    batch. The oracle accuracy for the batch is the best candidate
    accuracy on the whole batch, computed from the same frozen
    predictions.
4.  Learning candidates absorb the acquired labels, after deployment,
    so a batch is never scored with labels bought for that batch.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .candidates import make_candidates
from .controller import frozen_predictions
from .data import Batch, StreamConfig, generate_stream
from .metrics import accuracy, batch_harm, summarise
from .policies import PolicySpec, build_policy


@dataclass
class StreamResult:
    policy: str
    scenario: str
    seed: int
    batch_accuracies: list[float] = field(default_factory=list)
    batch_harms: list[float] = field(default_factory=list)
    labels_used: list[int] = field(default_factory=list)
    oracle_accuracies: list[float] = field(default_factory=list)
    choices: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, float]:
        out = summarise(
            self.batch_accuracies,
            self.batch_harms,
            self.labels_used,
            self.oracle_accuracies,
        )
        out["policy"] = self.policy
        out["scenario"] = self.scenario
        out["seed"] = float(self.seed)
        return out


def run_stream(
    spec: PolicySpec, config: StreamConfig, batches: list[Batch] | None = None
) -> StreamResult:
    candidates = make_candidates(spec.portfolio)
    by_name = {c.name: c for c in candidates}
    policy = build_policy(spec)
    stream = batches if batches is not None else generate_stream(config)
    result = StreamResult(policy=spec.name, scenario=config.scenario, seed=config.seed)

    for batch in stream:
        predictions = frozen_predictions(candidates, batch.x)

        def labels_for(indices: list[int], _y=batch.y) -> np.ndarray:
            return _y[np.asarray(indices)]

        decision = policy.decide(batch.index, predictions, labels_for)
        deployed = predictions[decision.choice]
        deployed_acc = accuracy(deployed, batch.y)
        base_acc = accuracy(predictions["base"], batch.y)
        oracle_acc = max(
            accuracy(predictions[name], batch.y) for name in predictions
        )
        result.batch_accuracies.append(deployed_acc)
        result.batch_harms.append(batch_harm(deployed_acc, base_acc))
        result.labels_used.append(decision.n_labels)
        result.oracle_accuracies.append(oracle_acc)
        result.choices.append(decision.choice)

        # Learning only from labels this policy actually acquired.
        if decision.queried:
            idx = np.asarray(decision.queried)
            for candidate in candidates:
                candidate.update(batch.x[idx], batch.y[idx])

    return result


def run_grid(
    scenarios: list[str],
    specs: list[PolicySpec],
    seeds: list[int],
    n_batches: int = 12,
    batch_size: int = 64,
) -> list[dict]:
    """Run every policy on the same streams, per seed.

    Streams are generated once per (scenario, seed) and shared across
    policies, so policy differences are paired, not redraw noise.
    """
    rows: list[dict] = []
    for scenario in scenarios:
        for seed in seeds:
            config = StreamConfig(
                scenario=scenario, seed=seed, n_batches=n_batches, batch_size=batch_size
            )
            batches = generate_stream(config)
            for spec in specs:
                result = run_stream(spec, config, batches=batches)
                rows.append(result.summary())
    return rows
