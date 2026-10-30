"""The PORTAL-style controller and a naive controller for comparison.

Controller principle (from Zhang et al., arXiv:2610.37658, abstract and
reported design; this is an independent implementation, not the paper's
code):

1.  Predictions are frozen before feedback. Every acquired label
    evaluates all candidates on the same input, so the evidence is a
    set of paired comparisons.
2.  Queries are directed at inputs where a challenger could still
    overtake the current leader -- inputs where the leader and a
    challenger disagree. Inputs where every candidate agrees carry no
    information about which rule to deploy and are never queried.
3.  Querying stops early when the remaining labels cannot change the
    choice (the leader's margin exceeds the maximum possible swing),
    or when even the most optimistic outcome for a challenger is not
    worth another label at the configured label cost.
4.  Periodic probe batches refresh stale evidence: when conditions
    change, old paired comparisons can point at the wrong rule, so a
    probe batch discards them and gathers fresh evidence.

Ties are broken in favour of ``base`` (no intervention). Deploying an
intervention needs positive evidence, not merely the absence of evidence
against it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .candidates import Candidate


@dataclass
class Decision:
    choice: str
    n_labels: int
    correct: dict[str, int]
    total_labelled: int
    queried: list[int] = field(default_factory=list)


@dataclass
class PortalConfig:
    max_labels: int = 16
    min_labels: int = 3
    label_cost: float = 0.01
    prefer: str = "base"
    probe_every: int = 4
    refresh_probe: int = 4
    seed: int = 0

    def __post_init__(self) -> None:
        if self.max_labels < 1:
            raise ValueError("max_labels must be >= 1")
        if not 0 <= self.min_labels <= self.max_labels:
            raise ValueError("min_labels must lie in [0, max_labels]")
        if self.label_cost < 0:
            raise ValueError("label_cost must be >= 0")
        if self.probe_every < 0 or self.refresh_probe < 0:
            raise ValueError("probe settings must be >= 0")


class PortalController:
    """Shared-comparison controller with early stopping and refresh."""

    def __init__(self, config: PortalConfig | None = None) -> None:
        self.config = config or PortalConfig()
        self._rng = np.random.default_rng(self.config.seed)
        self._correct: dict[str, int] = {}
        self._total = 0
        self._generation = 0

    # -- evidence ---------------------------------------------------------
    @property
    def generation(self) -> int:
        return self._generation

    def _leader(self, names: list[str]) -> str:
        if self._total == 0:
            return self.config.prefer if self.config.prefer in names else names[0]
        best = max(
            names,
            key=lambda n: (
                self._correct.get(n, 0),
                1 if n == self.config.prefer else 0,
            ),
        )
        return best

    def _stopping_met(
        self,
        names: list[str],
        predictions: dict[str, np.ndarray],
        remaining_pool: list[int],
        n_queried: int,
    ) -> bool:
        """Stop when no challenger can still overtake the leader, or when
        even the best possible outcome is not worth another label.

        Only inputs where a challenger disagrees with the leader can
        change a paired comparison, so the maximum swing for a
        challenger is its remaining disagreement count -- not the size
        of the whole pool. This is what makes the bound tight when most
        inputs are ones every candidate already agrees on.
        """
        if n_queried < self.config.min_labels:
            return False
        if self._total == 0:
            return False
        leader = self._leader(names)
        lead_correct = self._correct.get(leader, 0)
        leader_pred = predictions[leader]
        for name in names:
            if name == leader:
                continue
            challenger_correct = self._correct.get(name, 0)
            discordant = sum(
                1 for idx in remaining_pool if predictions[name][idx] != leader_pred[idx]
            )
            if discordant == 0:
                continue
            # Impossibility: even winning every remaining disagreement
            # cannot close the gap.
            if challenger_correct + discordant <= lead_correct:
                continue
            # Value: the largest accuracy-margin gain the challenger
            # could still achieve, if it won every remaining
            # disagreement and the leader gained nothing on them.
            optimistic_gain = (
                (challenger_correct + discordant) - lead_correct
            ) / (self._total + discordant)
            if optimistic_gain > self.config.label_cost:
                return False
        return True

    # -- batch lifecycle --------------------------------------------------
    def begin_batch(self, batch_index: int) -> None:
        """Start a batch; probe batches discard stale evidence."""
        if (
            self.config.probe_every > 0
            and batch_index > 0
            and batch_index % self.config.probe_every == 0
        ):
            self._correct = {}
            self._total = 0
            self._generation += 1

    def is_probe_batch(self, batch_index: int) -> bool:
        return (
            self.config.probe_every > 0
            and batch_index > 0
            and batch_index % self.config.probe_every == 0
        )

    # -- selection --------------------------------------------------------
    def select(
        self,
        predictions: dict[str, np.ndarray],
        labels_for: object,
        batch_index: int = -1,
    ) -> Decision:
        """Choose a candidate for one batch.

        ``predictions`` maps candidate name to frozen predictions for the
        batch. ``labels_for`` is a callable taking a list of indices and
        returning their true labels. It is only ever called with indices
        this controller chose to query.
        """
        names = list(predictions.keys())
        n = len(next(iter(predictions.values())))
        min_labels = self.config.min_labels
        if batch_index >= 0 and self.is_probe_batch(batch_index):
            min_labels = max(min_labels, self.config.refresh_probe)

        queried: list[int] = []
        remaining_pool = list(range(n))
        while len(queried) < self.config.max_labels and remaining_pool:
            if self._stopping_met(
                names, predictions, remaining_pool, n_queried=len(queried)
            ):
                break
            leader = self._leader(names)
            leader_pred = predictions[leader]
            # Score each unqueried input by how many challengers
            # disagree with the leader on it. Zero-disagreement inputs
            # cannot change the choice and are never queried.
            scores = np.zeros(len(remaining_pool), dtype=int)
            for name in names:
                if name == leader:
                    continue
                pred = predictions[name]
                for j, idx in enumerate(remaining_pool):
                    if pred[idx] != leader_pred[idx]:
                        scores[j] += 1
            best_score = int(scores.max())
            if best_score == 0:
                break
            best_positions = np.flatnonzero(scores == best_score)
            pick_pos = int(self._rng.choice(best_positions))
            idx = remaining_pool.pop(pick_pos)
            queried.append(idx)
            true_label = int(labels_for([idx])[0])
            self._total += 1
            for name in names:
                if int(predictions[name][idx]) == true_label:
                    self._correct[name] = self._correct.get(name, 0) + 1
            # Keep every candidate present in the counts, even at zero.
            for name in names:
                self._correct.setdefault(name, 0)

        choice = self._leader(names)
        return Decision(
            choice=choice,
            n_labels=len(queried),
            correct=dict(self._correct),
            total_labelled=self._total,
            queried=queried,
        )

    def decide(self, batch_index: int, predictions, labels_for) -> Decision:
        self.begin_batch(batch_index)
        return self.select(predictions, labels_for, batch_index=batch_index)


class NaiveController:
    """Random-query controller with no stopping rule and no refresh.

    The comparison baseline in the style of controllers that acquire a
    fixed number of random labels and deploy the argmax: it pays the
    full label budget on every batch, including batches where the choice
    is already settled, and it can be hurt by adding a weak candidate
    to the portfolio.
    """

    def __init__(self, max_labels: int = 16, seed: int = 0) -> None:
        if max_labels < 1:
            raise ValueError("max_labels must be >= 1")
        self.max_labels = max_labels
        self._rng = np.random.default_rng(seed)
        self._correct: dict[str, int] = {}
        self._total = 0

    def decide(self, batch_index: int, predictions, labels_for) -> Decision:
        names = list(predictions.keys())
        n = len(next(iter(predictions.values())))
        k = min(self.max_labels, n)
        queried = sorted(self._rng.choice(n, size=k, replace=False).tolist())
        for idx in queried:
            true_label = int(labels_for([idx])[0])
            self._total += 1
            for name in names:
                if int(predictions[name][idx]) == true_label:
                    self._correct[name] = self._correct.get(name, 0) + 1
            for name in names:
                self._correct.setdefault(name, 0)
        choice = max(names, key=lambda nm: self._correct.get(nm, 0))
        return Decision(
            choice=choice,
            n_labels=len(queried),
            correct=dict(self._correct),
            total_labelled=self._total,
            queried=queried,
        )


def frozen_predictions(
    candidates: list[Candidate], x: np.ndarray
) -> dict[str, np.ndarray]:
    """Compute every candidate's predictions once, before any labels."""
    return {c.name: c.predict(x) for c in candidates}
