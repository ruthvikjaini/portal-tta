"""Policies: fixed rules, the oracle, and controller wrappers.

A policy decides, for one batch, which candidate to deploy. It receives
the frozen predictions for the batch and a ``labels_for`` callable, and
returns a :class:`~portal_tta.controller.Decision`. Fixed policies never
call ``labels_for``; the oracle calls it for every input; controllers
call it only for inputs they choose to query.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .controller import Decision, NaiveController, PortalConfig, PortalController


class FixedPolicy:
    """Always deploy one named candidate; never query a label."""

    def __init__(self, choice: str) -> None:
        self.choice = choice

    def decide(self, batch_index, predictions, labels_for) -> Decision:
        return Decision(
            choice=self.choice, n_labels=0, correct={}, total_labelled=0
        )


class OraclePolicy:
    """Deploy the per-batch best candidate, using every label.

    An upper bound on what any controller could achieve on the batch,
    and a statement of its cost: it pays for a label on every input.
    """

    def decide(self, batch_index, predictions, labels_for) -> Decision:
        names = list(predictions.keys())
        n = len(next(iter(predictions.values())))
        labels = np.asarray(labels_for(list(range(n))))
        correct = {
            name: int((predictions[name] == labels).sum()) for name in names
        }
        choice = max(names, key=lambda nm: correct[nm])
        return Decision(
            choice=choice,
            n_labels=n,
            correct=correct,
            total_labelled=n,
            queried=list(range(n)),
        )


@dataclass
class PolicySpec:
    name: str
    portfolio: str = "core"
    kind: str = "portal"  # fixed | oracle | portal | naive
    fixed_choice: str = "base"
    portal_config: PortalConfig | None = None
    max_labels: int = 16


def build_policy(spec: PolicySpec):
    if spec.kind == "fixed":
        return FixedPolicy(spec.fixed_choice)
    if spec.kind == "oracle":
        return OraclePolicy()
    if spec.kind == "portal":
        return PortalController(spec.portal_config or PortalConfig())
    if spec.kind == "naive":
        return NaiveController(max_labels=spec.max_labels, seed=0)
    raise ValueError(f"unknown policy kind: {spec.kind!r}")


def default_policies() -> list[PolicySpec]:
    """The policy set used by the headline experiment grid."""
    return [
        PolicySpec(name="always_base", kind="fixed", fixed_choice="base"),
        PolicySpec(name="always_batch_norm", kind="fixed", fixed_choice="batch_norm"),
        PolicySpec(name="always_shrink", kind="fixed", fixed_choice="shrink"),
        PolicySpec(name="naive", kind="naive", max_labels=16),
        PolicySpec(
            name="portal",
            kind="portal",
            portal_config=PortalConfig(max_labels=16, seed=0),
        ),
        PolicySpec(
            name="portal_no_refresh",
            kind="portal",
            portal_config=PortalConfig(max_labels=16, seed=0, probe_every=0),
        ),
        PolicySpec(
            name="portal_expanded",
            kind="portal",
            portfolio="expanded",
            portal_config=PortalConfig(max_labels=16, seed=0),
        ),
        PolicySpec(
            name="naive_expanded",
            kind="naive",
            portfolio="expanded",
            max_labels=16,
        ),
        PolicySpec(name="oracle", kind="oracle"),
    ]
