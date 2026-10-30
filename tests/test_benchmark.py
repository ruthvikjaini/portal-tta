import json
from pathlib import Path

import numpy as np

from portal_tta.benchmark import run_grid, run_stream
from portal_tta.data import StreamConfig, generate_stream
from portal_tta.policies import PolicySpec, default_policies

ROOT = Path(__file__).resolve().parents[1]


def test_grid_is_deterministic():
    specs = default_policies()[:5]
    rows_a = run_grid(["clean", "mixed_arrivals"], specs, [0, 1], n_batches=4)
    rows_b = run_grid(["clean", "mixed_arrivals"], specs, [0, 1], n_batches=4)
    assert rows_a == rows_b


def test_portal_beats_always_batch_norm_on_mixed_arrivals():
    config = StreamConfig(scenario="mixed_arrivals", seed=0)
    batches = generate_stream(config)
    portal = run_stream(
        PolicySpec(name="portal", kind="portal"), config, batches=batches
    )
    always_bn = run_stream(
        PolicySpec(name="always_batch_norm", kind="fixed", fixed_choice="batch_norm"),
        config,
        batches=batches,
    )
    portal_summary = portal.summary()
    bn_summary = always_bn.summary()
    assert portal_summary["accuracy"] > bn_summary["accuracy"] + 0.05
    assert portal_summary["mean_harm"] < bn_summary["mean_harm"]


def test_portal_beats_always_base_on_corrupt_mixed():
    config = StreamConfig(scenario="corrupt_mixed", seed=0)
    batches = generate_stream(config)
    portal = run_stream(
        PolicySpec(name="portal", kind="portal"), config, batches=batches
    )
    base = run_stream(
        PolicySpec(name="always_base", kind="fixed", fixed_choice="base"),
        config,
        batches=batches,
    )
    assert portal.summary()["accuracy"] > base.summary()["accuracy"] + 0.05


def test_portal_avoids_batch_norm_harm_on_corrupt_block():
    config = StreamConfig(scenario="corrupt_block", seed=0)
    batches = generate_stream(config)
    portal = run_stream(
        PolicySpec(name="portal", kind="portal"), config, batches=batches
    )
    always_bn = run_stream(
        PolicySpec(name="always_batch_norm", kind="fixed", fixed_choice="batch_norm"),
        config,
        batches=batches,
    )
    assert portal.summary()["mean_harm"] < 0.10
    assert always_bn.summary()["mean_harm"] > 0.40


def test_learning_candidate_only_uses_acquired_labels():
    # The adapted candidate must not change on batches where the fixed
    # policy acquires no labels.
    config = StreamConfig(scenario="clean", seed=0)
    batches = generate_stream(config)
    result = run_stream(
        PolicySpec(name="always_base", kind="fixed", fixed_choice="base"),
        config,
        batches=batches,
    )
    assert all(n == 0 for n in result.labels_used)


def test_expanded_portfolio_does_not_collapse_portal():
    # Adding the weak first-8 candidate must not wreck the portal
    # controller on mixed arrivals, relative to the core portfolio.
    config = StreamConfig(scenario="mixed_arrivals", seed=1)
    batches = generate_stream(config)
    core = run_stream(
        PolicySpec(name="portal", kind="portal"), config, batches=batches
    )
    expanded = run_stream(
        PolicySpec(name="portal_expanded", kind="portal", portfolio="expanded"),
        config,
        batches=batches,
    )
    assert (
        expanded.summary()["accuracy"]
        >= core.summary()["accuracy"] - 0.05
    )
