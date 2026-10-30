import numpy as np

from portal_tta.controller import PortalConfig
from portal_tta.data import StreamConfig, generate_stream
from portal_tta.policies import FixedPolicy, OraclePolicy, PolicySpec, build_policy
from portal_tta.benchmark import run_stream


def _decide(policy):
    preds = {
        "base": np.array([0, 1, 0]),
        "batch_norm": np.array([1, 1, 0]),
    }
    labels = np.array([0, 1, 0])
    calls = []

    def labels_for(indices):
        calls.extend(int(i) for i in indices)
        return labels[np.asarray(indices)]

    return policy.decide(0, preds, labels_for), calls


def test_fixed_policy_never_queries():
    decision, calls = _decide(FixedPolicy("batch_norm"))
    assert decision.choice == "batch_norm"
    assert decision.n_labels == 0
    assert calls == []


def test_oracle_queries_everything_and_picks_best():
    decision, calls = _decide(OraclePolicy())
    assert decision.n_labels == 3
    assert len(calls) == 3
    # base correct on all 3, batch_norm on 2 -> oracle picks base.
    assert decision.choice == "base"


def test_unknown_policy_kind_rejected():
    try:
        build_policy(PolicySpec(name="x", kind="bogus"))
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_oracle_never_worse_than_fixed_on_a_stream():
    config = StreamConfig(scenario="mixed_arrivals", seed=0)
    batches = generate_stream(config)
    oracle = run_stream(PolicySpec(name="oracle", kind="oracle"), config, batches=batches)
    fixed = run_stream(
        PolicySpec(name="always_base", kind="fixed", fixed_choice="base"),
        config,
        batches=batches,
    )
    assert oracle.summary()["accuracy"] >= fixed.summary()["accuracy"]


def test_portal_policy_spec_builds_controller():
    policy = build_policy(
        PolicySpec(
            name="portal",
            kind="portal",
            portal_config=PortalConfig(max_labels=4, seed=0),
        )
    )
    assert policy.config.max_labels == 4
