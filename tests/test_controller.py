import numpy as np

from portal_tta.candidates import make_candidates
from portal_tta.controller import (
    NaiveController,
    PortalConfig,
    PortalController,
    frozen_predictions,
)
from portal_tta.data import StreamConfig, generate_stream


def _predictions(scenario="clean", seed=0, portfolio="core"):
    batches = generate_stream(StreamConfig(scenario=scenario, seed=seed))
    candidates = make_candidates(portfolio)
    return batches[0], frozen_predictions(candidates, batches[0].x)


def _labels_for(batch):
    def labels_for(indices):
        return batch.y[np.asarray(indices)]

    return labels_for


def test_queries_only_disagreements():
    batch, predictions = _predictions()
    controller = PortalController(PortalConfig(max_labels=16, seed=0))
    decision = controller.decide(0, predictions, _labels_for(batch))
    leader = predictions["base"]
    for idx in decision.queried:
        # Every queried input carried a disagreement between at least
        # two candidates; unanimous inputs are never queried.
        assert len({int(predictions[name][idx]) for name in predictions}) > 1


def test_shared_comparisons_one_label_counts_for_all():
    batch, predictions = _predictions()
    controller = PortalController(PortalConfig(max_labels=1, min_labels=1, seed=0))
    decision = controller.decide(0, predictions, _labels_for(batch))
    assert decision.total_labelled == 1
    assert decision.n_labels == 1
    idx = decision.queried[0]
    for name, pred in predictions.items():
        expected = 1 if int(pred[idx]) == int(batch.y[idx]) else 0
        assert decision.correct[name] == expected


def test_label_budget_respected():
    batch, predictions = _predictions()
    controller = PortalController(PortalConfig(max_labels=5, seed=0))
    decision = controller.decide(0, predictions, _labels_for(batch))
    assert decision.n_labels <= 5


def test_no_labels_when_nothing_disagrees():
    # All candidates identical: no input can change the choice, so the
    # controller queries nothing at all, whatever the label budget is.
    pred = np.zeros(64, dtype=int)
    predictions = {"base": pred.copy(), "batch_norm": pred.copy()}
    labels = np.zeros(64, dtype=int)
    controller = PortalController(
        PortalConfig(max_labels=16, min_labels=1, seed=0)
    )
    decision = controller.decide(
        0, predictions, lambda idx: labels[np.asarray(idx)]
    )
    assert decision.n_labels == 0
    assert decision.choice == "base"


def test_tie_prefers_base():
    # Every candidate wrong on the queried inputs except the counts
    # tie: the leader must be the preferred (base) rule.
    predictions = {
        "base": np.array([0, 1, 0, 1]),
        "batch_norm": np.array([0, 1, 0, 1]),
    }
    labels = np.array([0, 1, 1, 0])
    controller = PortalController(
        PortalConfig(max_labels=4, min_labels=4, seed=0, label_cost=0.0)
    )
    decision = controller.decide(
        0, predictions, lambda idx: labels[np.asarray(idx)]
    )
    assert decision.choice == "base"


def test_uses_fewer_labels_than_naive_on_clean_stream():
    batches = generate_stream(StreamConfig(scenario="clean", seed=0))
    candidates = make_candidates()
    portal = PortalController(PortalConfig(max_labels=16, seed=0))
    naive = NaiveController(max_labels=16, seed=0)
    portal_labels = naive_labels = 0
    for batch in batches:
        preds = frozen_predictions(candidates, batch.x)
        portal_labels += portal.decide(
            batch.index, preds, _labels_for(batch)
        ).n_labels
        naive_labels += naive.decide(
            batch.index, preds, _labels_for(batch)
        ).n_labels
    assert portal_labels < naive_labels


def test_refresh_discards_stale_evidence():
    batch, predictions = _predictions()
    controller = PortalController(
        PortalConfig(max_labels=4, min_labels=1, probe_every=2, seed=0)
    )
    controller.decide(0, predictions, _labels_for(batch))
    assert controller._total > 0
    controller.decide(2, predictions, _labels_for(batch))
    # Batch 2 is a probe batch: evidence was reset before selection,
    # so the total reflects only this batch's queries.
    assert controller.generation == 1
    assert controller._total <= 4


def test_deterministic_given_seed():
    batch, predictions = _predictions()

    def run():
        controller = PortalController(PortalConfig(max_labels=16, seed=7))
        return controller.decide(0, predictions, _labels_for(batch))

    first, second = run(), run()
    assert first.queried == second.queried
    assert first.choice == second.choice


def test_invalid_config_rejected():
    for kwargs in ({"max_labels": 0}, {"min_labels": 99}, {"label_cost": -1.0}):
        try:
            PortalConfig(**kwargs)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {kwargs}")
