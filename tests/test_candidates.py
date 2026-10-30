import numpy as np

from portal_tta.candidates import (
    BaseCandidate,
    BatchNormCandidate,
    LabelAdaptedCandidate,
    ShrinkCandidate,
    make_candidates,
)
from portal_tta.data import StreamConfig, generate_stream
from portal_tta.metrics import accuracy


def _batch_accuracy(candidate, scenario, seed=0):
    batches = generate_stream(StreamConfig(scenario=scenario, seed=seed))
    return float(
        np.mean([accuracy(candidate.predict(b.x), b.y) for b in batches])
    )


def test_batch_norm_rescues_corrupt_mixed():
    base = _batch_accuracy(BaseCandidate(), "corrupt_mixed", seed=0)
    bn = _batch_accuracy(BatchNormCandidate(), "corrupt_mixed", seed=0)
    assert bn > base + 0.10


def test_batch_norm_collapses_under_class_block():
    base = _batch_accuracy(BaseCandidate(), "corrupt_block", seed=0)
    bn = _batch_accuracy(BatchNormCandidate(), "corrupt_block", seed=0)
    assert bn < base - 0.20
    assert bn < 0.40


def test_batch_norm_collapses_on_clean_block_too():
    base = _batch_accuracy(BaseCandidate(), "clean_block", seed=0)
    bn = _batch_accuracy(BatchNormCandidate(), "clean_block", seed=0)
    assert bn < base - 0.20


def test_shrink_is_between_base_and_batch_norm_on_block():
    base = _batch_accuracy(BaseCandidate(), "corrupt_block", seed=0)
    bn = _batch_accuracy(BatchNormCandidate(), "corrupt_block", seed=0)
    shrink = _batch_accuracy(ShrinkCandidate(), "corrupt_block", seed=0)
    assert bn < shrink < base


def test_non_learning_candidates_are_frozen():
    candidate = BatchNormCandidate()
    batch = generate_stream(StreamConfig(scenario="clean", seed=0))[0]
    first = candidate.predict(batch.x)
    candidate.update(batch.x, batch.y)
    assert np.array_equal(first, candidate.predict(batch.x))


def test_label_adapted_moves_toward_labelled_means():
    candidate = LabelAdaptedCandidate(alpha=1.0)
    before = candidate.prototypes.copy()
    batches = generate_stream(StreamConfig(scenario="clean", seed=0))
    batch = batches[0]
    candidate.update(batch.x, batch.y)
    # With alpha=1, each seen prototype becomes its labelled class mean.
    for k in range(4):
        mask = batch.y == k
        assert np.allclose(candidate.prototypes[k], batch.x[mask].mean(axis=0))
    assert not np.allclose(before, candidate.prototypes)


def test_label_adapted_update_with_no_labels_is_noop():
    candidate = LabelAdaptedCandidate()
    before = candidate.prototypes.copy()
    candidate.update(np.zeros((0, 8)), np.zeros((0,), dtype=int))
    assert np.array_equal(before, candidate.prototypes)


def test_make_candidates_portfolios():
    assert [c.name for c in make_candidates("core")] == [
        "base",
        "batch_norm",
        "shrink",
        "label_adapted",
    ]
    expanded = make_candidates("expanded")
    assert expanded[-1].name == "batch_norm_first8"
    assert len(expanded) == 5
    try:
        make_candidates("bogus")
    except ValueError:
        return
    raise AssertionError("expected ValueError")
