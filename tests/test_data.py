import numpy as np

from portal_tta.data import StreamConfig, generate_stream


def test_stream_shapes_and_labels():
    config = StreamConfig(scenario="corrupt_mixed", seed=3, n_batches=4)
    batches = generate_stream(config)
    assert len(batches) == 4
    for batch in batches:
        assert batch.x.shape == (64, 8)
        assert batch.y.shape == (64,)
        assert set(batch.y.tolist()) <= {0, 1, 2, 3}


def test_same_seed_same_stream():
    a = generate_stream(StreamConfig(scenario="mixed_arrivals", seed=5))
    b = generate_stream(StreamConfig(scenario="mixed_arrivals", seed=5))
    for x, y in zip(a, b):
        assert np.array_equal(x.x, y.x)
        assert np.array_equal(x.y, y.y)


def test_different_seeds_differ():
    a = generate_stream(StreamConfig(scenario="corrupt_mixed", seed=0))
    b = generate_stream(StreamConfig(scenario="corrupt_mixed", seed=1))
    assert not np.array_equal(a[0].x, b[0].x)


def test_block_batches_are_single_class():
    batches = generate_stream(StreamConfig(scenario="corrupt_block", seed=0))
    for batch in batches:
        assert len(set(batch.y.tolist())) == 1
        assert batch.arrival == "block"


def test_mixed_batches_contain_multiple_classes():
    batches = generate_stream(StreamConfig(scenario="clean", seed=0))
    assert all(len(set(b.y.tolist())) > 1 for b in batches)


def test_drift_changes_domain_mid_stream():
    batches = generate_stream(
        StreamConfig(scenario="drift", seed=0, drift_batch=6)
    )
    assert all(b.domain == "clean" for b in batches[:6])
    assert all(b.domain == "corrupt" for b in batches[6:])


def test_unknown_scenario_rejected():
    try:
        StreamConfig(scenario="nope")
    except ValueError:
        return
    raise AssertionError("expected ValueError")
