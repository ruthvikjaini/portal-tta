import numpy as np
import pytest

from portal_tta.metrics import accuracy, batch_harm, summarise, worst_batch


def test_accuracy():
    assert accuracy(np.array([0, 1, 2]), np.array([0, 1, 1])) == 2 / 3


def test_harm_is_only_the_loss_vs_base():
    assert batch_harm(0.9, 0.8) == 0.0  # deployed beat base: no harm
    assert batch_harm(0.6, 0.9) == pytest.approx(0.3)
    assert batch_harm(0.0, 0.0) == 0.0


def test_worst_batch():
    assert worst_batch([0.9, 0.4, 0.8]) == 0.4
    assert worst_batch([]) == 0.0


def test_summarise_regret():
    out = summarise(
        batch_accuracies=[0.8, 0.9],
        batch_harms=[0.0, 0.1],
        labels_used=[4, 6],
        oracle_accuracies=[0.9, 0.95],
    )
    assert out["accuracy"] == pytest.approx(0.85)
    assert out["mean_harm"] == pytest.approx(0.05)
    assert out["labels_per_batch"] == pytest.approx(5.0)
    assert out["total_labels"] == pytest.approx(10.0)
    assert out["regret_vs_oracle"] == pytest.approx(0.075)
