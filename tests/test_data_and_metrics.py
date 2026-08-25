import numpy as np

from data.dataset import InMemoryCytometryDataset, cytometry_collate
from metrics import classification_metrics


def test_collate_variable_bags_and_deterministic_sampling():
    samples = [np.arange(30).reshape(10, 3), np.ones((4, 3))]
    dataset = InMemoryCytometryDataset(samples, [0, 1], max_cells=6, seed=7)
    first = dataset[0]["cells"]
    second = dataset[0]["cells"]
    assert np.array_equal(first, second)
    batch = cytometry_collate([dataset[0], dataset[1]])
    assert batch["cells"].shape == (2, 6, 3)
    assert batch["mask"].sum().item() == 10


def test_binary_metrics():
    result = classification_metrics([0, 1, 1], [[0.8, 0.2], [0.1, 0.9], [0.4, 0.6]])
    assert all(value == 1.0 for value in result.values())
