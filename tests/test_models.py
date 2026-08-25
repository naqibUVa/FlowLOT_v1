import numpy as np
import pytest
import torch

from data.dataset import InMemoryCytometryDataset, cytometry_collate
from models import AttentionMIL, CellCNN, CytoSet, DGCNN, PointNet2
from models.flowsom_baseline import FlowSOMClassifier


@pytest.fixture
def batch():
    dataset = InMemoryCytometryDataset(
        [np.random.randn(11, 5), np.random.randn(7, 5)], [0, 1]
    )
    return cytometry_collate([dataset[0], dataset[1]])


@pytest.mark.parametrize(
    "model",
    [
        CellCNN(5, 2, num_filters=8),
        AttentionMIL(5, 2, hidden_dim=8, attention_dim=4),
        CytoSet(5, 2, hidden_dim=8, set_dim=8),
        DGCNN(5, 2, k=3, hidden_dim=8),
        PointNet2(5, 2, npoint1=6, npoint2=3, neighbours=3, hidden_dim=8),
    ],
)
def test_model_forward_and_backward(model, batch):
    logits = model(batch["cells"], batch["mask"])
    assert logits.shape == (2, 2)
    logits.sum().backward()


def test_attention_respects_padding(batch):
    model = AttentionMIL(5, 2, hidden_dim=8, attention_dim=4)
    _, attention = model(batch["cells"], batch["mask"], return_attention=True)
    assert torch.allclose(attention.sum(1), torch.ones(2))
    assert torch.count_nonzero(attention[~batch["mask"]]) == 0


def test_flowsom_probabilities():
    rng = np.random.default_rng(1)
    samples = [rng.normal(i % 2, 0.2, size=(20, 3)) for i in range(8)]
    labels = np.arange(8) % 2
    model = FlowSOMClassifier(
        random_state=1,
        grid=(3, 3),
        n_meta_clusters=3,
        som_iterations=30,
        max_training_cells=100,
    ).fit(samples, labels)
    probabilities = model.predict_proba(samples)
    assert probabilities.shape == (8, 2)
    np.testing.assert_allclose(probabilities.sum(1), 1)
