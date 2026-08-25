"""Unified access to the cytometry baselines shipped with FlowLOT."""

from models import AttentionMIL, CellCNN, CytoSet, DGCNN, PointNet2
from models.flowsom_baseline import FlowSOMClassifier, FlowSOMFeatureExtractor

BASELINE_REGISTRY = {
    "cellcnn": CellCNN,
    "attention_mil": AttentionMIL,
    "cytoset": CytoSet,
    "dgcnn": DGCNN,
    "pointnet2": PointNet2,
}

__all__ = [
    "AttentionMIL",
    "BASELINE_REGISTRY",
    "CellCNN",
    "CytoSet",
    "DGCNN",
    "FlowSOMClassifier",
    "FlowSOMFeatureExtractor",
    "PointNet2",
]
