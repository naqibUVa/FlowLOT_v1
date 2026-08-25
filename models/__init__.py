"""Cytometry set-classification models."""

from .attention_mil import AttentionMIL
from .cell_cnn import CellCNN
from .cytoset import CytoSet
from .dgcnn import DGCNN
from .pointnet2 import PointNet2

MODEL_REGISTRY = {
    "cellcnn": CellCNN,
    "attention_mil": AttentionMIL,
    "cytoset": CytoSet,
    "dgcnn": DGCNN,
    "pointnet2": PointNet2,
}

__all__ = ["AttentionMIL", "CellCNN", "CytoSet", "DGCNN", "PointNet2", "MODEL_REGISTRY"]
