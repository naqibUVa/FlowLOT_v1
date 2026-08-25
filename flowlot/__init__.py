"""FlowLOT: linear optimal transport for multi-tube cytometry."""

from .transport.lot_engine import LOTResult, compute_lot, compute_stage2_embeddings

__version__ = "0.1.0"
__all__ = ["LOTResult", "compute_lot", "compute_stage2_embeddings", "__version__"]
