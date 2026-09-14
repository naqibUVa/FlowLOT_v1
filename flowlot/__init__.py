import sys
sys.modules.setdefault("FlowLOT", sys.modules[__name__])

from .transport.lot_engine import LOTResult, compute_lot, compute_stage2_embeddings

__version__ = "0.1.0"
__all__ = ["LOTResult", "compute_lot", "compute_stage2_embeddings", "__version__"]

