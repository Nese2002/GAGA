"""GAGA - Group AGgregation enhanced TrAnsformer for graph fraud detection.

A clean, DGL-free reimplementation. The pipeline has three stages:

1. ``data``     - load the YelpChi / Amazon fraud graphs from their ``.mat`` files.
2. ``sequence`` - turn each node's multi-hop neighbourhood into a grouped sequence.
3. ``model``    - a Transformer encoder that classifies those sequences.
"""

from .data import load_dataset
from .sequence import build_sequences
from .model import GAGA

__all__ = ["load_dataset", "build_sequences", "GAGA"]
