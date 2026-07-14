"""
Training module for EQ-KA-GCN.
Contains dataset splitting, loader configuration, and optimization logic.
"""

from training.dataset_split import split_graph_dataset
from training.dataloader import create_dataloaders

__all__ = [
    "split_graph_dataset",
    "create_dataloaders",
]
