from .potsdam import PotsdamMulticlassDataset
from .vaihingen import VaihingenMulticlassDataset
from .loveda import LoveDAMulticlassDataset

DATASET_REGISTRY = {
    "potsdam": PotsdamMulticlassDataset,
    "vaihingen": VaihingenMulticlassDataset,
    "loveda": LoveDAMulticlassDataset,
}

__all__ = [
    "PotsdamMulticlassDataset",
    "VaihingenMulticlassDataset",
    "LoveDAMulticlassDataset",
    "DATASET_REGISTRY",
]
