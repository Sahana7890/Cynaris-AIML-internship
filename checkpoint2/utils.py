"""Utility functions for MedScan Diagnostics Checkpoint 2."""

from pathlib import Path
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path) -> Path:
    """Create a directory if it does not exist and return it as Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
