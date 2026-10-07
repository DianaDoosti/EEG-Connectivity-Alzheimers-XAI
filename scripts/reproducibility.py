"""Seed control so every run is exactly reproducible on any machine."""

import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Fix seeds across python, numpy, torch (CPU + CUDA)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic cuDNN. Slower, but required for exact reproducibility.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_generator(seed: int) -> torch.Generator:
    """A dedicated generator for DataLoader shuffling, seeded separately
    from the global seed so dataloader order is reproducible independent
    of other random draws."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def seed_worker(worker_id: int) -> None:
    """Per-worker seeding for DataLoader (only matters if NUM_WORKERS > 0)."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
