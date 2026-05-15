"""Reproducibility helpers."""

import random
import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True):
    """
    Seed Python, NumPy, and PyTorch (CPU + CUDA) for reproducible runs.
    Must be called BEFORE any randomness, including the optical-flow point
    sampling in features.py.

    If `deterministic` is True, also forces cuDNN into deterministic mode,
    which slightly slows training but pins the outputs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def print_seed_info(seed: int = 42, deterministic: bool = True):
    cuda = "off"
    if torch.cuda.is_available():
        cuda = "deterministic" if deterministic else "non-deterministic"
    print(f"seed={seed}  cuda={cuda}")
