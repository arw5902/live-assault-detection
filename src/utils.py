"""Utility functions for reproducibility and common operations."""

import random
import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True):
    """
    Set random seeds for reproducibility across all libraries.

    This function ensures deterministic behavior by setting seeds for:
    - Python's built-in random module
    - NumPy random number generator
    - PyTorch (CPU and CUDA)
    - PyTorch backends (cuDNN)

    IMPORTANT: This must be called BEFORE any operations that use randomness,
    including feature extraction (optical flow sampling uses np.random).

    Args:
        seed: Random seed value (default: 42)
        deterministic: If True, enables fully deterministic CUDA operations
                      (may reduce performance slightly)

    Example:
        >>> from src.utils import set_seed
        >>> set_seed(42, deterministic=True)
        >>> # Now all random operations are deterministic
    """
    # Python built-in random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU

        if deterministic:
            # Make CUDA operations deterministic
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # Note: This may reduce performance but ensures reproducibility


def print_seed_info(seed: int = 42, deterministic: bool = True):
    """Print information about random seed configuration."""
    print(f"Random seeds set for reproducibility (seed={seed})")
    print(f"  - Python random seed: {seed}")
    print(f"  - NumPy random seed: {seed}")
    print(f"  - PyTorch manual seed: {seed}")

    if torch.cuda.is_available():
        print(f"  - CUDA manual seed: {seed}")
        if deterministic:
            print(f"  - CUDA deterministic mode: ON")
            print(f"  - cuDNN benchmark: OFF")
        else:
            print(f"  - CUDA deterministic mode: OFF (faster but non-deterministic)")
    else:
        print(f"  - CUDA: Not available")
