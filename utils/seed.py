"""
Utilities for reproducible experiments.

Call `set_seed(seed)` once at the start of each run to make all the
random number generators deterministic (numpy, torch CPU, torch CUDA, cuDNN).

Note: full bitwise reproducibility on GPU requires `deterministic=True`,
which is slower than the default. We use it because we want our experiments
to be reproducible across runs and machines.
"""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Seed all random number generators used in the project for reproducibility.

    Args:
        seed: integer seed value. Pass the same seed to get identical results
              across runs (assuming hardware/software stack stays the same).
        deterministic: if True, force PyTorch to use deterministic algorithms
              (slightly slower but reproducible across runs).
    """
    # Python's built-in random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch GPU (all visible devices)
    torch.cuda.manual_seed_all(seed)

    # Hash-based seeds (for some hash collisions in dict ordering, etc.)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        # cuDNN: disable nondeterministic auto-tuning, enable deterministic algos
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    # Sanity check: run the same code twice with the same seed -> identical output
    print("=" * 60)
    print("Sanity check: same seed -> same random numbers")
    print("=" * 60)

    set_seed(42)
    a_np = np.random.rand(3)
    a_torch = torch.randn(3)
    a_cuda = torch.randn(3, device="cuda") if torch.cuda.is_available() else None

    set_seed(42)
    b_np = np.random.rand(3)
    b_torch = torch.randn(3)
    b_cuda = torch.randn(3, device="cuda") if torch.cuda.is_available() else None

    print(f"NumPy run 1:  {a_np}")
    print(f"NumPy run 2:  {b_np}")
    assert np.allclose(a_np, b_np), "NumPy seeding failed"

    print(f"\nTorch CPU run 1: {a_torch}")
    print(f"Torch CPU run 2: {b_torch}")
    assert torch.allclose(a_torch, b_torch), "Torch CPU seeding failed"

    if a_cuda is not None:
        print(f"\nTorch CUDA run 1: {a_cuda}")
        print(f"Torch CUDA run 2: {b_cuda}")
        assert torch.allclose(a_cuda, b_cuda), "Torch CUDA seeding failed"

    print("\n✓ All seeding checks passed: identical seed -> identical output.")