"""Utilities for setting random seeds for reproducibility."""

import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    # Test reproducibility
    set_seed(42)
    tensor1 = torch.randn(3, 3)
    
    set_seed(42)
    tensor2 = torch.randn(3, 3)
    
    assert torch.allclose(tensor1, tensor2), "Tensors should be identical with same seed"
    print("✓ Seed test passed: tensors are identical")

