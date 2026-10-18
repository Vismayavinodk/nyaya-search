"""
Deterministic seed setting for end-to-end reproducibility across CPU and GPU.
"""

import os
import random
import numpy as np
import torch


def seed_everything(seed: int = 42) -> int:
    """
    Sets deterministic random seeds for python random, numpy, and torch.
    Enforces deterministic CuDNN backend settings if CUDA is available.

    Args:
        seed: The integer random seed to apply globally.

    Returns:
        The applied seed integer.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    return seed
