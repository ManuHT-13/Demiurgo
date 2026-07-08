"""
Set all possible random seeds to make experiments reproducible
If you run the same script twice with the same seed, you should
obtain almost identical results

When comparing the three scenarios (supervised learning, memorization, and SimCLR), we
want the only real difference between them to be the labels or
training objective, not random chance
"""

import random

import numpy as np
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)