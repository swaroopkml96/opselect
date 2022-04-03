import random
import numpy as np
import torch

def set_seed(seed):
    """Set a random seed consistently across multiple packages
    Args:
        seed (int): The random seed to set
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True