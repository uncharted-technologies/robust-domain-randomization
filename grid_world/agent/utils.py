
import random

import numpy as np
import torch


def set_global_seed(seed, env):
    torch.manual_seed(seed)
    env.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
