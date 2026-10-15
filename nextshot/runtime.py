"""Deterministic runtime setup."""
import os
import random
import numpy as np
import torch

def configure_determinism():
    """Local-process policy: deterministic math SDPA, including checkpoint recomputation."""
    previous = os.environ.get('CUBLAS_WORKSPACE_CONFIG')
    if previous not in (None, ':4096:8'):
        raise ValueError('Conflicting CUBLAS_WORKSPACE_CONFIG; use a fresh process with :4096:8')
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
