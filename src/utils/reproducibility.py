from __future__ import annotations

import os
import random

# Must be set before the first CUDA op for cuBLAS determinism; setting it at
# module import time is the latest reliable point (later than this, the CUDA
# context may already exist and the variable is ignored).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Seed every RNG and optionally enable deterministic algorithms.

    NOTE: ``deterministic=True`` previously called
    ``torch.use_deterministic_algorithms(False)`` — inverting the flag's
    documented meaning. It now enables them.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if hasattr(torch, "use_deterministic_algorithms"):
            # warn_only: some ops used by HF Trainer have no deterministic
            # implementation; erroring on those would break training runs,
            # while still forcing deterministic paths wherever they exist.
            torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(False)
