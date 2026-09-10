from __future__ import annotations

import inspect
import os
from typing import Any

from datasets import IterableDataset
from transformers import Trainer


def dataloader_num_workers(dataset: Any) -> int:
    """Return a safe number of dataloader worker processes for a dataset."""
    return 0 if isinstance(dataset, IterableDataset) else min(4, os.cpu_count() or 4)


def trainer_tokenizer_kwarg() -> str:
    """Return the correct Trainer keyword for tokenizer/processing_class compatibility."""
    params = inspect.signature(Trainer.__init__).parameters
    return "processing_class" if "processing_class" in params else "tokenizer"
