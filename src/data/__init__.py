import importlib
import logging

logger = logging.getLogger(__name__)


def __getattr__(name):
    if name == "DataPipeline":
        mod = importlib.import_module("src.data.pipeline")
        return mod.DataPipeline
    if name == "MassiveDataCollector":
        mod = importlib.import_module("src.data.streaming")
        return mod.MassiveDataCollector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["DataPipeline", "MassiveDataCollector"]
