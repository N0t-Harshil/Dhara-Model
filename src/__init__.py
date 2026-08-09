import importlib
import logging as _logging

_logger = _logging.getLogger(__name__)


def __getattr__(name):
    _lazy = {
        "SpecializedCoderModel": ("src.model", "SpecializedCoderModel"),
        "ModelTrainer": ("src.trainer", "ModelTrainer"),
        "SpecializedDataset": ("src.dataset", "SpecializedDataset"),
        "CodeGenerator": ("src.generator", "CodeGenerator"),
        "CodeValidator": ("src.validator", "CodeValidator"),
        "CodingBenchmark": ("src.benchmark", "CodingBenchmark"),
        "MassiveDataCollector": ("src.data.streaming", "MassiveDataCollector"),
        "train_custom_tokenizer": ("src.tokenizer_trainer", "train_custom_tokenizer"),
        "GraphMemory": ("src.knowledge_graph", "GraphMemory"),
        "Config": ("src.config.schema", "Config"),
        "load_config": ("src.config.schema", "load_config"),
        "TrainingPipeline": ("src.training.pipeline", "TrainingPipeline"),
        "AlignmentPipeline": ("src.alignment.pipeline", "AlignmentPipeline"),
        "BenchmarkRunner": ("src.evaluation.benchmarks", "BenchmarkRunner"),
        "EvaluationReport": ("src.evaluation.reporting", "EvaluationReport"),
    }
    if name in _lazy:
        mod_path, attr = _lazy[name]
        mod = importlib.import_module(mod_path)
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "SpecializedCoderModel", "ModelTrainer", "SpecializedDataset",
    "CodeGenerator", "CodeValidator", "CodingBenchmark",
    "MassiveDataCollector", "train_custom_tokenizer", "GraphMemory",
    "Config", "load_config", "TrainingPipeline", "AlignmentPipeline",
    "BenchmarkRunner", "EvaluationReport",
]
