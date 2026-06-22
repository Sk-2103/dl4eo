__version__ = "0.5.4"
__author__  = "Saurabh Kaushik"

from .pipeline import generate_dataset
from .config   import PipelineConfig
from . import stats
from . import qc
from . import splits


def __getattr__(name):
    import importlib
    if name == "io":
        mod = importlib.import_module("dl4eo.io")
        globals()["io"] = mod
        return mod
    if name == "eval":
        mod = importlib.import_module("dl4eo.eval")
        globals()["eval"] = mod
        return mod
    if name == "load_module":
        from .eval import load_module
        globals()["load_module"] = load_module
        return load_module
    if name == "train":
        # Expose the train() function, not the module (dl4eo.train(...) calls the function).
        # Access the module directly: from dl4eo.train import build_model, ...
        from .train import train as _train_fn
        globals()["train"] = _train_fn
        return _train_fn
    if name == "build_model":
        from .train import build_model
        globals()["build_model"] = build_model
        return build_model
    if name == "SUPPORTED_MODELS":
        from .train import SUPPORTED_MODELS
        globals()["SUPPORTED_MODELS"] = SUPPORTED_MODELS
        return SUPPORTED_MODELS
    raise AttributeError(f"module 'dl4eo' has no attribute '{name}'")


__all__ = [
    "generate_dataset",
    "PipelineConfig",
    "stats",
    "qc",
    "splits",
    "train",
    "eval",
    "io",
    "build_model",
    "SUPPORTED_MODELS",
]
