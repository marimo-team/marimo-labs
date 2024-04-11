import importlib.metadata
import warnings

import marimo_labs.huggingface as huggingface

try:
    __version__ = importlib.metadata.version(__name__)
except importlib.metadata.PackageNotFoundError as e:
    warnings.warn(
        f"Could not determine version of {__name__}\n{e!r}", stacklevel=2
    )
    __version__ = "unknown"

__all__ = ["huggingface"]
