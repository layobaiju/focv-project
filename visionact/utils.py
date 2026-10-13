"""Small shared helpers: logging, seeding, device selection and JSON I/O."""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any

from .config import DEVICE

# NumPy and PyTorch are imported inside the functions that need them so that
# light commands (for example the dataset download) start without them.

_LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


class ModelNotTrainedError(FileNotFoundError):
    """A trained model file is required but has not been created yet."""


def get_logger(name: str = "visionact") -> logging.Logger:
    """Return a logger under the ``visionact`` namespace, configuring it once."""
    root = logging.getLogger("visionact")
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt="%H:%M:%S"))
        root.addHandler(handler)
        root.setLevel(os.environ.get("VISIONACT_LOG_LEVEL", "INFO").upper())
        root.propagate = False
    return logging.getLogger(name if name.startswith("visionact") else f"visionact.{name}")


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch so a run can be repeated exactly on CPU."""
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def get_device():
    """The project is CPU-only by design; this never returns a CUDA device."""
    import torch

    return torch.device(DEVICE)


def configure_torch_threads(num_threads: int) -> None:
    if num_threads > 0:
        import torch

        torch.set_num_threads(num_threads)


def write_json(path: str | Path, data: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=_json_default), encoding="utf-8")
    return path


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_json_if_exists(path: str | Path) -> Any | None:
    return read_json(path) if Path(path).is_file() else None


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):  # NumPy scalars and arrays
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serialisable")


def reported_interval(model: dict) -> tuple[float, float, str]:
    """The 95% accuracy interval to show for a model, and what it is based on.

    Normally the interval that resamples source videos. When a model makes no
    errors (or only errors) every resample gives the same score, so that
    interval collapses to a point; the Wilson interval, which treats videos as
    independent, is shown instead and labelled as such.
    """
    low, high = model["accuracy_ci95_group_bootstrap"]
    if low == high:
        return (*model["accuracy_ci95"], "independent videos")
    return (low, high, "source videos")


def file_size_mb(path: str | Path) -> float:
    return Path(path).stat().st_size / 1e6
