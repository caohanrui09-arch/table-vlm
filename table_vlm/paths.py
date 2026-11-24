"""Portable path and environment resolution.

The handoff package used to hard-code the author's ``<DATA_ROOT>/...`` paths in
the demo defaults.  This module keeps every path relative to the project root
and lets the receiving machine override it with a CLI flag or an environment
variable.  It deliberately imports only the standard library so it can be used
by the dependency-light preflight checks.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT_ROOT / 'models' / 'Qwen3-VL-2B-Instruct'
DEFAULT_ADAPTER = PROJECT_ROOT / 'artifacts' / 'q1-adapter'
DEFAULT_OUTPUT = PROJECT_ROOT / 'demo-output'
DEFAULT_EXAMPLE = PROJECT_ROOT / 'examples' / 'table.png'
DEFAULT_MANIFEST = PROJECT_ROOT / 'research' / 'qwen2b-fixed-manifest.json'

# Frozen evidence from the handoff.  Changing these values breaks the claim
# that the demo is running the same adapter and base revision as the sealed
# final test.
Q1_SHA256 = '5ba4f106312a738283916fb5c691376584539d0824c597e02f6485c995da250c'
BASE_REVISION = '89644892e4d85e24eaac8bacfd4f463576704203'
ADAPTER_FILES = ('adapter_config.json', 'adapter_model.safetensors')
MODEL_WEIGHT = 'model.safetensors'
MODEL_METADATA = (
    'config.json',
    'preprocessor_config.json',
    'tokenizer_config.json',
    'tokenizer.json',
    'vocab.json',
    'merges.txt',
)

ENV_MODEL = 'TABLE_VLM_MODEL'
ENV_ADAPTER = 'TABLE_VLM_ADAPTER'
ENV_OUTPUT = 'TABLE_VLM_OUTPUT'
ENV_GPU_PYTHON = 'TABLE_VLM_GPU_PYTHON'
ENV_PYTHON = 'TABLE_VLM_PYTHON'
ENV_EXTRA_PYTHONPATH = 'TABLE_VLM_PYTHONPATH'


def _explicit(value: str | os.PathLike[str] | None) -> Path | None:
    if value is None:
        return None
    text = os.fspath(value).strip()
    return Path(text).expanduser() if text else None


def _resolve(value: str | os.PathLike[str] | None, env: str, default: Path) -> Path:
    explicit = _explicit(value)
    if explicit is not None:
        return explicit.resolve()
    from_env = _explicit(os.environ.get(env))
    if from_env is not None:
        return from_env.resolve()
    return default.resolve()


def model_path(value: str | os.PathLike[str] | None = None) -> Path:
    """Directory that holds the pinned base processor and ``model.safetensors``."""
    return _resolve(value, ENV_MODEL, DEFAULT_MODEL)


def adapter_path(value: str | os.PathLike[str] | None = None) -> Path:
    """Directory that holds the frozen Q1 LoRA adapter."""
    return _resolve(value, ENV_ADAPTER, DEFAULT_ADAPTER)


def output_path(value: str | os.PathLike[str] | None = None) -> Path:
    """Root for request records, previews and downloads (never the source tree)."""
    return _resolve(value, ENV_OUTPUT, DEFAULT_OUTPUT)


def example_path(value: str | os.PathLike[str] | None = None) -> Path:
    return _explicit(value).resolve() if value else DEFAULT_EXAMPLE.resolve()


def gpu_python(value: str | os.PathLike[str] | None = None) -> Path:
    """Interpreter that runs the model worker.

    A single environment is now the normal case, so the default is the current
    interpreter.  Machines that keep a separate CUDA environment can pass
    ``--gpu-python`` or set ``TABLE_VLM_GPU_PYTHON``.
    """
    explicit = _explicit(value)
    if explicit is not None:
        return explicit.resolve()
    from_env = _explicit(os.environ.get(ENV_GPU_PYTHON))
    if from_env is not None:
        return from_env.resolve()
    return Path(sys.executable).resolve()


def manifest_path(value: str | os.PathLike[str] | None = None) -> Path:
    return _explicit(value).resolve() if value else DEFAULT_MANIFEST.resolve()


def extra_pythonpath(value: str | os.PathLike[str] | None = None) -> list[Path]:
    """Optional out-of-tree packages for the worker (e.g. a separate torchvision).

    The original machine kept ``torchvision`` in a sibling ``.inference-deps``
    directory because installing it into the CUDA environment caused resolver
    conflicts.  That setup is now explicit instead of hard-coded: pass
    ``--extra-pythonpath`` or set ``TABLE_VLM_PYTHONPATH`` with one or more
    ``os.pathsep``-separated directories.
    """
    raw = value if value is not None else os.environ.get(ENV_EXTRA_PYTHONPATH, '')
    parts = [part.strip() for part in str(raw).split(os.pathsep) if part.strip()]
    return [Path(part).expanduser().resolve() for part in parts]


def as_posix(path: Path) -> str:
    """Stable forward-slash form for receipts written to JSON."""
    return path.resolve().as_posix()
