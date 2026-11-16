"""Configuration loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path) -> Path:
    """Return an absolute path; relative paths are resolved against the project root."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_config(path: str | Path = "config/config.yaml") -> dict[str, Any]:
    """Load the YAML configuration file into a dictionary."""
    with resolve_path(path).open(encoding="utf-8") as handle:
        cfg: dict[str, Any] = yaml.safe_load(handle)
    return cfg
