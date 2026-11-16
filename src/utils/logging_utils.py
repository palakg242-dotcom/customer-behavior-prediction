"""Structured logging setup (key=value messages, no print statements)."""

from __future__ import annotations

import logging

_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def configure_logging(level: str = "INFO") -> None:
    """Configure the root logger once for pipeline runs."""
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format=_FORMAT)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger."""
    return logging.getLogger(name)
