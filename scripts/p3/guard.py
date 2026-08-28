"""Refuse both sealed holdouts. P3 never opens them."""
from __future__ import annotations

from .config import FORBIDDEN_SPLITS, HOLDOUT_MARKERS


def refuse_holdout(name: str) -> None:
    n = (name or "").lower()
    if name in FORBIDDEN_SPLITS or any(m in n for m in HOLDOUT_MARKERS):
        raise RuntimeError(f"P3 FORBIDDEN: sealed holdout access ({name})")


def refuse_path(path) -> None:
    s = str(path).replace("\\", "/").lower()
    if "final_sealed_holdout" in s:
        raise RuntimeError(f"P3 FORBIDDEN: sealed holdout path {path}")
