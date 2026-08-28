"""Refuse holdouts and the frozen P3 1040 stress set."""
from __future__ import annotations

from pathlib import Path

from p3.guard import refuse_holdout, refuse_path

from .config import FORBIDDEN_P3_STRESS, ROOT


def refuse_legacy_p3_stress(path: str | Path | None = None) -> None:
    refuse_holdout(str(path or ""))
    if path is not None:
        refuse_path(path)
    s = str(path or "")
    if FORBIDDEN_P3_STRESS in s.replace("\\", "/"):
        raise RuntimeError("P3R-EC FORBIDDEN: frozen P3 1040 stress set")
    p3_pert = ROOT / "results" / "p3" / "perturbation_cases_head.json"
    if path is not None and Path(path).resolve() == p3_pert.resolve():
        raise RuntimeError("P3R-EC FORBIDDEN: frozen P3 1040 stress set")
