"""Shared unused-window selection. Same cap as Phase 2 Exp 1."""
from __future__ import annotations

from p2_phase2.construct import build_items, _named
from p35.windows_ir import load_unused_windows

from .config import MAX_PER_DATASET, MAX_WINDOWS


def selected_windows() -> list[dict]:
    wins = load_unused_windows()
    by_ds: dict[str, list] = {}
    for w in wins:
        by_ds.setdefault(w["dataset"], []).append(w)
    selected = []
    for ds in sorted(by_ds):
        selected.extend(by_ds[ds][:MAX_PER_DATASET])
    selected = selected[:MAX_WINDOWS]
    selected.sort(key=lambda w: w["window_id"])
    return selected


def e1_items() -> list[dict]:
    """Same carrier programs as Phase 2 Exp 1."""
    return build_items()


__all__ = ["selected_windows", "e1_items", "_named"]
