"""Build frozen clean-SUPPORTED ClaimPrograms on unused windows.

Thresholds are written once from the independent DSP reference on the clean
waveform and never adjusted after a perturbation is applied.
"""
from __future__ import annotations

from p2.independent_dsp import MeasurementError, measure, tolerance_for
from p2r.schema import ClaimProgram, Predicate
from p35.windows_ir import load_unused_windows

from .config import (
    MAX_PER_DATASET,
    MAX_WINDOWS,
    OPS,
    THRESHOLD_TOL_MULT,
    UNITS,
)


def _named(op: str, available: list[str]) -> list[str] | None:
    if op == "cross_channel_lag_ms":
        if len(available) < 2:
            return None
        return [available[0], available[1]]
    if not available:
        return None
    return [available[0]]


def _program(op: str, named: list[str], threshold: float) -> ClaimProgram:
    pred = Predicate(
        measurement=op,
        channel_a=named[0],
        comparator="gt",
        channel_b=named[1] if op == "cross_channel_lag_ms" else None,
        reference_value=float(threshold),
        reference_channel=None,
        unit=UNITS[op],
    )
    return ClaimProgram("SINGLE", [pred], parse_status="OK")


def build_items() -> list[dict]:
    wins = load_unused_windows()
    by_ds: dict[str, list] = {}
    for w in wins:
        by_ds.setdefault(w["dataset"], []).append(w)
    selected = []
    for ds in sorted(by_ds):
        selected.extend(by_ds[ds][:MAX_PER_DATASET])
    selected = selected[:MAX_WINDOWS]
    selected.sort(key=lambda w: w["window_id"])

    items = []
    skipped = 0
    for w in selected:
        avail = list(w["available_channels"])
        ch = w["channels"]
        fs = float(w["fs"])
        for op in OPS:
            named = _named(op, avail)
            if named is None:
                skipped += 1
                continue
            try:
                cmap = {n: ch[n] for n in named}
                v = float(measure(op, cmap, fs))
                tol = float(tolerance_for(op, v))
            except MeasurementError:
                skipped += 1
                continue
            if not (v == v) or abs(v) == float("inf"):
                skipped += 1
                continue
            thr = v - THRESHOLD_TOL_MULT * tol
            prog = _program(op, named, thr)
            items.append({
                "item_id": f"{w['window_id']}:{op}",
                "window_id": w["window_id"],
                "dataset": w["dataset"],
                "subject": w.get("subject"),
                "fs": fs,
                "available_channels": avail,
                "named_channels": named,
                "channels": ch,
                "op": op,
                "clean_value": v,
                "tolerance": tol,
                "threshold": thr,
                "program": prog,
            })
    return items
