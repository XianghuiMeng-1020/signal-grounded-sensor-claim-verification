"""Module 3 — paired production vs independent reference. Evaluation only."""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from f_round6_operators import compute as prod_compute
from p2.config import VALIDATION_TOL, dominant_freq_abs_tol
from p2.independent_dsp import MeasurementError, measure as ref_measure

from .config import EXPERIMENT_IDS, OPS, RESULTS, SEED
from .windows import _named, selected_windows


def _within_tol(op: str, prod: float, ref: float, n: int, fs: float) -> bool:
    err = abs(float(prod) - float(ref))
    if op == "dominant_frequency":
        return err <= dominant_freq_abs_tol(n, fs)
    if op == "cross_channel_lag_ms":
        return (err * float(fs) / 1000.0) <= (VALIDATION_TOL[op]["abs_samples"] + 1e-12)
    spec = VALIDATION_TOL[op]
    abs_ok = err <= float(spec["abs"])
    rel = spec.get("rel")
    if rel is None:
        return abs_ok
    rel_ok = err <= float(rel) * max(abs(float(ref)), 1e-15)
    return bool(abs_ok or rel_ok)


def _rel_err(prod: float, ref: float) -> float:
    return abs(float(prod) - float(ref)) / max(abs(float(ref)), 1e-15)


def pair_windows() -> tuple[list[dict], dict]:
    records = []
    skipped = 0
    for w in selected_windows():
        avail = list(w["available_channels"])
        ch = w["channels"]
        fs = float(w["fs"])
        for op in OPS:
            named = _named(op, avail)
            if named is None:
                skipped += 1
                continue
            cmap = {n: ch[n] for n in named}
            n = min(np.asarray(ch[k]).size for k in named)
            try:
                ref = float(ref_measure(op, cmap, fs))
            except MeasurementError:
                skipped += 1
                continue
            if not np.isfinite(ref):
                skipped += 1
                continue
            try:
                prod = float(prod_compute(op, cmap, fs))
            except Exception:
                skipped += 1
                continue
            if not np.isfinite(prod):
                skipped += 1
                continue
            err = abs(prod - ref)
            rec = {
                "window_id": w["window_id"],
                "dataset": w["dataset"],
                "op": op,
                "fs": fs,
                "n": int(n),
                "production": prod,
                "reference": ref,
                "abs_error": err,
                "rel_error": _rel_err(prod, ref),
                "within_tol": _within_tol(op, prod, ref, int(n), fs),
            }
            records.append(rec)
    by_op = defaultdict(list)
    for r in records:
        by_op[r["op"]].append(r)
    summary = {"n_paired": len(records), "n_skipped": skipped, "by_op": {}}
    all_ok = True
    for op in OPS:
        rows = by_op[op]
        if not rows:
            summary["by_op"][op] = {"n": 0, "n_within_tol": 0, "frac_within_tol": None}
            all_ok = False
            continue
        worst = max(rows, key=lambda x: x["abs_error"])
        n_ok = sum(1 for x in rows if x["within_tol"])
        frac = n_ok / len(rows)
        if frac < 1.0:
            all_ok = False
        summary["by_op"][op] = {
            "n": len(rows),
            "max_abs_error": max(x["abs_error"] for x in rows),
            "max_rel_error": max(x["rel_error"] for x in rows),
            "mean_abs_error": float(np.mean([x["abs_error"] for x in rows])),
            "n_within_tol": n_ok,
            "frac_within_tol": frac,
            "worst": {
                "window_id": worst["window_id"],
                "dataset": worst["dataset"],
                "production": worst["production"],
                "reference": worst["reference"],
                "abs_error": worst["abs_error"],
            },
        }
    summary["decision"] = "PASS" if all_ok else "STOP"
    return records, summary


def gate_c_table() -> dict:
    path = RESULTS.parents[0] / "p2" / "primitive_validation.json"
    if not path.exists():
        return {"available": False}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {"available": True, "payload": raw}


def run() -> dict:
    records, summary = pair_windows()
    RESULTS.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "experiment_id": EXPERIMENT_IDS["m3"],
            "seed": SEED,
            "run_count": 1,
        },
        "summary": summary,
        "gate_c": gate_c_table(),
        "records": records,
    }
    (RESULTS / "kernel_numerical_run.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return payload
