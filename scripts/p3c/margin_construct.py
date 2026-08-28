"""MARGIN-CLOSURE-BLIND. Independent measurements. New set, not the P3 n=80."""
from __future__ import annotations

import hashlib
import json
import random

import numpy as np

from p2.independent_adjudicator import adjudicate as ref_adjudicate
from p2.independent_dsp import MeasurementError, measure
from p3.windows_p3 import unique_windows

from .config import BENCH, MARGIN_BANDS, RESULTS, SEED

OPS = (
    "dominant_frequency", "rms_amplitude", "peak_amplitude", "signal_range",
    "trend_ratio", "periodicity_strength", "spectral_energy_ratio_low", "cross_channel_lag_ms",
)


def _cid(*p):
    return hashlib.sha256("|".join(map(str, p)).encode()).hexdigest()[:16]


def _actual(op, chs, data, fs):
    try:
        if op == "cross_channel_lag_ms":
            return float(measure(op, {chs[0]: data[chs[0]], chs[1]: data[chs[1]]}, fs))
        return float(measure(op, {chs[0]: data[chs[0]]}, fs))
    except MeasurementError:
        return None


def _thr(op, actual, frac, side, fs):
    """Normalized scale frozen before evaluation.

    Nonnegative / ratio primitives: relative to |actual| (or 1 if ~0).
    Lag: fraction of the ±30-sample search box in milliseconds.
    """
    if op == "cross_channel_lag_ms":
        box = 30.0 * 1000.0 / float(fs)
        return actual + side * frac * box
    scale = max(abs(actual), 1e-6)
    thr = actual + side * frac * scale
    if op in ("spectral_energy_ratio_low", "periodicity_strength"):
        thr = float(min(1.0, max(0.0, thr)))
    if op in ("rms_amplitude", "peak_amplitude", "signal_range", "trend_ratio", "dominant_frequency"):
        thr = float(max(0.0, thr))
    return float(thr)


def _perturb(data, rng):
    out = {}
    for k, v in data.items():
        x = np.asarray(v, dtype=float)
        p = float(np.mean(x ** 2) + 1e-12)
        out[k] = x + rng.normal(0.0, np.sqrt(p) * 0.03, size=x.shape)
    return out


def construct(target=1000):
    rng = random.Random(SEED + 11)
    nprng = np.random.default_rng(SEED + 11)
    wins = unique_windows()
    rows = []
    i = 0
    while len(rows) < target and i < 50000:
        w = wins[i % len(wins)]
        op = OPS[i % len(OPS)]
        band, frac = MARGIN_BANDS[(i // len(OPS)) % len(MARGIN_BANDS)]
        side = 1.0 if (i // (len(OPS) * len(MARGIN_BANDS))) % 2 == 0 else -1.0
        cond = "clean" if (i % 3) else "perturbed"
        i += 1
        chs = list(w["available_channels"])
        used = chs[:2] if op == "cross_channel_lag_ms" and len(chs) >= 2 else [chs[0]]
        if op == "cross_channel_lag_ms" and len(used) < 2:
            continue
        data0 = {k: np.asarray(v, dtype=float) for k, v in w["channels"].items()}
        data = data0 if cond == "clean" else _perturb(data0, nprng)
        actual = _actual(op, used, data, w["fs"])
        if actual is None:
            continue
        thr = _thr(op, actual, frac, side, w["fs"])
        if abs(thr - actual) < 1e-15 and frac > 0:
            continue
        rel = "gt" if actual > thr else "lt"
        if abs(actual - thr) < 1e-15:
            rel = "gt"
        st = {"connective": "SINGLE", "predicates": [{"op": op, "channels": used, "mode": "vs_threshold", "threshold": thr, "relation": rel}]}
        gold = ref_adjudicate({"channels": data, "fs": w["fs"]}, st)
        pair = " and ".join(c.split("_")[0] for c in used)
        text = (
            f"Boundary note: {op.replace('_', ' ')} on {pair} is "
            f"{'greater than' if rel=='gt' else 'less than'} {thr:.6g}."
        )
        rows.append({
            "claim_id": _cid("mc", w["window_id"], op, band, side, cond, thr),
            "source": "margin_closure",
            "band": band,
            "frac": frac,
            "condition": cond,
            "primitive": op,
            "dataset": w["dataset"],
            "subject": w["subject"],
            "window_id": w["window_id"],
            "fs": w["fs"],
            "available_channels": w["available_channels"],
            "channels_data": {k: v.tolist() for k, v in data.items()},
            "semantic_program": st,
            "surface_text": text,
            "gold_composed_verdict": gold["verdict"],
            "actual": actual,
            "threshold": thr,
            "split": "margin_closure_blind",
        })
    man = {
        "n": len(rows),
        "n_clean": sum(1 for r in rows if r["condition"] == "clean"),
        "n_perturbed": sum(1 for r in rows if r["condition"] == "perturbed"),
        "by_band": {b: sum(1 for r in rows if r["band"] == b) for b, _ in MARGIN_BANDS},
        "by_primitive": {op: sum(1 for r in rows if r["primitive"] == op) for op in OPS},
        "old_p3_margin80_used": False,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    BENCH.mkdir(parents=True, exist_ok=True)
    (RESULTS / "margin_closure_rows.json").write_text(json.dumps(rows), encoding="utf-8")
    slim = [{k: r[k] for k in r if k != "channels_data"} for r in rows]
    man["sha256"] = hashlib.sha256(json.dumps(slim, sort_keys=True, default=str).encode()).hexdigest()
    (RESULTS / "margin_closure_FROZEN.json").write_text(json.dumps(man, indent=2), encoding="utf-8")
    print("MARGIN_CONSTRUCT", man, flush=True)
    return man, rows
