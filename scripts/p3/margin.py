"""Measurement-margin protocol. Normalization frozen before results."""
from __future__ import annotations

import hashlib
import random

from p2.independent_adjudicator import adjudicate as ref_adjudicate
from p2.independent_dsp import MeasurementError, measure, tolerance_for

from .config import MARGIN_BANDS, M1_CANDIDATES, SEED
from .windows_p3 import unique_windows

OPS = (
    "dominant_frequency", "rms_amplitude", "peak_amplitude", "signal_range",
    "trend_ratio", "periodicity_strength", "spectral_energy_ratio_low",
)


def _cid(*p):
    return hashlib.sha256("|".join(map(str, p)).encode()).hexdigest()[:16]


def _val(w, op, ch):
    try:
        return float(measure(op, {ch: w["channels"][ch]}, w["fs"]))
    except MeasurementError:
        return None


def build_margin_set(windows, tag: str, per_band: int = 12) -> list[dict]:
    rng = random.Random(SEED + hash(tag) % 10000)
    rows = []
    for band, frac in MARGIN_BANDS:
        made = 0
        order = list(windows)
        rng.shuffle(order)
        for w in order:
            if made >= per_band:
                break
            op = rng.choice(OPS)
            ch = rng.choice(w["available_channels"])
            actual = _val(w, op, ch)
            if actual is None or abs(actual) < 1e-9:
                continue
            side = rng.choice([-1.0, 1.0])
            thr = actual * (1.0 + side * frac) if abs(actual) > 1e-6 else actual + side * frac
            if op in ("spectral_energy_ratio_low", "periodicity_strength"):
                thr = float(min(1.0, max(0.0, thr)))
                if abs(thr - actual) < 1e-12:
                    continue
            rel = "gt" if actual > thr else "lt"
            st = {"connective": "SINGLE", "predicates": [{"op": op, "channels": [ch], "mode": "vs_threshold", "threshold": float(thr), "relation": rel}]}
            gold = ref_adjudicate({"channels": w["channels"], "fs": w["fs"]}, st)
            text = (
                f"The {ch.split('_')[0]} channel {op.replace('_', ' ')} is "
                f"{'above' if rel=='gt' else 'below'} {thr:.6g}."
            )
            rows.append({
                "claim_id": _cid(tag, w["window_id"], op, band, side, thr),
                "band": band,
                "frac": frac,
                "side": side,
                "dataset": w["dataset"],
                "subject": w["subject"],
                "window_id": w["window_id"],
                "fs": w["fs"],
                "available_channels": w["available_channels"],
                "channels_data": {k: v.tolist() for k, v in w["channels"].items()},
                "semantic_program": st,
                "surface_text": text,
                "gold_composed_verdict": gold["verdict"],
                "actual": actual,
                "threshold": float(thr),
                "primitive": op,
                "split": f"p3_margin_{tag}",
            })
            made += 1
    return rows


def m1_unknown(op: str, actual: float, thr: float, alpha: float) -> bool:
    tol = tolerance_for(op, actual)
    return abs(actual - thr) <= alpha * max(tol, 1e-12)
