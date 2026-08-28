"""MARGIN2-DEV (oracle check) and MARGIN2-BLIND. New windows/programs."""
from __future__ import annotations

import hashlib
import json
import random

import numpy as np

from p2.independent_adjudicator import adjudicate as ref_adjudicate
from p2.independent_dsp import MeasurementError, measure
from p3.windows_p3 import unique_windows

from .config import BENCH, MARGIN_BANDS, RESULTS, SEED, OPS
from .sem_construct import _split_windows


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


def construct_margin(wins, target, seed, split, include_invalid=True):
    rng = random.Random(seed)
    nprng = np.random.default_rng(seed)
    rows = []
    i = 0
    while len(rows) < target and i < 80000:
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
            f"Boundary ledger: {op.replace('_', ' ')} on {pair} is "
            f"{'greater than' if rel == 'gt' else 'less than'} {thr:.6g}."
        )
        rows.append({
            "claim_id": _cid("m2", split, w["window_id"], op, band, side, cond, thr),
            "source": "margin2",
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
            "gold_evidence_status": "ANSWERABLE" if gold["verdict"] != "UNVERIFIABLE" else "UNVERIFIABLE",
            "actual": actual,
            "threshold": thr,
            "equality_cell": abs(actual - thr) < 1e-12 or band == "exact",
            "split": split,
            "family": "valid_boundary",
        })
    if include_invalid:
        extra = min(80, max(40, target // 20))
        for j in range(extra):
            w = wins[j % len(wins)]
            kind = ("missing_channel", "invalid_fs")[j % 2]
            data = {k: np.asarray(v, dtype=float) for k, v in w["channels"].items()}
            if kind == "missing_channel":
                st = {"connective": "SINGLE", "predicates": [{"op": "rms_amplitude", "channels": ["wrist_accel"], "mode": "vs_threshold", "threshold": 1.0, "relation": "gt"}]}
                text = "Boundary ledger: rms amplitude on wrist is greater than 1.0."
                avail = list(w["available_channels"])
                fs = w["fs"]
                gold_v = "UNVERIFIABLE"
            else:
                st = {"connective": "SINGLE", "predicates": [{"op": "dominant_frequency", "channels": [w["available_channels"][0]], "mode": "vs_threshold", "threshold": 2.0, "relation": "gt"}]}
                text = "Boundary ledger: dominant frequency on " + w["available_channels"][0].split("_")[0] + " is greater than 2.0, but sampling rate metadata is invalid."
                avail = list(w["available_channels"])
                fs = 0.0
                gold_v = "UNVERIFIABLE"
            rows.append({
                "claim_id": _cid("m2inv", split, w["window_id"], kind, j),
                "source": "margin2",
                "band": "invalid_control",
                "frac": None,
                "condition": "invalid",
                "primitive": st["predicates"][0]["op"],
                "dataset": w["dataset"],
                "subject": w["subject"],
                "window_id": w["window_id"],
                "fs": fs,
                "available_channels": avail,
                "channels_data": {k: v.tolist() for k, v in data.items()},
                "semantic_program": st,
                "surface_text": text,
                "gold_composed_verdict": gold_v,
                "gold_evidence_status": "UNVERIFIABLE",
                "actual": None,
                "threshold": None,
                "equality_cell": False,
                "split": split,
                "family": kind,
            })
    return rows


def _write(name, rows, extra):
    RESULTS.mkdir(parents=True, exist_ok=True)
    BENCH.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{name}_rows.json").write_text(json.dumps(rows), encoding="utf-8")
    slim = [{k: r[k] for k in r if k != "channels_data"} for r in rows]
    man = {
        **extra,
        "n": len(rows),
        "n_valid": sum(1 for r in rows if r["gold_evidence_status"] == "ANSWERABLE"),
        "n_invalid": sum(1 for r in rows if r["gold_evidence_status"] == "UNVERIFIABLE"),
        "n_exact": sum(1 for r in rows if r.get("equality_cell") and r["family"] == "valid_boundary"),
        "n_clean": sum(1 for r in rows if r.get("condition") == "clean"),
        "n_perturbed": sum(1 for r in rows if r.get("condition") == "perturbed"),
        "by_primitive": {op: sum(1 for r in rows if r.get("primitive") == op) for op in OPS},
        "old_margin_used": False,
        "sha256": hashlib.sha256(json.dumps(slim, sort_keys=True, default=str).encode()).hexdigest(),
    }
    (RESULTS / f"{name}_FROZEN.json").write_text(json.dumps(man, indent=2), encoding="utf-8")
    print("MARGIN2_CONSTRUCT", name, {k: man[k] for k in ("n", "n_valid", "n_invalid", "sha256")}, flush=True)
    return man


def construct(n_dev=240, n_blind=1200):
    wins = unique_windows()
    dev_w, blind_w = _split_windows(wins)
    drows = construct_margin(dev_w, n_dev, SEED + 31, "margin2_dev", include_invalid=True)
    brows = construct_margin(blind_w, n_blind, SEED + 47, "margin2_blind", include_invalid=True)
    return _write("margin2_dev", drows, {"split": "margin2_dev"}), _write("margin2_blind", brows, {"split": "margin2_blind"})
