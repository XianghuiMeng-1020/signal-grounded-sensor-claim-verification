"""HARTH-CLOSURE-BLIND on unused subjects. Independent gold. No prompt tuning."""
from __future__ import annotations

import hashlib
import json
import random

import numpy as np

from p2.independent_adjudicator import adjudicate as ref_adjudicate
from p2.independent_dsp import MeasurementError, measure
from p3.harth import download_subjects, windows_from_csv
from p3r_ec.independent_oracle import VALID, gold_status, gold_verdict

from .config import BENCH, HARTH_CLOSURE_SUBJECTS, HARTH_P3_EVAL_SUBJECTS, RESULTS, SEED

OPS = (
    "rms_amplitude", "peak_amplitude", "signal_range", "trend_ratio",
    "dominant_frequency", "periodicity_strength", "spectral_energy_ratio_low", "cross_channel_lag_ms",
)


def _cid(*p):
    return hashlib.sha256("|".join(map(str, p)).encode()).hexdigest()[:16]


def _load_new_windows():
    sids = download_subjects(HARTH_CLOSURE_SUBJECTS)
    wins = []
    for sid in sids:
        wins.extend(windows_from_csv(sid, n_windows=24, start=8000))
    return wins, sids


def _meas(op, chs, data, fs):
    try:
        if op == "cross_channel_lag_ms":
            return float(measure(op, {chs[0]: data[chs[0]], chs[1]: data[chs[1]]}, fs))
        return float(measure(op, {chs[0]: data[chs[0]]}, fs))
    except MeasurementError:
        return None


def _st_value(op, chs, val):
    return {"connective": "SINGLE", "predicates": [{"op": op, "channels": list(chs), "mode": "vs_value", "asserted_value": float(val)}]}


def _text(op, chs, val, rel="eq"):
    pair = " and ".join(c.replace("_accel", "") for c in chs)
    if rel == "eq":
        return f"Lab note: {op.replace('_', ' ')} on {pair} equals {val:.4g}."
    return f"Lab note: {op.replace('_', ' ')} on {pair} is {'above' if rel=='gt' else 'below'} {val:.4g}."


def construct(n_ans=320, n_unv=220):
    rng = random.Random(SEED + 7)
    wins, sids = _load_new_windows()
    if not wins:
        raise RuntimeError("no unused HARTH windows")
    rows = []
    # answerable
    i = 0
    while len([r for r in rows if r["gold_composed_verdict"] != "UNVERIFIABLE"]) < n_ans and i < 20000:
        w = wins[i % len(wins)]
        op = OPS[i % len(OPS)]
        i += 1
        chs = list(w["available_channels"])
        used = chs[:2] if op == "cross_channel_lag_ms" else [chs[i % len(chs)]]
        actual = _meas(op, used, w["channels"], w["fs"])
        if actual is None:
            continue
        force = (i % 2 == 0)
        val = actual if not force else actual + (0.25 * abs(actual) + 0.05)
        if op in ("spectral_energy_ratio_low", "periodicity_strength"):
            val = min(1.0, max(0.0, val if not force else (0.9 if actual < 0.5 else 0.1)))
        st = _st_value(op, used, val)
        gold = ref_adjudicate({"channels": w["channels"], "fs": w["fs"]}, st)["verdict"]
        if gold == "UNVERIFIABLE":
            continue
        named = {used[0]: w["channels"][used[0]]}
        if len(used) == 2:
            named[used[1]] = w["channels"][used[1]]
        if gold_status(op, named, w["fs"])["status"] != VALID:
            continue
        rows.append(_pack(w, st, _text(op, used, val), "answerable_clean", gold, "VALID", sids))
    # unverifiable families
    fams = (
        "missing_required_channel", "invalid_fs", "insufficient_n", "sparse_nan",
        "unsupported_measurement", "unsupported_channel_relation",
    )
    u = 0
    while len([r for r in rows if r["gold_composed_verdict"] == "UNVERIFIABLE"]) < n_unv and u < 20000:
        w = wins[u % len(wins)]
        fam = fams[u % len(fams)]
        op = OPS[u % len(OPS)]
        u += 1
        chs = list(w["available_channels"])
        used = chs[:2] if op == "cross_channel_lag_ms" else [chs[0]]
        data = {k: np.asarray(v, dtype=float).copy() for k, v in w["channels"].items()}
        fs = w["fs"]
        avail = list(chs)
        if fam == "unsupported_measurement":
            text = f"The heart-rate estimate of the {used[0].replace('_accel','')} placement is 72 bpm."
            st = {"connective": "SINGLE", "predicates": [{"op": "heart_rate_estimate", "channels": used, "mode": "vs_value", "asserted_value": 72.0}], "unverifiable": True}
            rows.append(_pack(w, st, text, fam, "UNVERIFIABLE", "UNSUPPORTED", sids, data, fs, avail))
            continue
        if fam == "unsupported_channel_relation":
            text = f"The {chs[0].replace('_accel','')} placement is healthier than the {chs[1].replace('_accel','')} placement."
            st = {"connective": "SINGLE", "predicates": [], "unverifiable": True}
            rows.append(_pack(w, st, text, fam, "UNVERIFIABLE", "UNSUPPORTED", sids, data, fs, avail))
            continue
        actual = _meas(op, used, data, fs)
        if actual is None:
            continue
        st = _st_value(op, used, actual)
        text = _text(op, used, actual)
        if fam == "missing_required_channel":
            data.pop(used[0], None)
        elif fam == "invalid_fs":
            fs = 0.0
        elif fam == "insufficient_n":
            data[used[0]] = data[used[0]][:3]
        elif fam == "sparse_nan":
            x = data[used[0]]
            x[::11] = np.nan
            data[used[0]] = x
        named = {n: data.get(n) for n in used}
        stt = gold_status(op, named, fs)
        gv = "UNVERIFIABLE" if stt["status"] != VALID else gold_verdict(op, named, fs, actual)
        if gv != "UNVERIFIABLE":
            continue
        rows.append(_pack(w, st, text, fam, gv, stt["status"], sids, data, fs, avail))
    man = {
        "dataset": "HARTH",
        "subjects": list(sids),
        "p3_eval_subjects_excluded": list(HARTH_P3_EVAL_SUBJECTS),
        "n": len(rows),
        "n_answerable": sum(1 for r in rows if r["gold_composed_verdict"] != "UNVERIFIABLE"),
        "n_unverifiable": sum(1 for r in rows if r["gold_composed_verdict"] == "UNVERIFIABLE"),
        "by_family": {f: sum(1 for r in rows if r["family"] == f) for f in {r["family"] for r in rows}},
        "prompt_tuning": False,
        "disjoint_from_p3_eval_subjects": True,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    BENCH.mkdir(parents=True, exist_ok=True)
    (RESULTS / "harth_closure_rows.json").write_text(json.dumps(rows), encoding="utf-8")
    slim = [{k: r[k] for k in r if k != "channels_data"} for r in rows]
    man["sha256"] = hashlib.sha256(json.dumps(slim, sort_keys=True, default=str).encode()).hexdigest()
    (RESULTS / "harth_closure_FROZEN.json").write_text(json.dumps(man, indent=2), encoding="utf-8")
    print("HARTH_CONSTRUCT", man, flush=True)
    return man, rows


def _pack(w, st, text, family, gold, estatus, sids, data=None, fs=None, avail=None):
    data = data if data is not None else {k: np.asarray(v) for k, v in w["channels"].items()}
    fs = w["fs"] if fs is None else fs
    avail = list(w["available_channels"]) if avail is None else avail
    return {
        "claim_id": _cid("hc", family, w["window_id"], text, json.dumps(st, sort_keys=True)),
        "source": "harth_closure",
        "family": family,
        "evidence_status": estatus,
        "dataset": "HARTH",
        "subject": w["subject"],
        "window_id": w["window_id"],
        "fs": fs,
        "available_channels": avail,
        "channels_data": {k: (None if v is None else np.asarray(v, dtype=float).tolist()) for k, v in data.items()},
        "semantic_program": st,
        "surface_text": text,
        "gold_composed_verdict": gold,
        "split": "harth_closure_blind",
        "closure_subjects": list(sids),
    }
