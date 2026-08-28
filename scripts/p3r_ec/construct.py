"""Build EC-DEV and EC-BLIND before any production repair evaluation."""
from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

from p2.independent_dsp import MeasurementError, measure, tolerance_for
from p2r.schema import ClaimProgram, Predicate
from p3.windows_p3 import unique_windows

from .config import (
    BENCH,
    HARD_INVALID,
    MISSINGNESS,
    OPS,
    ORACLE_SPEC,
    RESULTS,
    SEED_BLIND,
    SEED_DEV,
    VALID_CTRL,
)
from .guard import refuse_legacy_p3_stress
from .independent_oracle import gold_status, gold_verdict, required_names


def _sha_obj(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def _split_windows(wins: list[dict]) -> tuple[list[dict], list[dict]]:
    dev, blind = [], []
    for w in wins:
        key = f"{w.get('dataset')}|{w.get('subject')}|{w['window_id']}"
        h = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
        (dev if h % 2 == 0 else blind).append(w)
    return dev, blind


def _named(op: str, a: str, b: str | None, data: dict) -> dict:
    names = required_names(op, a, b)
    return {n: data.get(n) for n in names}


def _pick(w: dict, op: str) -> tuple[str, str | None] | None:
    chs = [c for c in w["available_channels"] if w["channels"].get(c) is not None]
    if ORACLE_SPEC[op]["n_channels"] == 2:
        if len(chs) < 2:
            return None
        return chs[0], chs[1]
    if not chs:
        return None
    return chs[0], None


def _apply(family: str, op: str, data0: dict, fs: float, a: str, b: str | None, rng: np.random.Generator):
    data = {k: np.asarray(v, dtype=np.float64).copy() for k, v in data0.items() if v is not None}
    nfs = fs
    req = required_names(op, a, b)
    extra = [k for k in data if k not in req]

    if family == "clean":
        pass
    elif family == "mild_noise":
        for k, x in data.items():
            p = float(np.mean(x ** 2) + 1e-12)
            data[k] = x + rng.normal(0, np.sqrt(p / 100.0), size=x.shape)
    elif family == "quantize_8bit":
        for k, x in data.items():
            lo, hi = float(np.min(x)), float(np.max(x))
            if hi <= lo:
                continue
            data[k] = np.round((x - lo) / (hi - lo) * 255.0) / 255.0 * (hi - lo) + lo
    elif family == "scale_x2":
        data = {k: v * 2.0 for k, v in data.items()}
    elif family == "resample_x2_meta_ok":
        nfs = fs * 2.0
        data = {k: np.repeat(v, 2) for k, v in data.items()}
    elif family == "unused_channel_nan":
        if not extra:
            return None
        x = data[extra[0]].astype(float)
        x[::7] = np.nan
        data[extra[0]] = x
    elif family == "unused_channel_dropout":
        if not extra:
            return None
        data.pop(extra[0], None)
    elif family == "truncate_above_min":
        mn = ORACLE_SPEC[op]["min_n"]
        keep = max(mn, min(x.size for x in data.values()) // 2)
        if keep < mn:
            return None
        data = {k: v[:keep] for k, v in data.items()}
    elif family == "missing_required_channel":
        data.pop(req[0], None)
    elif family == "required_all_nan":
        for n in req:
            if n in data:
                data[n] = np.full(data[n].shape, np.nan)
    elif family == "empty_channel":
        data[req[0]] = np.asarray([], dtype=np.float64)
    elif family == "insufficient_n":
        mn = ORACLE_SPEC[op]["min_n"]
        n = max(0, mn - 1)
        if req[0] not in data:
            return None
        data[req[0]] = data[req[0]][:n]
        if b and b in data and ORACLE_SPEC[op]["n_channels"] == 2:
            data[b] = data[b][:n]
    elif family == "invalid_fs":
        if not ORACLE_SPEC[op]["required_fs"]:
            return None
        nfs = 0.0
    elif family == "missing_fs":
        if not ORACLE_SPEC[op]["required_fs"]:
            return None
        nfs = None
    elif family == "sparse_nan_required":
        for n in req:
            x = data[n].astype(float)
            m = rng.random(x.size) < 0.05
            if not np.any(m):
                m[0] = True
            x[m] = np.nan
            data[n] = x
    elif family == "contiguous_gap_required":
        for n in req:
            x = data[n].astype(float)
            lo = min(40, max(1, x.size // 4))
            hi = min(x.size, lo + max(8, x.size // 8))
            x[lo:hi] = np.nan
            data[n] = x
    elif family == "multiple_gaps_required":
        for n in req:
            x = data[n].astype(float)
            if x.size < 20:
                return None
            x[2:6] = np.nan
            x[x.size // 2 : x.size // 2 + 4] = np.nan
            data[n] = x
    elif family == "partial_required_dropout":
        x = data[req[0]].astype(float)
        x[x.size // 2 :] = np.nan
        data[req[0]] = x
    elif family == "async_pair_support":
        if ORACLE_SPEC[op]["n_channels"] != 2 or b is None or b not in data:
            return None
        xa = data[a].astype(float)
        xb = data[b].astype(float)
        xa[: xa.size // 3] = np.nan
        xb[xb.size // 2 :] = np.nan
        data[a], data[b] = xa, xb
    else:
        return None
    return data, nfs


def _program(op: str, a: str, b: str | None, asserted: float) -> ClaimProgram:
    return ClaimProgram(
        "SINGLE",
        [Predicate(op, a, "eq", channel_b=b, reference_value=float(asserted), reference_channel=None, unit=None)],
    )


def _build_pool(wins: list[dict], seed: int, target_inv: int, target_val: int, label: str) -> dict:
    rng = np.random.default_rng(seed)
    rows = []
    inv_fams = list(HARD_INVALID + MISSINGNESS)
    val_fams = list(VALID_CTRL)
    # cycle windows × ops × families
    i = 0
    n_inv = n_val = 0
    safety = 0
    while (n_inv < target_inv or n_val < target_val) and safety < 200000:
        safety += 1
        w = wins[i % len(wins)]
        op = OPS[(i // len(wins)) % len(OPS)]
        i += 1
        pair = _pick(w, op)
        if pair is None:
            continue
        a, b = pair
        need_inv = n_inv < target_inv
        family = (inv_fams if need_inv else val_fams)[(n_inv if need_inv else n_val) % len(inv_fams if need_inv else val_fams)]
        data0 = {k: np.asarray(v, dtype=np.float64) for k, v in w["channels"].items()}
        named0 = _named(op, a, b, data0)
        if gold_status(op, named0, w["fs"])["status"] != "VALID":
            continue
        try:
            actual = float(measure(op, named0, w["fs"]))
        except MeasurementError:
            continue
        force_false = ((n_inv + n_val) % 2 == 1)
        asserted = actual + (3.5 * tolerance_for(op, actual) if force_false else 0.0)
        applied = _apply(family, op, data0, float(w["fs"]), a, b, rng)
        if applied is None:
            continue
        pdata, pfs = applied
        named = _named(op, a, b, pdata)
        st = gold_status(op, named, pfs)
        gv = gold_verdict(op, named, pfs, asserted)
        is_inv = st["status"] != "VALID"
        if need_inv and not is_inv:
            continue
        if (not need_inv) and is_inv:
            continue
        cid = hashlib.sha256(f"{label}|{w['window_id']}|{op}|{family}|{n_inv}|{n_val}".encode()).hexdigest()[:16]
        prog = _program(op, a, b, asserted)
        rec = {
            "claim_id": cid,
            "split": label,
            "dataset": w.get("dataset"),
            "subject": w.get("subject"),
            "window_id": w["window_id"],
            "family": family,
            "family_class": "invalid" if is_inv else "valid",
            "hard_invalid": family in HARD_INVALID,
            "primitive": op,
            "channel_a": a,
            "channel_b": b,
            "fs": pfs,
            "available_channels": list(w["available_channels"]),
            "asserted": asserted,
            "gold_status": st["status"],
            "gold_reason": st.get("reason"),
            "gold_verdict": gv,
            "program": prog.to_dict(),
            "channels_data": {k: (None if v is None else np.asarray(v, dtype=float).tolist()) for k, v in pdata.items()},
        }
        rows.append(rec)
        if is_inv:
            n_inv += 1
        else:
            n_val += 1
    return {"rows": rows, "n_invalid": n_inv, "n_valid": n_val, "n": len(rows)}


def _manifest(pack: dict, label: str) -> dict:
    slim = []
    for r in pack["rows"]:
        slim.append({k: r[k] for k in r if k != "channels_data"})
    return {
        "label": label,
        "n": pack["n"],
        "n_invalid": pack["n_invalid"],
        "n_valid": pack["n_valid"],
        "sha256_manifest": _sha_obj(slim),
        "by_family": _count(pack["rows"], "family"),
        "by_primitive": _count(pack["rows"], "primitive"),
        "by_gold_status": _count(pack["rows"], "gold_status"),
        "items": slim,
    }


def _count(rows, key):
    from collections import Counter
    return dict(Counter(r[key] for r in rows))


def construct_both():
    refuse_legacy_p3_stress()
    RESULTS.mkdir(parents=True, exist_ok=True)
    BENCH.mkdir(parents=True, exist_ok=True)
    wins = unique_windows()
    if any("holdout" in str(w.get("split_source") or "").lower() for w in wins):
        raise RuntimeError("holdout window leaked into P3R-EC")
    dev_w, blind_w = _split_windows(wins)
    # targets
    dev = _build_pool(dev_w, SEED_DEV, target_inv=360, target_val=360, label="EC-DEV")
    blind = _build_pool(blind_w, SEED_BLIND, target_inv=450, target_val=450, label="EC-BLIND")
    m_dev = _manifest(dev, "EC-DEV")
    m_blind = _manifest(blind, "EC-BLIND")
    (RESULTS / "ec_dev_rows.json").write_text(json.dumps(dev["rows"]), encoding="utf-8")
    (RESULTS / "ec_blind_rows.json").write_text(json.dumps(blind["rows"]), encoding="utf-8")
    (RESULTS / "ec_dev_manifest.json").write_text(json.dumps(m_dev, indent=2), encoding="utf-8")
    (RESULTS / "ec_blind_manifest.json").write_text(json.dumps(m_blind, indent=2), encoding="utf-8")
    (BENCH / "ec_dev.manifest.json").write_text(json.dumps({k: m_dev[k] for k in m_dev if k != "items"}, indent=2), encoding="utf-8")
    (BENCH / "ec_blind.manifest.json").write_text(json.dumps({k: m_blind[k] for k in m_blind if k != "items"}, indent=2), encoding="utf-8")
    summary = {
        "n_windows_dev": len(dev_w),
        "n_windows_blind": len(blind_w),
        "ec_dev": {k: m_dev[k] for k in ("n", "n_invalid", "n_valid", "sha256_manifest", "by_family", "by_primitive", "by_gold_status")},
        "ec_blind": {k: m_blind[k] for k in ("n", "n_invalid", "n_valid", "sha256_manifest", "by_family", "by_primitive", "by_gold_status")},
        "disjoint_windows": len({w["window_id"] for w in dev_w} & {w["window_id"] for w in blind_w}) == 0,
        "old_p3_1040_used": False,
        "constructed_before_repair_eval": True,
    }
    (RESULTS / "construction_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("CONSTRUCT", json.dumps(summary, indent=2), flush=True)
    return summary
