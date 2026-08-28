"""Independent language-shift construction. Surfaces frozen before PRIMARY."""
from __future__ import annotations

import hashlib
import json
import random
import re
from typing import Any

import numpy as np

from p2.independent_adjudicator import adjudicate as ref_adjudicate
from p2.independent_dsp import MeasurementError, measure
from p2r.eval_p2r import _gold_program
from p2r.schema import ClaimProgram
from p2r.validator import from_legacy

from .config import SEED
from .numeric_domain import vs_value_in_domain
from .windows_p3 import unique_windows

OP_WORDS = {
    "dominant_frequency": ["dominant frequency", "main oscillatory frequency", "peak spectral frequency"],
    "rms_amplitude": ["RMS amplitude", "root-mean-square amplitude", "RMS level"],
    "peak_amplitude": ["peak amplitude", "largest excursion from the mean", "peak deviation"],
    "signal_range": ["peak-to-peak range", "signal range", "peak-to-peak span"],
    "trend_ratio": ["half-window energy ratio", "second-to-first-half energy ratio", "trend ratio"],
    "cross_channel_lag_ms": ["cross-channel timing lag", "inter-channel delay", "timing offset"],
    "periodicity_strength": ["periodicity strength", "rhythmicity score", "autocorrelation periodicity"],
    "spectral_energy_ratio_low": ["low-frequency spectral energy fraction", "sub-3 Hz energy share", "low-band energy fraction"],
}
UNITS = {
    "dominant_frequency": "Hz",
    "rms_amplitude": "raw units",
    "peak_amplitude": "raw units",
    "signal_range": "raw units",
    "trend_ratio": "ratio",
    "cross_channel_lag_ms": "ms",
    "periodicity_strength": "score_0_1",
    "spectral_energy_ratio_low": "fraction",
}


def _nm(ch: str) -> str:
    return ch.split("_")[0]


def _clip_domain(op: str, v: float) -> float:
    if op in ("spectral_energy_ratio_low", "periodicity_strength"):
        return float(np.clip(v, 0.0, 1.0))
    if op in ("rms_amplitude", "peak_amplitude", "signal_range", "trend_ratio", "dominant_frequency"):
        return float(max(0.0, v))
    return float(v)


def _safe_measure(op, chs, data, fs):
    try:
        if op == "cross_channel_lag_ms":
            return float(measure(op, {chs[0]: data[chs[0]], chs[1]: data[chs[1]]}, fs))
        return float(measure(op, {chs[0]: data[chs[0]]}, fs))
    except MeasurementError:
        return None


def _verdict(structure, window) -> str:
    gold = from_legacy(structure, window["available_channels"])
    if gold.parse_status != "OK" or not gold.predicates:
        return "UNVERIFIABLE"
    ref = ref_adjudicate({"channels": window["channels"], "fs": window["fs"]}, structure)
    return ref["verdict"]


def _make_vs_value(window, rng, op, force_false) -> dict | None:
    chs = window["available_channels"]
    if op == "cross_channel_lag_ms":
        if len(chs) < 2:
            return None
        used = [chs[0], chs[1]]
    else:
        used = [rng.choice(chs)]
    actual = _safe_measure(op, used, window["channels"], window["fs"])
    if actual is None:
        return None
    from p2.independent_dsp import tolerance_for
    tol = tolerance_for(op, actual)
    if force_false:
        claimed = actual + rng.choice([-1.0, 1.0]) * rng.uniform(3.0, 6.0) * max(tol, 1e-6)
        claimed = _clip_domain(op, claimed)
        if abs(claimed - actual) <= 1.1 * tol:
            claimed = _clip_domain(op, actual + (1 if actual < 0.5 else -1) * 4 * max(tol, 1e-3))
            if abs(claimed - actual) <= 1.1 * tol:
                return None
    else:
        claimed = _clip_domain(op, actual)
    pred = {"op": op, "channels": used, "mode": "vs_value", "asserted_value": float(claimed)}
    st = {"connective": "SINGLE", "predicates": [pred]}
    p = from_legacy(st, window["available_channels"])
    if p.parse_status != "OK":
        return None
    ok, _ = vs_value_in_domain(p.predicates[0], window["fs"])
    if not ok:
        return None
    return st


def _make_vs_channel(window, rng, op) -> dict | None:
    chs = window["available_channels"]
    if len(chs) < 2 or op == "cross_channel_lag_ms":
        return None
    a, b = chs[0], chs[1]
    va, vb = _safe_measure(op, [a], window["channels"], window["fs"]), _safe_measure(op, [b], window["channels"], window["fs"])
    if va is None or vb is None:
        return None
    rel = "gt" if va > vb else "lt"
    return {"connective": "SINGLE", "predicates": [{"op": op, "channels": [a], "mode": "vs_channel", "compare_channel": b, "relation": rel}]}


def _make_unv(window, rng) -> dict:
    texts = [
        "Jerk entropy of the recording exceeds 4.2 nats.",
        "The gyroscope z-axis has RMS amplitude of approximately 1.20 raw units.",
        "The movement looks vigorous and athletic.",
        "Based on this window, heart rate was likely above 130 bpm.",
        f"The { _nm(window['available_channels'][0]) } channel RMS is either greater or smaller than 2.0.",
    ]
    return {"connective": "SINGLE", "predicates": []}, rng.choice(texts)


def _cid(*parts) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:16]


def _channel_phrase(ch: str, style: str) -> str:
    n = _nm(ch)
    if style == "alias":
        return f"{n} accelerometer"
    if style == "full":
        return f"{ch} stream"
    return f"{n} channel"


def render_deterministic(pred: dict, kind: str) -> str:
    op = pred["op"]
    qty = OP_WORDS[op][hash(kind) % len(OP_WORDS[op])]
    if pred["mode"] == "vs_value":
        val = float(pred["asserted_value"])
        chs = pred["channels"]
        if op == "cross_channel_lag_ms":
            pair = f"{_channel_phrase(chs[0], kind)} and {_channel_phrase(chs[1], kind)}"
        else:
            pair = _channel_phrase(chs[0], kind)
        unit = UNITS[op]
        if kind == "reorder":
            return f"Approximately {val:.4g} {unit} is the {qty} of {pair}."
        if kind == "passive":
            return f"On {pair}, a {qty} of about {val:.4g} {unit} is observed."
        if kind == "unit":
            if op == "cross_channel_lag_ms":
                return f"{pair} are offset by about {val/1000.0:.5f} seconds."
            if op == "spectral_energy_ratio_low":
                return f"About {100*val:.2f}% of the spectral energy of {pair} sits below 3 Hz."
            if op == "dominant_frequency":
                return f"{pair} oscillates at roughly {val:.4g} cycles per second."
            return f"{pair} {qty} ≈ {val:.4g} {unit}."
        if kind == "distractor":
            return f"Ignoring battery notes and session comments, the {qty} for {pair} is about {val:.4g} {unit}."
        if kind == "irrelevant_number":
            return f"The {qty} of {pair} is about {val:.4g} {unit} (the recording id 17 is irrelevant)."
        if kind == "parenthetical":
            return f"The {qty} of {pair} (computed on this window only) is about {val:.4g} {unit}."
        if kind == "synonym":
            return f"{pair} shows {qty} near {val:.4g} {unit}."
        return f"The {qty} of {pair} is approximately {val:.4g} {unit}."
    if pred["mode"] == "vs_channel":
        a, b, rel = pred["channels"][0], pred["compare_channel"], pred["relation"]
        qty = OP_WORDS[op][0]
        if kind == "reverse_eq":
            flip = "lt" if rel == "gt" else "gt" if rel == "lt" else rel
            if flip == "lt":
                return f"{_channel_phrase(b, kind)} has a higher {qty} than {_channel_phrase(a, kind)}."
            if flip == "gt":
                return f"{_channel_phrase(b, kind)} has a lower {qty} than {_channel_phrase(a, kind)}."
        if kind == "passive":
            return f"A higher {qty} is recorded on {_channel_phrase(a if rel=='gt' else b, kind)} than on {_channel_phrase(b if rel=='gt' else a, kind)}."
        hi, lo = (a, b) if rel == "gt" else (b, a)
        return f"The {_channel_phrase(hi, kind)} has a higher {qty} than the {_channel_phrase(lo, kind)}."
    return "Unsupported claim."


KINDS = ("canonical", "reorder", "passive", "unit", "distractor", "irrelevant_number", "parenthetical", "synonym", "reverse_eq")


def _numbers_in(text: str) -> list[float]:
    return [float(x) for x in re.findall(r"[-+]?\d*\.\d+|\d+", text)]


def qc_surface(text: str, structure: dict, kind: str) -> tuple[bool, str]:
    if not text or len(text) < 12:
        return False, "too_short"
    if any(k in text.lower() for k in ("gold", "supported", "contradicted", "unverifiable", "claim_id")):
        return False, "leak_word"
    preds = structure.get("predicates") or []
    if not preds:
        return True, "unv_ok"
    p0 = preds[0]
    if p0.get("mode") == "vs_value":
        val = float(p0["asserted_value"])
        nums = _numbers_in(text)
        if not nums:
            return False, "missing_number"
        ok = any(abs(n - val) < 1e-2 * max(1.0, abs(val)) or abs(n - 100 * val) < 1.5 or abs(n - val / 1000.0) < 1e-4 for n in nums)
        if not ok:
            return False, "number_drift"
        for ch in p0.get("channels") or []:
            if _nm(ch).lower() not in text.lower() and ch.lower() not in text.lower():
                return False, "channel_missing"
    if p0.get("mode") == "vs_channel":
        for ch in [p0["channels"][0], p0["compare_channel"]]:
            if _nm(ch).lower() not in text.lower() and ch.lower() not in text.lower():
                return False, "channel_missing"
    return True, "ok"


GEMMA_SYS = """You write ONE natural English sentence that asserts the given sensor measurement.
Return JSON only with keys: surface, measurement, channels, comparator, value.
The surface MUST be a full sentence (>= 12 words) that names the channels and, if value is not null, includes that number.
Copy measurement exactly from the input. Copy numeric value exactly.
Do not mention gold, verdicts, waveforms, or templates.
Example input: {"measurement":"rms_amplitude","channels":["hand_accel"],"mode":"vs_value","comparator":"eq","value":1.25,"allowed_channels":["hand_accel","chest_accel"]}
Example output: {"surface":"The RMS amplitude of the hand channel is approximately 1.25 raw units.","measurement":"rms_amplitude","channels":["hand_accel"],"comparator":"eq","value":1.25}
"""


def gemma_surface(structure: dict, channels: list[str], seed: int) -> tuple[str | None, dict]:
    from .llm_chat import cached_chat
    pred = (structure.get("predicates") or [{}])[0]
    intent = {
        "measurement": pred.get("op"),
        "channels": pred.get("channels"),
        "compare_channel": pred.get("compare_channel"),
        "mode": pred.get("mode"),
        "comparator": pred.get("relation") or ("eq" if pred.get("mode") == "vs_value" else None),
        "value": pred.get("asserted_value"),
        "allowed_channels": channels,
    }
    rec = cached_chat(
        "gemma_surface_v2",
        "gemma3:12b",
        [
            {"role": "system", "content": GEMMA_SYS},
            {"role": "user", "content": json.dumps(intent)},
        ],
        seed=seed,
        temperature=0.4,
        fmt="json",
    )
    raw = rec.get("raw") or ""
    try:
        obj = json.loads(raw[raw.find("{") : raw.rfind("}") + 1])
        text = str(obj.get("surface") or "").strip()
    except Exception:
        return None, {"reason": "bad_json", "raw": raw[:300]}
    # structured back-reference
    if obj.get("measurement") and obj.get("measurement") != pred.get("op"):
        return None, {"reason": "backref_measurement", "obj": obj}
    if pred.get("mode") == "vs_value" and obj.get("value") is not None:
        try:
            if abs(float(obj["value"]) - float(pred["asserted_value"])) > 1e-2 * max(1.0, abs(float(pred["asserted_value"]))):
                return None, {"reason": "backref_value", "obj": obj}
        except Exception:
            return None, {"reason": "backref_value_parse", "obj": obj}
    return text, {"obj": obj, "latency_s": rec.get("latency_s"), "cache_hit": rec.get("cache_hit")}


def build_language_shift(n_target: int = 1000, gemma_n: int = 400) -> dict:
    rng = random.Random(SEED)
    windows = unique_windows()
    rng.shuffle(windows)
    ops = [
        "dominant_frequency", "rms_amplitude", "peak_amplitude", "signal_range",
        "trend_ratio", "cross_channel_lag_ms", "periodicity_strength", "spectral_energy_ratio_low",
    ]
    programs = []
    for w in windows:
        for op in ops:
            for ff in (False, True):
                st = _make_vs_value(w, rng, op, ff)
                if st:
                    programs.append((w, st, "vs_value"))
            st = _make_vs_channel(w, rng, op)
            if st:
                programs.append((w, st, "vs_channel"))
        st, unv_text = _make_unv(w, rng)
        programs.append((w, st, "unv", unv_text))
    rng.shuffle(programs)

    generated = []
    rejected = []
    # deterministic
    for item in programs:
        if len([g for g in generated if g["source"] == "deterministic"]) >= n_target - gemma_n:
            break
        w, st = item[0], item[1]
        kind = rng.choice([k for k in KINDS if not (k == "reverse_eq" and (st.get("predicates") or [{}])[0].get("mode") != "vs_channel")])
        if item[2] == "unv":
            text = item[3]
            ok, reason = True, "unv_ok"
        else:
            text = render_deterministic((st.get("predicates") or [{}])[0], kind)
            ok, reason = qc_surface(text, st, kind)
        if not ok:
            rejected.append({"source": "deterministic", "reason": reason, "text": text})
            continue
        generated.append(_pack(w, st, text, "deterministic", kind, item[2]))

    # gemma
    gcount = 0
    gattempts = 0
    for item in programs:
        if gcount >= gemma_n or gattempts >= gemma_n * 2:
            break
        if item[2] == "unv":
            continue
        w, st = item[0], item[1]
        gattempts += 1
        if gattempts == 1 or gattempts % 25 == 0:
            print(f"  gemma surface {gattempts} kept={gcount}", flush=True)
        text, meta = gemma_surface(st, w["available_channels"], SEED + gattempts)
        if not text:
            rejected.append({"source": "gemma3:12b", "reason": meta.get("reason"), "meta": {k: meta[k] for k in meta if k != "obj"}})
            continue
        ok, reason = qc_surface(text, st, "gemma")
        if not ok:
            rejected.append({"source": "gemma3:12b", "reason": reason, "text": text})
            continue
        generated.append(_pack(w, st, text, "gemma3:12b", "independent_model", item[2], extra={"gen_meta": {k: meta.get(k) for k in ("latency_s", "cache_hit")}}))
        gcount += 1

    retained = generated
    return {
        "generated_n": len(generated) + len(rejected),
        "retained_n": len(retained),
        "rejected_n": len(rejected),
        "rejection_causes": {k: sum(1 for r in rejected if r.get("reason") == k) for k in {r.get("reason") for r in rejected}},
        "by_source": {
            "deterministic": sum(1 for r in retained if r["source"] == "deterministic"),
            "gemma3:12b": sum(1 for r in retained if r["source"] == "gemma3:12b"),
            "public_human": 0,
        },
        "rows": retained,
        "rejected_head": rejected[:40],
    }


def _pack(window, structure, text, source, kind, family, extra=None) -> dict:
    if family == "unv":
        verdict = "UNVERIFIABLE"
    else:
        try:
            verdict = _verdict(structure, window)
        except Exception:
            verdict = "UNVERIFIABLE"
    rec = {
        "claim_id": _cid(window["window_id"], source, kind, text, json.dumps(structure, sort_keys=True)),
        "source": source,
        "surface_kind": kind,
        "family": family,
        "dataset": window["dataset"],
        "subject": window["subject"],
        "window_id": window["window_id"],
        "fs": window["fs"],
        "available_channels": window["available_channels"],
        "channels_data": {k: v.tolist() for k, v in window["channels"].items()},
        "semantic_program": structure,
        "surface_text": text,
        "gold_composed_verdict": verdict,
        "split": "p3_language_shift",
    }
    if extra:
        rec.update(extra)
    return rec
