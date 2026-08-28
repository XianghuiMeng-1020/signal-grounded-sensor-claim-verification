"""LS-CLOSURE-BLIND. New surfaces; freeze before PRIMARY. No Qwen QC."""
from __future__ import annotations

import hashlib
import json
import random
import re

from p2.independent_adjudicator import adjudicate as ref_adjudicate
from p2.independent_dsp import MeasurementError, measure
from p2r.schema import ClaimProgram
from p2r.validator import from_legacy
from p3.windows_p3 import unique_windows

from .config import BENCH, GEMMA_GEN, RESULTS, SEED

OPS = (
    "dominant_frequency", "rms_amplitude", "peak_amplitude", "signal_range",
    "trend_ratio", "cross_channel_lag_ms", "periodicity_strength", "spectral_energy_ratio_low",
)
QTY = {
    "dominant_frequency": "dominant frequency",
    "rms_amplitude": "RMS amplitude",
    "peak_amplitude": "peak amplitude",
    "signal_range": "peak-to-peak range",
    "trend_ratio": "second-to-first-half energy ratio",
    "cross_channel_lag_ms": "timing lag",
    "periodicity_strength": "periodicity score",
    "spectral_energy_ratio_low": "sub-3 Hz energy fraction",
}
UNIT = {
    "dominant_frequency": "Hz", "rms_amplitude": "raw units", "peak_amplitude": "raw units",
    "signal_range": "raw units", "trend_ratio": "ratio", "cross_channel_lag_ms": "ms",
    "periodicity_strength": "score", "spectral_energy_ratio_low": "fraction",
}


def _cid(*p):
    return hashlib.sha256("|".join(map(str, p)).encode()).hexdigest()[:16]


def _nm(ch):
    return ch.split("_")[0]


def _meas(op, chs, data, fs):
    try:
        if op == "cross_channel_lag_ms":
            if len(chs) < 2:
                return None
            return float(measure(op, {chs[0]: data[chs[0]], chs[1]: data[chs[1]]}, fs))
        return float(measure(op, {chs[0]: data[chs[0]]}, fs))
    except MeasurementError:
        return None


def _vs_value(w, rng, op, force_false):
    chs = list(w["available_channels"])
    if op == "cross_channel_lag_ms":
        if len(chs) < 2:
            return None
        used = chs[:2]
    else:
        used = [rng.choice(chs)]
    actual = _meas(op, used, w["channels"], w["fs"])
    if actual is None:
        return None
    val = actual if not force_false else actual + (4.0 if actual == 0 else 4.0 * (0.15 * abs(actual) if abs(actual) > 1e-6 else 0.2))
    if op in ("spectral_energy_ratio_low", "periodicity_strength"):
        val = min(1.0, max(0.0, val if not force_false else (0.95 if actual < 0.5 else 0.05)))
    if op in ("rms_amplitude", "peak_amplitude", "signal_range", "trend_ratio", "dominant_frequency"):
        val = max(0.0, val)
    return {"connective": "SINGLE", "predicates": [{"op": op, "channels": used, "mode": "vs_value", "asserted_value": float(val)}]}


def _vs_thr(w, rng, op):
    ch = rng.choice(w["available_channels"])
    actual = _meas(op, [ch], w["channels"], w["fs"])
    if actual is None:
        return None
    rel = rng.choice(["gt", "lt"])
    thr = actual * 0.8 if rel == "gt" else actual * 1.2
    if op in ("spectral_energy_ratio_low", "periodicity_strength"):
        thr = min(0.99, max(0.01, thr))
    return {"connective": "SINGLE", "predicates": [{"op": op, "channels": [ch], "mode": "vs_threshold", "threshold": float(thr), "relation": rel}]}


def _clause(pred, style: str) -> str:
    op = pred["op"]
    qty, unit = QTY[op], UNIT[op]
    if pred["mode"] == "vs_value":
        val = float(pred["asserted_value"])
        chs = pred["channels"]
        pair = " and ".join(_nm(c) for c in chs)
        frames = {
            "record": f"Measurement record: {qty} on {pair} equals {val:.4g} {unit}.",
            "asserted": f"It is asserted that {pair} exhibits a {qty} of {val:.4g} {unit}.",
            "bracket": f"{qty}[{pair}] equals {val:.4g} {unit} on this window.",
            "firmware": f"Despite a firmware-note about build 3.2, {pair} {qty} remains {val:.4g} {unit}.",
            "passive2": f"A {qty} of {val:.4g} {unit} is reported for {pair}.",
            "seconds": f"{pair} timing lag is {val/1000.0:.5f} s." if op == "cross_channel_lag_ms" else f"{pair} {qty} is {val:.4g} {unit}.",
            "percent": f"{pair} places {100*val:.2f} percent of spectral energy below 3 Hz." if op == "spectral_energy_ratio_low" else f"{pair} {qty} = {val:.4g} {unit}.",
            "negvalid": f"It is not the case that {pair} {qty} differs from {val:.4g} {unit}.",
        }
        return frames.get(style, frames["record"])
    a, rel, thr = pred["channels"][0], pred["relation"], float(pred["threshold"])
    word = "exceeds" if rel == "gt" else "is below"
    return f"The {qty} of {_nm(a)} {word} {thr:.4g} {UNIT[op]}."


def render_det(st: dict, style: str) -> str:
    preds = st["predicates"]
    conn = st.get("connective", "SINGLE")
    if conn == "SINGLE":
        return _clause(preds[0], style)
    bits = [_clause(p, style) for p in preds]
    if conn == "AND":
        return "Both statements hold. " + " Also, ".join(bits)
    if conn == "OR":
        return "At least one statement holds. " + " Alternatively, ".join(bits)
    if conn == "IF_THEN" and len(bits) == 2:
        return f"If {bits[0].rstrip('.')} then {bits[1]}"
    return " ".join(bits)


def _nums(text):
    return [float(x) for x in re.findall(r"[-+]?\d*\.\d+|\d+", text)]


def qc(text, st) -> tuple[bool, str]:
    if not text or len(text.split()) < 6:
        return False, "too_short"
    if any(k in text.lower() for k in ("supported", "contradicted", "gold verdict")):
        return False, "leak_word"
    for p in st.get("predicates") or []:
        for ch in p.get("channels") or []:
            if _nm(ch).lower() not in text.lower() and ch.lower() not in text.lower():
                return False, "channel_missing"
        target = p.get("asserted_value", p.get("threshold"))
        if target is None:
            continue
        val = float(target)
        nums = _nums(text)
        if not nums:
            return False, "missing_number"
        if not any(abs(n - val) < 1e-2 * max(1.0, abs(val)) or abs(n - 100 * val) < 2 or abs(n - val / 1000.0) < 1e-3 for n in nums):
            return False, "number_drift"
    return True, "ok"


GEMMA_SYS = """Write ONE English sentence asserting the sensor measurement JSON.
Return JSON {"surface": "..."}.
Name every channel with a short placement word (hand/chest/ankle/x/y/back/thigh).
If a numeric value is given, copy that number into the sentence.
Do not mention verdicts, gold, waveforms, or templates.
Do not copy the P3 phrasing 'The X of Y is approximately'.
Use a laboratory-note style.
"""


def gemma_surface(st, channels, seed):
    from p3.llm_chat import cached_chat
    pred = st["predicates"][0]
    rec = cached_chat(
        "gemma_surface_p3c_v1",
        GEMMA_GEN,
        [{"role": "system", "content": GEMMA_SYS},
         {"role": "user", "content": json.dumps({"measurement": pred.get("op"), "channels": pred.get("channels"), "mode": pred.get("mode"), "value": pred.get("asserted_value"), "threshold": pred.get("threshold"), "relation": pred.get("relation"), "allowed_channels": channels})}],
        seed=seed, temperature=0.5, fmt="json",
    )
    raw = rec.get("raw") or ""
    try:
        obj = json.loads(raw[raw.find("{") : raw.rfind("}") + 1])
        text = str(obj.get("surface") or "").strip()
    except Exception:
        return None, "bad_json"
    return text, rec


def _verdict(st, w):
    gold = from_legacy(st, w["available_channels"])
    if gold.parse_status != "OK":
        return "UNVERIFIABLE"
    return ref_adjudicate({"channels": w["channels"], "fs": w["fs"]}, st)["verdict"]


def _pack(w, st, text, source, kind, family):
    return {
        "claim_id": _cid("lsc", source, kind, w["window_id"], text, json.dumps(st, sort_keys=True)),
        "source": source,
        "surface_kind": kind,
        "family": family,
        "connective": st.get("connective", "SINGLE"),
        "n_pred": len(st.get("predicates") or []),
        "dataset": w["dataset"],
        "subject": w["subject"],
        "window_id": w["window_id"],
        "fs": w["fs"],
        "available_channels": w["available_channels"],
        "channels_data": {k: v.tolist() for k, v in w["channels"].items()},
        "semantic_program": st,
        "surface_text": text,
        "gold_composed_verdict": _verdict(st, w),
        "split": "ls_closure_blind",
    }


def construct(n_det=700, n_gemma=500):
    rng = random.Random(SEED)
    wins = unique_windows()
    rng.shuffle(wins)
    programs = []
    styles = ("record", "asserted", "bracket", "firmware", "passive2", "seconds", "percent", "negvalid")
    for w in wins:
        for op in OPS:
            for ff in (False, True):
                st = _vs_value(w, rng, op, ff)
                if st:
                    programs.append((w, st, "vs_value"))
            if op != "cross_channel_lag_ms":
                st = _vs_thr(w, rng, op)
                if st:
                    programs.append((w, st, "vs_threshold"))
        # compounds
        singles = [p for p in programs if p[0] is w and p[2] == "vs_value"][-4:]
        if len(singles) >= 2:
            a, b = singles[0][1]["predicates"][0], singles[1][1]["predicates"][0]
            for conn in ("AND", "OR"):
                programs.append((w, {"connective": conn, "predicates": [a, b]}, "compound2"))
            if len(singles) >= 3:
                c = singles[2][1]["predicates"][0]
                programs.append((w, {"connective": "AND", "predicates": [a, b, c]}, "compound3"))
            programs.append((w, {"connective": "IF_THEN", "predicates": [a, b]}, "ifthen"))
    rng.shuffle(programs)
    generated, rejected, rows = 0, [], []
    for item in programs:
        if sum(1 for r in rows if r["source"] == "deterministic") >= n_det:
            break
        w, st, fam = item
        style = rng.choice(styles)
        text = render_det(st, style)
        generated += 1
        ok, reason = qc(text, st)
        if not ok:
            rejected.append({"source": "deterministic", "reason": reason})
            continue
        rows.append(_pack(w, st, text, "deterministic", style, fam))
    gkeep = 0
    for i, item in enumerate(programs, 1):
        if gkeep >= n_gemma:
            break
        w, st, fam = item
        if fam.startswith("compound") or fam == "ifthen":
            continue
        generated += 1
        if i == 1 or i % 25 == 0:
            print(f"  gemma p3c {i} kept={gkeep}", flush=True)
        text, meta = gemma_surface(st, w["available_channels"], SEED + i)
        if not text:
            rejected.append({"source": GEMMA_GEN, "reason": meta})
            continue
        ok, reason = qc(text, st)
        if not ok:
            rejected.append({"source": GEMMA_GEN, "reason": reason})
            continue
        rows.append(_pack(w, st, text, GEMMA_GEN, "independent_model", fam))
        gkeep += 1
    causes = {}
    for r in rejected:
        causes[r["reason"] if isinstance(r["reason"], str) else str(r["reason"])] = causes.get(r["reason"] if isinstance(r["reason"], str) else "other", 0) + 1
    summary = {
        "generated_n": generated,
        "retained_n": len(rows),
        "rejected_n": len(rejected),
        "rejection_causes": causes,
        "by_source": {
            "deterministic": sum(1 for r in rows if r["source"] == "deterministic"),
            "gemma3:12b": sum(1 for r in rows if r["source"] == GEMMA_GEN),
            "third_family": 0,
        },
        "THIRD_GENERATOR_NOT_AVAILABLE": True,
        "third_family_reason": "local Ollama inventory contained only qwen3:8b and gemma3:12b; no new third-family model was provisioned",
        "by_connective": {c: sum(1 for r in rows if r["connective"] == c) for c in ("SINGLE", "AND", "OR", "IF_THEN")},
        "n_pred_2": sum(1 for r in rows if r["n_pred"] == 2),
        "n_pred_3": sum(1 for r in rows if r["n_pred"] == 3),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    BENCH.mkdir(parents=True, exist_ok=True)
    (RESULTS / "ls_closure_rows.json").write_text(json.dumps(rows), encoding="utf-8")
    slim = [{k: r[k] for k in r if k != "channels_data"} for r in rows]
    man = {**summary, "sha256": hashlib.sha256(json.dumps(slim, sort_keys=True, default=str).encode()).hexdigest()}
    (RESULTS / "ls_closure_FROZEN.json").write_text(json.dumps(man, indent=2), encoding="utf-8")
    (BENCH / "ls_closure.inference.jsonl").write_text("\n".join(json.dumps({k: r[k] for k in r if k != "channels_data"}) for r in rows), encoding="utf-8")
    print("LS_CONSTRUCT", man, flush=True)
    return man, rows
