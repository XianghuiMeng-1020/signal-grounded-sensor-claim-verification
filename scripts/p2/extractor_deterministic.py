"""Inference-only deterministic schema extractor.

Frozen independently of paraphrase-family IDs and gold metadata.
Receives only: sentence, available channel names, fs.

This is baseline B6 and, in this P2 run (no LLM credentials), the only
available stand-in for the production LLM extractor on new text.
It is NOT claimed to be the Qwen extractor.
"""
from __future__ import annotations

import re
from typing import Any, Optional

OPS = (
    "dominant_frequency",
    "rms_amplitude",
    "peak_amplitude",
    "signal_range",
    "trend_ratio",
    "cross_channel_lag_ms",
    "periodicity_strength",
    "spectral_energy_ratio_low",
)

# Cue lists are semantic, not copies of any one renderer template.
OP_CUES = {
    "dominant_frequency": (
        "dominant frequency", "main frequency", "primary frequency",
        "spectral peak frequency", "frequency peak", "oscillation rate",
        "tonal peak", "largest periodic frequency", "freq peaks",
        "higher frequency than", "lower frequency than", "faster oscillation",
        "slower oscillation", "cycles per second",
    ),
    "rms_amplitude": (
        "rms", "root-mean-square", "root mean square", "quadratic mean",
        "overall amplitude energy", "overall energy amplitude",
        "similar overall", "different overall", "higher rms", "lower rms",
    ),
    "peak_amplitude": (
        "peak amplitude", "peak (mean-removed)", "largest excursion",
        "maximum absolute deviation", "biggest spike", "peak excursion",
    ),
    "signal_range": (
        "peak-to-peak", "peak to peak", "value range", "spans a",
        "larger range", "smaller range", "min-to-max", "min to max",
    ),
    "trend_ratio": (
        "energy is increasing", "energy is decreasing", "second half",
        "first half", "energy grew", "energy dropped", "ramping up",
        "ramping down", "trend ratio", "changed by a factor",
    ),
    "cross_channel_lag_ms": (
        "timing lag", "time lag", "cross-correlation", "cross correlation",
        "lag of", "delayed relative", "millisecond lag", "ms lag",
    ),
    "periodicity_strength": (
        "periodic", "rhythmic", "periodicity", "autocorrelation",
        "more periodic", "less periodic", "not strongly periodic",
        "irregular", "noise-like",
    ),
    "spectral_energy_ratio_low": (
        "low-frequency", "low frequency", "below-3hz", "below 3 hz",
        "below 3hz", "spectral energy", "low-band", "under 3 hz",
    ),
}

UNVERIFIABLE_CUES = (
    "heart rate", "bpm", "emotional", "anxious", "battery",
    "temperature", "celsius", "fatigue", "mood", "stress hormone",
    "blood pressure", "spo2", "oxygen saturation", "cannot be checked",
    "not enough samples", "sampling rate is unknown", "corrupt",
    "missing channel", "which channel", "either greater or smaller",
    "and also if", "nested", "four conditions", "five conditions",
)

REL_GT = ("higher", "more ", "larger", "greater", "above", "increasing", "gt", "faster", "stronger")
REL_LT = ("lower", "less ", "smaller", "below", "decreasing", "lt", "slower", "weaker", "flat")
REL_SIM = ("similar", "approximately equal", "about the same", "comparable")
REL_DIFF = ("different", "dissimilar", "not similar", "clearly different")


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _find_ops(text: str) -> list[str]:
    t = _norm(text)
    hits = []
    for op, cues in OP_CUES.items():
        if any(c in t for c in cues):
            hits.append(op)
    # preserve first-occurrence order
    order = []
    for op in hits:
        pos = min((t.find(c) for c in OP_CUES[op] if c in t), default=10**9)
        order.append((pos, op))
    order.sort()
    seen = set()
    out = []
    for _, op in order:
        if op not in seen:
            seen.add(op)
            out.append(op)
    return out


def _channel_aliases(available: list[str]) -> dict[str, str]:
    aliases = {}
    for name in available:
        aliases[name.lower()] = name
        head = name.split("_")[0].lower()
        aliases[head] = name
        aliases[head + " channel"] = name
        aliases[head + "-axis"] = name
    # common extra words
    extras = {
        "wrist": None,
        "watch": None,
    }
    for name in available:
        if name.startswith("x_"):
            extras["x-axis"] = name
            extras["x axis"] = name
            extras["horizontal"] = name
        if name.startswith("y_"):
            extras["y-axis"] = name
            extras["y axis"] = name
            extras["vertical"] = name
    for k, v in extras.items():
        if v:
            aliases[k] = v
    return aliases


def _find_channels(text: str, available: list[str]) -> list[str]:
    t = _norm(text)
    aliases = _channel_aliases(available)
    found = []
    # longest alias first
    for alias in sorted(aliases, key=len, reverse=True):
        if re.search(rf"\b{re.escape(alias)}\b", t):
            name = aliases[alias]
            if name not in found:
                found.append(name)
    return found


def _numbers(text: str) -> list[float]:
    vals = []
    for m in re.finditer(r"[-+]?\d+(?:\.\d+)?", text or ""):
        raw = m.group(0)
        try:
            vals.append(float(raw))
        except ValueError:
            continue
    return vals


def _relation(text: str) -> Optional[str]:
    t = _norm(text)
    if any(k in t for k in REL_SIM):
        return "similar"
    if any(k in t for k in REL_DIFF):
        return "different"
    # check gt/lt after similar/different
    gt = any(k in t for k in REL_GT)
    lt = any(k in t for k in REL_LT)
    if gt and not lt:
        return "gt"
    if lt and not gt:
        return "lt"
    return None


def _connective(text: str, n_pred: int) -> str:
    t = _norm(text)
    if re.search(r"\bif\b", t) and re.search(r"\bthen\b", t):
        return "IF_THEN"
    if re.search(r"\bor alternatively\b", t) or re.search(r"\bor else\b", t):
        return "OR"
    if re.search(r"\bor\b", t) and n_pred >= 2:
        return "OR"
    if n_pred >= 2:
        return "AND"
    return "SINGLE"


def _looks_unverifiable(text: str) -> bool:
    t = _norm(text)
    return any(c in t for c in UNVERIFIABLE_CUES)


def _percent_to_frac(text: str, number: float) -> float:
    t = _norm(text)
    if "%" in text or "percent" in t:
        if 1.0 < abs(number) <= 100.0:
            return number / 100.0
    return number


def extract(sentence: str, available_channels: list[str], fs: Optional[float] = None) -> dict[str, Any]:
    """Return a structure dict. Never reads gold fields."""
    if _looks_unverifiable(sentence):
        return {"connective": "SINGLE", "predicates": [], "unverifiable": True}

    ops = _find_ops(sentence)
    chs = _find_channels(sentence, list(available_channels or []))
    nums = _numbers(sentence)
    rel = _relation(sentence)

    if not ops:
        return {"connective": "SINGLE", "predicates": [], "unverifiable": True}

    # Split composed sentences on connective boundaries for per-clause ops.
    parts = re.split(r"\b(?:or alternatively|or else|, and also|, while also|, while|, and |, then | then |\bor\b)", sentence, flags=re.I)
    parts = [p.strip() for p in parts if p and p.strip() and _norm(p) not in {"if", "then"}]
    if len(parts) <= 1:
        parts = [sentence]

    predicates = []
    used_ops = ops[:] if ops else []
    for i, part in enumerate(parts):
        pop = _find_ops(part)
        op = pop[0] if pop else (used_ops[i] if i < len(used_ops) else used_ops[0])
        local_ch = _find_channels(part, available_channels) or chs
        local_rel = _relation(part) or rel
        local_nums = _numbers(part) or nums

        if op == "cross_channel_lag_ms":
            pair = local_ch[:2] if len(local_ch) >= 2 else (available_channels[:2] if len(available_channels) >= 2 else local_ch)
            if len(pair) < 2:
                continue
            asserted = local_nums[0] if local_nums else None
            if asserted is None:
                predicates.append({"op": op, "channels": pair, "mode": "vs_value", "asserted_value": None})
            else:
                predicates.append({"op": op, "channels": pair, "mode": "vs_value", "asserted_value": float(asserted)})
            continue

        if len(local_ch) >= 2 and local_rel in ("gt", "lt", "similar", "different"):
            predicates.append({
                "op": op,
                "channels": [local_ch[0]],
                "mode": "vs_channel",
                "compare_channel": local_ch[1],
                "relation": local_rel,
            })
            continue

        if local_rel in ("gt", "lt") and local_ch:
            # qualitative threshold families used in the project
            default_thr = 1.0 if op == "trend_ratio" else (0.35 if op == "periodicity_strength" else None)
            thr = default_thr
            if local_nums:
                cand = _percent_to_frac(part, local_nums[0])
                if op != "periodicity_strength" or cand <= 1.5:
                    thr = cand
            if thr is not None and not _looks_like_vs_value(part, op):
                predicates.append({
                    "op": op,
                    "channels": [local_ch[0]],
                    "mode": "vs_threshold",
                    "threshold": float(thr),
                    "relation": local_rel,
                })
                continue

        if local_nums and local_ch:
            val = _percent_to_frac(part, local_nums[0])
            predicates.append({
                "op": op,
                "channels": [local_ch[0]] if op != "cross_channel_lag_ms" else local_ch[:2],
                "mode": "vs_value",
                "asserted_value": float(val),
            })
            continue

        if local_ch:
            predicates.append({
                "op": op,
                "channels": [local_ch[0]],
                "mode": "vs_threshold" if local_rel in ("gt", "lt") else "vs_value",
                "threshold": 1.0 if op == "trend_ratio" else 0.35,
                "relation": local_rel or "gt",
                "asserted_value": None,
            })
            if predicates[-1]["mode"] == "vs_value":
                predicates[-1] = {
                    "op": op,
                    "channels": [local_ch[0]],
                    "mode": "vs_value",
                    "asserted_value": None,
                }

    # drop empty / unusable
    cleaned = []
    for p in predicates:
        if p.get("mode") == "vs_value" and p.get("asserted_value") is None:
            # keep only if we truly have no number — will become UNVERIFIABLE later
            cleaned.append(p)
        else:
            cleaned.append(p)
    predicates = cleaned

    if not predicates:
        return {"connective": "SINGLE", "predicates": [], "unverifiable": True}

    conn = _connective(sentence, len(predicates))
    if conn == "SINGLE":
        predicates = predicates[:1]
    if conn == "IF_THEN":
        predicates = predicates[:2]
    return {"connective": conn, "predicates": predicates, "unverifiable": False}


def _looks_like_vs_value(text: str, op: str) -> bool:
    t = _norm(text)
    if op in ("trend_ratio", "periodicity_strength") and any(w in t for w in ("increasing", "decreasing", "periodic", "rhythmic", "flat")):
        if not re.search(r"approximately|about|roughly|~", t):
            return False
    return bool(re.search(r"approximately|about|roughly|of [-+]?\d", t))
