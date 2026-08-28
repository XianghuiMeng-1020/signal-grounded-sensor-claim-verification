"""V3 surfaces. Same families as v2, except vs_threshold numbers are written."""
from __future__ import annotations

from .language_realizations import (
    STYLES,
    _join,
    _name,
    _op_words,
    _rel_words,
    render_predicate as render_predicate_v2,
)
from .config import SURFACES_PER_PROGRAM


def _thr_words_v3(op: str, ch: str, rel: str, thr) -> str:
    n = _name(ch)
    if op == "trend_ratio":
        return (
            f"energy in the {n} channel is increasing across the window"
            if rel == "gt"
            else f"energy in the {n} channel is decreasing or flat across the window"
        )
    qty = _op_words(op)
    if thr is None:
        raise ValueError("v3 self-containment: numeric threshold required")
    side = "above" if rel == "gt" else "below"
    unit = {
        "rms_amplitude": "raw units",
        "peak_amplitude": "raw units",
        "signal_range": "raw units",
        "periodicity_strength": "on a 0-1 scale",
        "spectral_energy_ratio_low": "fraction",
        "dominant_frequency": "Hz",
        "cross_channel_lag_ms": "ms",
        "trend_ratio": "ratio",
    }.get(op, "raw units")
    return f"the {n} channel {qty} is {side} {float(thr):.3f} {unit}"


def render_predicate(pred: dict, style: str) -> str:
    if pred.get("mode") == "vs_value" and pred.get("op") == "cross_channel_lag_ms" and style == "implicit":
        chs = pred["channels"]
        val = pred["asserted_value"]
        return (
            f"cross-channel timing lag is about {val:.3f} ms on the "
            f"{_name(chs[0])} and {_name(chs[1])} channels"
        )
    if pred["mode"] != "vs_threshold":
        return render_predicate_v2(pred, style)
    a = pred["channels"][0]
    rel = pred["relation"]
    core = _thr_words_v3(pred["op"], a, rel, pred.get("threshold"))
    if style == "synonym":
        return core.replace("energy", "AC energy")
    if style == "distractor":
        return core + ", irrespective of ambient notes"
    if style == "colloquial":
        return "in plain terms, " + core
    return core


def realize(structure: dict, split: str, extra_styles: tuple[str, ...] = ()) -> list[dict]:
    n = SURFACES_PER_PROGRAM[split]
    conn = structure.get("connective", "SINGLE")
    styles_cycle = list(STYLES)
    if extra_styles:
        styles_cycle = list(extra_styles) + styles_cycle
    out = []
    for i in range(n):
        style = styles_cycle[i % len(styles_cycle)]
        join_styles = [styles_cycle[(i + j) % len(styles_cycle)] for j in range(3)]
        bits = [render_predicate(p, join_styles[j % len(join_styles)]) for j, p in enumerate(structure["predicates"])]
        if conn == "SINGLE":
            text = bits[0] + "."
        elif conn == "AND":
            text = bits[0] + ", and " + bits[1] + "." if len(bits) == 2 else ", ".join(bits[:-1]) + ", and " + bits[-1] + "."
        elif conn == "OR":
            text = bits[0] + ", OR alternatively " + (" OR alternatively ".join(bits[1:])) + "."
        elif conn == "IF_THEN":
            text = "If " + bits[0] + ", then " + bits[1] + "."
        else:
            text = "; ".join(bits) + "."
        out.append({
            "surface_text": text[0].upper() + text[1:],
            "surface_style": style,
            "paraphrase_index": i,
        })
    return out
