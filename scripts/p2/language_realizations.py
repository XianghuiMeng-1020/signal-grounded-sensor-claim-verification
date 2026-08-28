"""Surface realizations for a fixed semantic program.

Renderer does not write paraphrase_family into any inference field.
Family IDs exist only in the gold sidecar.
"""
from __future__ import annotations

from typing import Callable

from .config import SURFACES_PER_PROGRAM


def _name(ch: str) -> str:
    return ch.split("_")[0]


def _op_words(op: str) -> str:
    return {
        "dominant_frequency": "dominant frequency",
        "rms_amplitude": "RMS amplitude",
        "peak_amplitude": "peak amplitude",
        "signal_range": "peak-to-peak range",
        "trend_ratio": "half-window energy ratio",
        "cross_channel_lag_ms": "cross-channel timing lag",
        "periodicity_strength": "periodicity strength",
        "spectral_energy_ratio_low": "low-frequency spectral energy fraction",
    }[op]


def _rel_words(rel: str, a: str, b: str, op: str) -> str:
    an, bn = _name(a), _name(b)
    qty = _op_words(op)
    if rel == "gt":
        return f"the {an} channel has a higher {qty} than the {bn} channel"
    if rel == "lt":
        return f"the {an} channel has a lower {qty} than the {bn} channel"
    if rel == "similar":
        return f"the {an} and {bn} channels have similar {qty}"
    if rel == "different":
        return f"the {an} and {bn} channels have clearly different {qty}"
    raise KeyError(rel)


def _thr_words(op: str, ch: str, rel: str) -> str:
    n = _name(ch)
    if op == "trend_ratio":
        return (
            f"energy in the {n} channel is increasing across the window"
            if rel == "gt"
            else f"energy in the {n} channel is decreasing or flat across the window"
        )
    if op == "periodicity_strength":
        return (
            f"the {n} channel is clearly periodic/rhythmic"
            if rel == "gt"
            else f"the {n} channel is not strongly periodic"
        )
    qty = _op_words(op)
    return f"the {n} channel {qty} is {'above' if rel == 'gt' else 'below'} threshold"


def render_predicate(pred: dict, style: str) -> str:
    op = pred["op"]
    mode = pred["mode"]
    if mode == "vs_value":
        chs = pred["channels"]
        val = pred["asserted_value"]
        qty = _op_words(op)
        if op == "cross_channel_lag_ms":
            subj = f"the {_name(chs[0])} and {_name(chs[1])} channels"
        else:
            subj = f"the {_name(chs[0])} channel"
        unit = {
            "dominant_frequency": "Hz",
            "cross_channel_lag_ms": "ms",
            "rms_amplitude": "raw units",
            "peak_amplitude": "raw units",
            "signal_range": "raw units",
            "trend_ratio": "x",
            "periodicity_strength": "on a 0-1 scale",
            "spectral_energy_ratio_low": "fraction",
        }[op]
        if style == "direct":
            return f"the {qty} of {subj} is approximately {val:.3f} {unit}"
        if style == "comparative":
            return f"{subj} shows {qty} near {val:.3f} {unit}, rather than a markedly different value"
        if style == "reordered":
            return f"approximately {val:.3f} {unit} is the {qty} measured on {subj}"
        if style == "synonym":
            return f"{subj} has a {qty} of about {val:.3f} {unit}"
        if style == "units":
            if op == "spectral_energy_ratio_low":
                return f"about {100 * val:.1f} percent of {subj} spectral energy sits below 3 Hz"
            if op == "dominant_frequency":
                return f"{subj} oscillates at roughly {val:.3f} cycles per second"
            if op == "cross_channel_lag_ms":
                return f"{subj} are offset by about {val / 1000.0:.5f} seconds"
            return f"{subj} {qty} ≈ {val:.3f} {unit}"
        if style == "implicit":
            return f"{qty} is about {val:.3f} {unit} on {_name(chs[0])}"
        if style == "distractor":
            return (
                f"ignoring battery status and session notes, {qty} for {subj} is about {val:.3f} {unit}"
            )
        if style == "numeric_fmt":
            return f"{subj} {qty} = {val:.6f} {unit}"
        if style == "colloquial":
            return f"looks like {subj} sits around {val:.3f} for {qty}"
        if style == "negation_false":
            return f"{subj} does not have a {qty} anywhere near {val:.3f} {unit}"
        return f"{qty} of {subj} is {val:.3f} {unit}"

    if mode == "vs_channel":
        a = pred["channels"][0]
        b = pred["compare_channel"]
        rel = pred["relation"]
        core = _rel_words(rel, a, b, op)
        if style == "reordered":
            return core.replace("the ", "", 1)
        if style == "synonym":
            return core.replace("channel", "sensor stream")
        if style == "implicit":
            return core.replace(" channel", "")
        if style == "distractor":
            return core + " (strap tightness was not recorded)"
        if style == "colloquial":
            return "looks like " + core
        if style == "comparative":
            return "compared with each other, " + core
        return core

    if mode == "vs_threshold":
        a = pred["channels"][0]
        rel = pred["relation"]
        core = _thr_words(op, a, rel)
        if style == "synonym":
            return core.replace("energy", "AC energy")
        if style == "distractor":
            return core + ", irrespective of ambient notes"
        if style == "colloquial":
            return "in plain terms, " + core
        return core
    raise KeyError(mode)


STYLES = (
    "direct",
    "comparative",
    "reordered",
    "synonym",
    "units",
    "implicit",
    "distractor",
    "numeric_fmt",
    "colloquial",
    "conjunction",
    "disjunction",
    "conditional",
)


def _join(preds: list[dict], connective: str, styles: list[str]) -> str:
    bits = [render_predicate(p, styles[i % len(styles)]) for i, p in enumerate(preds)]
    if connective == "SINGLE":
        text = bits[0] + "."
    elif connective == "AND":
        if len(bits) == 2:
            text = bits[0] + ", and " + bits[1] + "."
        else:
            text = ", ".join(bits[:-1]) + ", and " + bits[-1] + "."
    elif connective == "OR":
        text = bits[0] + ", OR alternatively " + (" OR alternatively ".join(bits[1:])) + "."
    elif connective == "IF_THEN":
        text = "If " + bits[0] + ", then " + bits[1] + "."
    else:
        text = "; ".join(bits) + "."
    return text[0].upper() + text[1:]


def realize(structure: dict, split: str, extra_styles: tuple[str, ...] = ()) -> list[dict]:
    n = SURFACES_PER_PROGRAM[split]
    conn = structure.get("connective", "SINGLE")
    styles_cycle = list(STYLES)
    if extra_styles:
        styles_cycle = list(extra_styles) + styles_cycle
    out = []
    for i in range(n):
        style = styles_cycle[i % len(styles_cycle)]
        # vary joining style independently of clause style
        join_styles = [styles_cycle[(i + j) % len(styles_cycle)] for j in range(3)]
        if conn == "AND" and style == "conjunction":
            text = _join(structure["predicates"], "AND", join_styles)
        elif conn == "OR" and style == "disjunction":
            text = _join(structure["predicates"], "OR", join_styles)
        elif conn == "IF_THEN" and style == "conditional":
            text = _join(structure["predicates"], "IF_THEN", join_styles)
        else:
            text = _join(structure["predicates"], conn, join_styles)
        out.append({
            "surface_text": text,
            "surface_style": style,
            "paraphrase_index": i,
        })
    return out
