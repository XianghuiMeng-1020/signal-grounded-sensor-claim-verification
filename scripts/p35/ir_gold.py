"""Deterministic gold IR from the construction structure. No LLM."""
from __future__ import annotations

from p2r.schema import ClaimProgram
from p2r.validator import from_legacy

from .ir_schema import AMBIGUOUS, ATOMIC, COMPOSITE, CONDITIONAL, MISSING, UNSUPPORTED, validate_ir
from .compiler import compile_ir

SCHEMA_UNIT = {
    "dominant_frequency": "Hz",
    "rms_amplitude": "raw",
    "peak_amplitude": "raw",
    "signal_range": "raw",
    "trend_ratio": "ratio",
    "periodicity_strength": "score_0_1",
    "spectral_energy_ratio_low": "fraction",
    "cross_channel_lag_ms": "ms",
}

UNV_TO_IR = {
    "unsupported_measurement": UNSUPPORTED,
    "unsupported_logical_structure": UNSUPPORTED,
    "unresolved_channel": AMBIGUOUS,
    "missing_required_evidence": MISSING,
    "invalid_metadata": MISSING,
    "genuine_language_ambiguity": AMBIGUOUS,
}


def gold_ir_from_st(st: dict) -> dict:
    if st.get("unverifiable") or not st.get("predicates"):
        fam = st.get("unv_family") or "unsupported_measurement"
        return {"type": UNV_TO_IR.get(fam, UNSUPPORTED), "reason": fam}
    atoms = [_atom(p) for p in st["predicates"]]
    conn = st.get("connective", "SINGLE")
    if conn == "SINGLE":
        node = atoms[0]
    elif conn in ("AND", "OR"):
        node = {"type": COMPOSITE, "operator": conn, "children": atoms}
    elif conn == "IF_THEN":
        node = {"type": CONDITIONAL, "antecedent": atoms[0], "consequent": atoms[1]}
    else:
        return {"type": UNSUPPORTED, "reason": "bad_connective"}
    ir, err = validate_ir(node)
    if err or ir is None:
        raise ValueError(f"gold IR invalid: {err}")
    return ir


def _atom(p: dict) -> dict:
    op = p["op"]
    if p.get("mode") == "vs_value" or p.get("comparator") == "eq":
        operator = "EQ"
        thr = p.get("asserted_value", p.get("reference_value"))
    elif p.get("mode") == "vs_threshold":
        operator = "GREATER_THAN" if p.get("relation") == "gt" else "LESS_THAN"
        thr = p.get("threshold", p.get("reference_value"))
    else:
        operator = {"gt": "GREATER_THAN", "lt": "LESS_THAN", "eq": "EQ"}.get(
            p.get("relation") or p.get("comparator"), "EQ"
        )
        thr = p.get("threshold", p.get("asserted_value", p.get("reference_value")))
    node = {
        "type": ATOMIC,
        "primitive": op,
        "operator": operator,
        "threshold": None if thr is None else float(thr),
        "unit": SCHEMA_UNIT.get(op),
    }
    chs = list(p.get("channels") or [])
    if op == "cross_channel_lag_ms":
        node["source_channel"] = chs[0]
        node["target_channel"] = chs[1]
    else:
        node["channel"] = chs[0]
        if p.get("reference_channel") or p.get("compare_channel"):
            node["reference_channel"] = p.get("reference_channel") or p.get("compare_channel")
    return node


def gold_program_from_st(st: dict, available: list[str]) -> ClaimProgram:
    return from_legacy(st, available)


def compile_matches_gold(st: dict, available: list[str]) -> bool:
    ir = gold_ir_from_st(st)
    compiled = compile_ir(ir, available)
    gold = gold_program_from_st(st, available)
    from p3c.strict_semantic import programs_strictly_equivalent
    return programs_strictly_equivalent(compiled, gold, available)
