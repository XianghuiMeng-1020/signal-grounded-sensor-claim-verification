"""Semantic IR. No waveform, DSP, verdict, or dataset fields."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Optional

IR_VERSION = "p35_semantic_ir_v1"
ATOMIC = "ATOMIC"
COMPOSITE = "COMPOSITE"
CONDITIONAL = "CONDITIONAL"
UNSUPPORTED = "UNSUPPORTED_LANGUAGE"
AMBIGUOUS = "AMBIGUOUS_LANGUAGE"
MISSING = "MISSING_REQUIRED_INFORMATION"
UNCERTAIN = (UNSUPPORTED, AMBIGUOUS, MISSING)

PRIMITIVES = (
    "dominant_frequency",
    "rms_amplitude",
    "peak_amplitude",
    "signal_range",
    "trend_ratio",
    "periodicity_strength",
    "spectral_energy_ratio_low",
    "cross_channel_lag_ms",
)
IR_OPERATORS = ("EQ", "GREATER_THAN", "LESS_THAN", "SIMILAR", "DIFFERENT")
COMPOSITE_OPS = ("AND", "OR")

OP_TO_CMP = {
    "EQ": "eq",
    "GREATER_THAN": "gt",
    "LESS_THAN": "lt",
    "SIMILAR": "similar",
    "DIFFERENT": "different",
}
CMP_TO_OP = {v: k for k, v in OP_TO_CMP.items()}

FORBIDDEN_IR_KEYS = (
    "gold_composed_verdict",
    "gold_program",
    "channels_data",
    "waveform",
    "actual",
    "dsp",
    "dataset",
    "subject",
    "window_id",
    "source_dataset",
)


def ir_catalog() -> dict[str, Any]:
    return {
        "ir_version": IR_VERSION,
        "node_types": [ATOMIC, COMPOSITE, CONDITIONAL, UNSUPPORTED, AMBIGUOUS, MISSING],
        "primitives": list(PRIMITIVES),
        "operators": list(IR_OPERATORS),
        "composite_operators": list(COMPOSITE_OPS),
        "atomic_fields": [
            "type", "primitive", "operator", "threshold", "unit",
            "channel", "source_channel", "target_channel", "reference_channel",
        ],
        "forbidden": list(FORBIDDEN_IR_KEYS),
    }


def ir_schema_hash() -> str:
    return hashlib.sha256(json.dumps(ir_catalog(), sort_keys=True).encode()).hexdigest()


def _reject_forbidden(node: dict) -> Optional[str]:
    extra = set(node) & set(FORBIDDEN_IR_KEYS)
    if extra:
        return f"forbidden_ir_keys:{sorted(extra)}"
    return None


TYPE_ALIASES = {
    "ATOMIC": ATOMIC,
    "ATOM": ATOMIC,
    "COMPOSITE": COMPOSITE,
    "CONDITIONAL": CONDITIONAL,
    "IF_THEN": CONDITIONAL,
    "IFTHEN": CONDITIONAL,
    "IMPLICATION": CONDITIONAL,
    UNSUPPORTED: UNSUPPORTED,
    "UNSUPPORTED": UNSUPPORTED,
    AMBIGUOUS: AMBIGUOUS,
    "AMBIGUOUS": AMBIGUOUS,
    MISSING: MISSING,
    "MISSING": MISSING,
}

OP_ALIASES = {
    "EQ": "EQ", "EQUAL": "EQ", "EQUALS": "EQ", "EQUAL_TO": "EQ",
    "GREATER_THAN": "GREATER_THAN", "GT": "GREATER_THAN", "GREATER": "GREATER_THAN",
    "LESS_THAN": "LESS_THAN", "LT": "LESS_THAN", "LESS": "LESS_THAN",
    "SIMILAR": "SIMILAR", "DIFFERENT": "DIFFERENT",
    "AND": "AND", "OR": "OR",
}

PRIM_ALIASES = {
    "rms": "rms_amplitude",
    "rms_amplitude": "rms_amplitude",
    "root_mean_square": "rms_amplitude",
    "peak": "peak_amplitude",
    "peak_amplitude": "peak_amplitude",
    "range": "signal_range",
    "signal_range": "signal_range",
    "peak_to_peak": "signal_range",
    "lag": "cross_channel_lag_ms",
    "cross_channel_lag_ms": "cross_channel_lag_ms",
    "periodicity": "periodicity_strength",
    "periodicity_strength": "periodicity_strength",
    "low_band_ratio": "spectral_energy_ratio_low",
    "spectral_energy_ratio_low": "spectral_energy_ratio_low",
    "dominant_frequency": "dominant_frequency",
    "trend_ratio": "trend_ratio",
}


def normalize_llm_ir(node: Any) -> Any:
    """Interface parser only. Does not invent predicates or rewrite implication."""
    if not isinstance(node, dict):
        return node
    if "type" not in node:
        for k in ("ir", "semantic_ir", "node", "claim_ir"):
            if isinstance(node.get(k), dict):
                node = node[k]
                break
    out = dict(node)
    raw_t = str(out.get("type") or "").strip().upper().replace(" ", "_").replace("-", "_")
    if raw_t in ("AND", "OR") and out.get("children"):
        out["operator"] = raw_t
        raw_t = COMPOSITE
    out["type"] = TYPE_ALIASES.get(raw_t, raw_t)
    if "operator" in out and out["operator"] is not None:
        op = str(out["operator"]).strip().upper().replace(" ", "_").replace("-", "_")
        out["operator"] = OP_ALIASES.get(op, op)
    if "primitive" in out and out["primitive"] is not None:
        prim = str(out["primitive"]).strip().lower().replace(" ", "_").replace("-", "_")
        out["primitive"] = PRIM_ALIASES.get(prim, prim)
    if out.get("type") == COMPOSITE:
        out["children"] = [normalize_llm_ir(c) for c in (out.get("children") or [])]
    if out.get("type") == CONDITIONAL:
        if "antecedent" in out:
            out["antecedent"] = normalize_llm_ir(out.get("antecedent"))
        if "consequent" in out:
            out["consequent"] = normalize_llm_ir(out.get("consequent"))
    return out


def validate_ir(node: Any) -> tuple[Optional[dict], Optional[str]]:
    node = normalize_llm_ir(node)
    if not isinstance(node, dict):
        return None, "ir_not_object"
    err = _reject_forbidden(node)
    if err:
        return None, err
    t = node.get("type")
    if t in UNCERTAIN:
        return {"type": t, "reason": node.get("reason")}, None
    if t == ATOMIC:
        prim = node.get("primitive")
        if prim not in PRIMITIVES:
            return None, f"bad_primitive:{prim}"
        op = node.get("operator")
        if op not in IR_OPERATORS:
            return None, f"bad_operator:{op}"
        out = {
            "type": ATOMIC,
            "primitive": prim,
            "operator": op,
            "threshold": node.get("threshold"),
            "unit": node.get("unit"),
            "channel": node.get("channel"),
            "source_channel": node.get("source_channel"),
            "target_channel": node.get("target_channel"),
            "reference_channel": node.get("reference_channel"),
        }
        if prim == "cross_channel_lag_ms":
            if not out["source_channel"] or not out["target_channel"]:
                return None, "lag_needs_source_and_target"
            if out["reference_channel"]:
                return None, "lag_forbids_reference_channel"
            out["channel"] = None
        else:
            if out["operator"] in ("SIMILAR", "DIFFERENT") or (
                out["operator"] in ("GREATER_THAN", "LESS_THAN") and out.get("reference_channel")
            ):
                if not out.get("channel") or not out.get("reference_channel"):
                    return None, "vs_channel_needs_pair"
            elif not out.get("channel"):
                return None, "atomic_needs_channel"
            out["source_channel"] = None
            out["target_channel"] = None
        return out, None
    if t == COMPOSITE:
        op = node.get("operator")
        if op not in COMPOSITE_OPS:
            return None, f"bad_composite:{op}"
        kids = []
        for ch in node.get("children") or []:
            k, e = validate_ir(ch)
            if e:
                return None, e
            kids.append(k)
        if not kids:
            return None, "empty_composite"
        return {"type": COMPOSITE, "operator": op, "children": kids}, None
    if t == CONDITIONAL:
        a, e = validate_ir(node.get("antecedent"))
        if e:
            return None, e
        b, e = validate_ir(node.get("consequent"))
        if e:
            return None, e
        return {"type": CONDITIONAL, "antecedent": a, "consequent": b}, None
    return None, f"bad_ir_type:{t}"


def ir_key(node: dict) -> str:
    return json.dumps(node, sort_keys=True, default=str)


def irs_equal(a: dict, b: dict) -> bool:
    return ir_key(_canon_ir(a)) == ir_key(_canon_ir(b))


def _canon_ir(node: dict) -> dict:
    t = node.get("type")
    if t in UNCERTAIN:
        return {"type": t}
    if t == ATOMIC:
        return {k: node.get(k) for k in (
            "type", "primitive", "operator", "threshold", "unit",
            "channel", "source_channel", "target_channel", "reference_channel",
        )}
    if t == COMPOSITE:
        kids = [_canon_ir(c) for c in node.get("children") or []]
        if node.get("operator") in ("AND", "OR"):
            kids = sorted(kids, key=ir_key)
        return {"type": COMPOSITE, "operator": node.get("operator"), "children": kids}
    if t == CONDITIONAL:
        return {
            "type": CONDITIONAL,
            "antecedent": _canon_ir(node["antecedent"]),
            "consequent": _canon_ir(node["consequent"]),
        }
    return node
