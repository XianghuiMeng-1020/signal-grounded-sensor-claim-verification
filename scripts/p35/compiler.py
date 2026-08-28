"""Deterministic IR → ClaimProgram. LLM does not own structure or lag slots."""
from __future__ import annotations

from typing import Optional

from p2r.schema import ClaimProgram, Predicate
from p2r.validator import canonicalize_unit, resolve_channel, validate_program

from .ir_schema import (
    AMBIGUOUS,
    ATOMIC,
    COMPOSITE,
    CONDITIONAL,
    MISSING,
    OP_TO_CMP,
    UNSUPPORTED,
    validate_ir,
)


def compile_ir(node, available: list[str]) -> ClaimProgram:
    ir, err = validate_ir(node)
    if err or ir is None:
        return ClaimProgram("SINGLE", [], parse_status="UNPARSEABLE", parse_reason=err or "invalid_ir")
    return _compile_valid(ir, available)


def _compile_valid(ir: dict, available: list[str]) -> ClaimProgram:
    t = ir["type"]
    if t == UNSUPPORTED:
        return ClaimProgram("SINGLE", [], parse_status="UNPARSEABLE", parse_reason="unsupported_language")
    if t == MISSING:
        return ClaimProgram("SINGLE", [], parse_status="UNPARSEABLE", parse_reason="missing_required_information")
    if t == AMBIGUOUS:
        return ClaimProgram("SINGLE", [], parse_status="AMBIGUOUS", parse_reason="ambiguous_language")
    if t == ATOMIC:
        pred, err = _atomic_pred(ir, available)
        if err or pred is None:
            status = "AMBIGUOUS" if err == "ambiguous_channel" else "UNPARSEABLE"
            return ClaimProgram("SINGLE", [], parse_status=status, parse_reason=err)
        return validate_program(ClaimProgram("SINGLE", [pred], parse_status="OK"), available)
    if t == COMPOSITE:
        preds: list[Predicate] = []
        for child in ir.get("children") or []:
            if child.get("type") != ATOMIC:
                return ClaimProgram("SINGLE", [], parse_status="UNPARSEABLE", parse_reason="composite_child_not_atomic")
            pred, err = _atomic_pred(child, available)
            if err or pred is None:
                status = "AMBIGUOUS" if err == "ambiguous_channel" else "UNPARSEABLE"
                return ClaimProgram("SINGLE", [], parse_status=status, parse_reason=err)
            preds.append(pred)
        if len(preds) > 3:
            return ClaimProgram("SINGLE", [], parse_status="UNPARSEABLE", parse_reason="more_than_3_predicates")
        if not preds:
            return ClaimProgram("SINGLE", [], parse_status="UNPARSEABLE", parse_reason="empty_composite")
        conn = ir["operator"]
        if conn in ("AND", "OR"):
            preds = sorted(preds, key=_pred_key)
        return validate_program(ClaimProgram(conn, preds, parse_status="OK"), available)
    if t == CONDITIONAL:
        a, e1 = _require_atomic(ir["antecedent"], available)
        b, e2 = _require_atomic(ir["consequent"], available)
        err = e1 or e2
        if err or a is None or b is None:
            status = "AMBIGUOUS" if err == "ambiguous_channel" else "UNPARSEABLE"
            return ClaimProgram("SINGLE", [], parse_status=status, parse_reason=err or "bad_conditional")
        # Compiler owns implication: frozen IF_THEN == NOT A OR B in Kleene.
        return validate_program(ClaimProgram("IF_THEN", [a, b], parse_status="OK"), available)
    return ClaimProgram("SINGLE", [], parse_status="UNPARSEABLE", parse_reason="unhandled_ir")


def _require_atomic(node: dict, available: list[str]) -> tuple[Optional[Predicate], Optional[str]]:
    if node.get("type") != ATOMIC:
        return None, "conditional_child_not_atomic"
    return _atomic_pred(node, available)


def _atomic_pred(ir: dict, available: list[str]) -> tuple[Optional[Predicate], Optional[str]]:
    prim = ir["primitive"]
    cmp_ = OP_TO_CMP[ir["operator"]]
    unit, thr = canonicalize_unit(ir.get("unit"), ir.get("threshold"))
    if prim == "cross_channel_lag_ms":
        a, e = resolve_channel(ir.get("source_channel"), available)
        if e:
            return None, e
        b, e = resolve_channel(ir.get("target_channel"), available)
        if e:
            return None, e
        return Predicate(
            measurement=prim,
            channel_a=a,
            comparator=cmp_,
            channel_b=b,
            reference_value=thr,
            reference_channel=None,
            unit=unit or "ms",
        ), None
    ch, e = resolve_channel(ir.get("channel"), available)
    if e:
        return None, e
    ref_c = None
    if ir.get("reference_channel"):
        ref_c, e = resolve_channel(ir.get("reference_channel"), available)
        if e:
            return None, e
    return Predicate(
        measurement=prim,
        channel_a=ch,
        comparator=cmp_,
        channel_b=None,
        reference_value=None if ref_c is not None else thr,
        reference_channel=ref_c,
        unit=unit,
    ), None


def _pred_key(p: Predicate) -> tuple:
    return (
        p.measurement,
        p.channel_a or "",
        p.channel_b or "",
        p.comparator,
        p.reference_channel or "",
        None if p.reference_value is None else round(float(p.reference_value), 6),
    )
