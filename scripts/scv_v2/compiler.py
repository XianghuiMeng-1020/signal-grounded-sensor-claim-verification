"""SCV V2: deterministic fail-closed layer over frozen V1 extract_ollama.

Does not modify DSP kernels, thresholds, Kleene tables, or V1 sources.
"""
from __future__ import annotations

from typing import Any, Optional

from p2r.ollama_adapter import SEED, extract_ollama
from p2r.schema import ClaimProgram, Predicate

from .grounding import find_channel_mentions, resolve_mention_to_available, split_clauses
from .typecheck import type_status


def _fail(reason: str, traces: list[dict]) -> tuple[ClaimProgram, list[dict]]:
    prog = ClaimProgram("SINGLE", [], parse_status="UNPARSEABLE", parse_reason=f"v2:{reason}")
    return prog, traces


def _pred_channels(pred: Predicate) -> list[str]:
    out = [pred.channel_a]
    if pred.channel_b:
        out.append(pred.channel_b)
    if pred.reference_channel:
        out.append(pred.reference_channel)
    return out


def apply_v2(
    sentence: str,
    v1: ClaimProgram,
    available_channels: list[str],
    fs: Optional[float] = None,
) -> tuple[ClaimProgram, list[dict]]:
    """Enforce No Silent Measurement Substitution on a V1 program."""
    traces: list[dict] = []
    tstat, treason = type_status(v1, fs, sentence)
    if tstat != "VALID":
        traces.append({"grounding_status": tstat, "reason": treason, "evidence_license_status": "not_executed"})
        return _fail(treason, traces)

    clauses = split_clauses(sentence, v1.connective)
    if len(clauses) != len(v1.predicates):
        # If we cannot align clauses, treat the whole sentence as one bag but
        # still require every predicate channel to be source-grounded.
        clause_mentions = [find_channel_mentions(sentence) for _ in v1.predicates]
        global_mentions = find_channel_mentions(sentence)
    else:
        clause_mentions = [find_channel_mentions(c) for c in clauses]
        global_mentions = find_channel_mentions(sentence)

    used: set[tuple[int, int, str]] = set()
    new_preds: list[Predicate] = []
    for i, pred in enumerate(v1.predicates):
        local = clause_mentions[i] if i < len(clause_mentions) else global_mentions
        grounded_valid: dict[str, object] = {}
        for ment in local:
            cid, st = resolve_mention_to_available(ment, available_channels)
            traces.append({
                "source_span": ment.span,
                "resolved_channel": cid,
                "operator": pred.measurement,
                "unit": pred.unit,
                "comparator": pred.comparator,
                "threshold": pred.reference_value,
                "grounding_status": st,
                "evidence_license_status": "pending",
            })
            if st == "UNSUPPORTED":
                return _fail("unsupported_channel", traces)
            if st == "AMBIGUOUS":
                return _fail("ambiguous_channel", traces)
            if st == "VALID" and cid:
                grounded_valid[cid] = ment
        needed = _pred_channels(pred)
        for ch in needed:
            if ch not in grounded_valid:
                return _fail(f"ungrounded_or_substituted:{ch}", traces)
            if ch not in available_channels:
                return _fail(f"missing_grounded_channel:{ch}", traces)
        for cid, ment in grounded_valid.items():
            if cid not in needed:
                return _fail(f"unaccounted_mention:{cid}", traces)
            used.add((ment.start, ment.end, cid))  # type: ignore[attr-defined]
        new_preds.append(pred)

    for ment in global_mentions:
        if ment.status == "UNSUPPORTED":
            return _fail("unsupported_channel_unaccounted", traces)
        if ment.status == "AMBIGUOUS":
            return _fail("ambiguous_channel_unaccounted", traces)
        if ment.status == "VALID" and ment.candidates:
            cid = ment.candidates[0]
            bound = {ch for p in new_preds for ch in _pred_channels(p)}
            if cid not in bound:
                return _fail(f"unaccounted_mention:{cid}", traces)

    ok = ClaimProgram(v1.connective, new_preds, parse_status="OK", parse_reason=None)
    for t in traces:
        t["evidence_license_status"] = "eligible_for_license"
    return ok, traces


def extract_v2(
    sentence: str,
    available_channels: list[str],
    fs: Optional[float] = None,
    model: str = "",
    prompt_version: str = "v2",
    seed: int = SEED,
    use_cache: bool = True,
) -> tuple[ClaimProgram, dict[str, Any]]:
    """V1 extract then V2 gate. One LLM call; V1 remains independently recoverable."""
    v1, meta = extract_ollama(
        sentence,
        available_channels,
        fs,
        model=model,
        prompt_version=prompt_version,
        seed=seed,
        use_cache=use_cache,
    )
    v2, traces = apply_v2(sentence, v1, available_channels, fs)
    meta = dict(meta)
    meta["v1_parse_status"] = v1.parse_status
    meta["v1_parse_reason"] = v1.parse_reason
    meta["v1_program"] = v1.to_dict()
    meta["v2_traces"] = traces
    meta["compiler_version"] = "scv_v2"
    return v2, meta
