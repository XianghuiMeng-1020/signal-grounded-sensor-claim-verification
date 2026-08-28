"""Explicit pipeline. Extractor never verdicts. DSP never reads text. Adjudicator never calls an LLM."""
from __future__ import annotations

from typing import Any, Callable, Optional

from .kleene import compose, verdict_from_tv
from .schema import ClaimProgram, Predicate
from .validator import inference_view, validate_program
from .executor import predicate_truth


Extractor = Callable[[str, list[str], Optional[float]], ClaimProgram]

# Evaluation-only. Default "production" is the frozen oracle path.
# lag_sample_threshold: lag reference_value is a sample count; convert to ms via fs.
# lag_physical_threshold: lag reference_value is already milliseconds.
EVAL_MODE_PRODUCTION = "production"
EVAL_MODE_LAG_SAMPLE = "lag_sample_threshold"
EVAL_MODE_LAG_PHYSICAL = "lag_physical_threshold"


def apply_lag_eval_mode(program: ClaimProgram, fs: float, eval_mode: str) -> ClaimProgram:
    """Rewrite lag numeric thresholds for an evaluation mode. Kernels unchanged."""
    if eval_mode == EVAL_MODE_PRODUCTION:
        return program
    if eval_mode not in (EVAL_MODE_LAG_SAMPLE, EVAL_MODE_LAG_PHYSICAL):
        raise ValueError(f"unknown eval_mode: {eval_mode}")
    preds = []
    for pred in program.predicates:
        if pred.measurement != "cross_channel_lag_ms" or pred.reference_value is None:
            preds.append(pred)
            continue
        if eval_mode == EVAL_MODE_LAG_SAMPLE:
            ref = float(pred.reference_value) * 1000.0 / float(fs)
        else:
            ref = float(pred.reference_value)
        preds.append(
            Predicate(
                measurement=pred.measurement,
                channel_a=pred.channel_a,
                comparator=pred.comparator,
                channel_b=pred.channel_b,
                reference_value=ref,
                reference_channel=pred.reference_channel,
                unit="ms",
            )
        )
    return ClaimProgram(
        program.connective,
        preds,
        parse_status=program.parse_status,
        parse_reason=program.parse_reason,
    )


def run_pipeline(
    surface_text: str,
    available_channels: list[str],
    fs: float,
    channel_data: dict,
    extractor: Extractor,
) -> dict[str, Any]:
    inf = inference_view(surface_text, available_channels, fs)
    program = extractor(inf["surface_text"], inf["available_channels"], inf["fs"])
    if not isinstance(program, ClaimProgram):
        raise TypeError("extractor must return ClaimProgram, not a verdict")
    if hasattr(program, "verdict"):
        raise TypeError("extractor assigned a verdict")
    validated = validate_program(program, available_channels)
    if validated.parse_status != "OK":
        return {
            "program": validated.to_dict(),
            "predicate_tvs": [],
            "composed_tv": "UNKNOWN",
            "verdict": "UNVERIFIABLE",
            "reason": validated.parse_reason,
            "stage": "schema",
        }
    tvs, evs = [], []
    for pred in validated.predicates:
        tv, ev = predicate_truth(pred, channel_data, fs)
        tvs.append(tv)
        evs.append(ev)
    composed = compose(validated.connective, tvs)
    return {
        "program": validated.to_dict(),
        "predicate_tvs": tvs,
        "evidence": evs,
        "composed_tv": composed,
        "verdict": verdict_from_tv(composed),
        "stage": "adjudication",
    }


def run_oracle(
    program: ClaimProgram,
    available_channels: list[str],
    fs: float,
    channel_data: dict,
    eval_mode: str = EVAL_MODE_PRODUCTION,
) -> dict[str, Any]:
    """Gold schema → validator → contracts → production DSP → Kleene. Diagnostic only.

    eval_mode is evaluation-only. The default production path does not convert
    thresholds. Kernels and contracts are not switched.
    """
    scored = apply_lag_eval_mode(program, fs, eval_mode)
    validated = validate_program(scored, available_channels)
    if validated.parse_status != "OK":
        return {"verdict": "UNVERIFIABLE", "reason": validated.parse_reason, "program": validated.to_dict()}
    tvs = [predicate_truth(p, channel_data, fs)[0] for p in validated.predicates]
    return {
        "verdict": verdict_from_tv(compose(validated.connective, tvs)),
        "predicate_tvs": tvs,
        "program": validated.to_dict(),
        "eval_mode": eval_mode,
    }
