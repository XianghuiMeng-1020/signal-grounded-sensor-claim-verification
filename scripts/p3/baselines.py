"""Mechanism baselines B2/B3 and ablations. DEVELOPMENT-frozen agent prompt."""
from __future__ import annotations

import json
from typing import Any

import numpy as np

from p2.independent_dsp import MeasurementError, measure
from p2r.contracts import check_contract
from p2r.extractor import extract_b6_baseline
from p2r.kleene import compose, verdict_from_tv
from p2r.pipeline import run_oracle, run_pipeline
from p2r.validator import from_legacy
from f_round6_operators import compute as prod_compute

from .config import PRIMARY_MODEL, SEED
from .eval_common import _gold, _summarize
from .io_util import program_from_dict
from .llm_chat import cached_chat

AGENT_PROMPT = """You verify a sensor claim using measurement tools.
Allowed tools (names only): dominant_frequency, rms_amplitude, peak_amplitude, signal_range, trend_ratio, cross_channel_lag_ms, periodicity_strength, spectral_energy_ratio_low.
You receive the claim and available channel names plus fs.
Reply JSON:
{"tool_calls":[{"name":"...","channels":["..."]}]}
or {"final_verdict":"SUPPORTED"|"CONTRADICTED"|"UNVERIFIABLE","reason":"..."}
Do not invent numeric measurements. You may call 1-4 tools. Do not use gold labels."""

ADJ_PROMPT = """You are given a claim, its extracted semantic program, and the numeric evidence from DSP tools.
Return JSON {"verdict":"SUPPORTED"|"CONTRADICTED"|"UNVERIFIABLE"}.
Do not recompute measurements. Judge from the provided numbers and program only."""


def _tools(name, channels, data, fs):
    subset = {c: data.get(c) for c in channels}
    if any(v is None for v in subset.values()):
        return {"error": "missing_channel", "name": name}
    try:
        val = float(measure(name, subset, fs))
        return {"name": name, "channels": channels, "value": val}
    except MeasurementError as exc:
        return {"error": str(exc), "name": name}


def run_tool_agent(rows, model: str = PRIMARY_MODEL, max_tools: int = 4) -> dict:
    recs = []
    stats = {"invalid": 0, "omitted": 0, "invented": 0, "calls": 0}
    for row in rows:
        data = {k: np.asarray(v, dtype=float) for k, v in row["channels_data"].items() if v is not None}
        user = json.dumps({
            "claim": row["surface_text"],
            "available_channels": row["available_channels"],
            "fs": row["fs"],
        })
        r1 = cached_chat("b2_plan_v1", model, [
            {"role": "system", "content": AGENT_PROMPT},
            {"role": "user", "content": user},
        ], seed=SEED, temperature=0.0, fmt="json")
        obj = _json(r1.get("raw"))
        calls = list((obj or {}).get("tool_calls") or [])[:max_tools]
        results = []
        for c in calls:
            stats["calls"] += 1
            name = c.get("name")
            chs = c.get("channels") or []
            if name not in (
                "dominant_frequency", "rms_amplitude", "peak_amplitude", "signal_range",
                "trend_ratio", "cross_channel_lag_ms", "periodicity_strength", "spectral_energy_ratio_low",
            ):
                stats["invalid"] += 1
                results.append({"error": "unknown_tool", "name": name})
                continue
            results.append(_tools(name, chs, data, row["fs"]))
        gold = _gold(row)
        needed = {p.measurement for p in gold.predicates}
        used = {c.get("name") for c in calls}
        if gold.predicates and not (needed & used):
            stats["omitted"] += 1
        if (obj or {}).get("final_verdict") and not calls:
            # agent judged without tools
            if gold.predicates:
                stats["omitted"] += 1
        r2 = cached_chat("b2_judge_v1", model, [
            {"role": "system", "content": AGENT_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": r1.get("raw") or ""},
            {"role": "user", "content": "Tool results: " + json.dumps(results) + " Now return {\"final_verdict\":...}"},
        ], seed=SEED, temperature=0.0, fmt="json")
        obj2 = _json(r2.get("raw")) or {}
        pv = str(obj2.get("final_verdict") or (obj or {}).get("final_verdict") or "UNVERIFIABLE").upper()
        if pv not in ("SUPPORTED", "CONTRADICTED", "UNVERIFIABLE"):
            pv = "UNVERIFIABLE"
        recs.append({
            "claim_id": row["claim_id"],
            "dataset": row.get("dataset"),
            "source": row.get("source"),
            "gold_verdict": row["gold_composed_verdict"],
            "pred_verdict": pv,
            "correct": pv == row["gold_composed_verdict"],
            "exact": False,
            "canonical": False,
            "pred_program": {"parse_status": "AGENT"},
            "error": "agent",
            "tool_calls": calls,
            "tool_results": results,
        })
    n = len(recs)
    return {
        **_summarize(f"B2:{model}", recs, {}, n),
        "agent_stats": {**stats, "invalid_omitted_invented_rate": (stats["invalid"] + stats["omitted"] + stats["invented"]) / max(1, n)},
    }


def run_llm_adjudicator(rows, primary_recs, model: str = PRIMARY_MODEL) -> dict:
    by = {r["claim_id"]: r for r in primary_recs}
    recs = []
    for row in rows:
        pred = by[row["claim_id"]]["pred_program"]
        gold = _gold(row)
        data = {k: np.asarray(v, dtype=float) for k, v in row["channels_data"].items() if v is not None}
        evidence = []
        prog = program_from_dict(pred) if pred.get("predicates") is not None else gold
        use = prog if prog.parse_status == "OK" and prog.predicates else gold
        for p in use.predicates:
            chs = [p.channel_a] + ([p.channel_b] if p.channel_b else [])
            evidence.append(_tools(p.measurement, chs, data, row["fs"]))
        payload = {
            "claim": row["surface_text"],
            "program": pred,
            "evidence": evidence,
        }
        r = cached_chat("b3_adj_v1", model, [
            {"role": "system", "content": ADJ_PROMPT},
            {"role": "user", "content": json.dumps(payload)},
        ], seed=SEED, temperature=0.0, fmt="json")
        obj = _json(r.get("raw")) or {}
        pv = str(obj.get("verdict") or "UNVERIFIABLE").upper()
        if pv not in ("SUPPORTED", "CONTRADICTED", "UNVERIFIABLE"):
            pv = "UNVERIFIABLE"
        recs.append({
            "claim_id": row["claim_id"],
            "dataset": row.get("dataset"),
            "gold_verdict": row["gold_composed_verdict"],
            "pred_verdict": pv,
            "correct": pv == row["gold_composed_verdict"],
            "exact": True,
            "canonical": True,
            "pred_program": pred,
            "error": "llm_adjudication",
        })
    return _summarize(f"B3:{model}", recs, {}, len(recs))


def _json(raw: str | None):
    if not raw:
        return None
    try:
        return json.loads(raw[raw.find("{") : raw.rfind("}") + 1])
    except Exception:
        return None
