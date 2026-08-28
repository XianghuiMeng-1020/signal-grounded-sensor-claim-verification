"""Evaluate production against the independent oracle. No LLM."""
from __future__ import annotations

import json
from collections import Counter, defaultdict

import numpy as np

from p2.stats_ci import proportion
from p2r.executor import predicate_truth
from p2r.kleene import compose, verdict_from_tv
from p2r.schema import ClaimProgram, Predicate
from p2r.validator import validate_program

from .config import HARD_INVALID, RESULTS
from .guard import refuse_legacy_p3_stress
from .independent_oracle import VALID, gold_status, required_names

PROD_OK = {"OK", "VALID"}


def _prog(d: dict) -> ClaimProgram:
    preds = [Predicate(**{k: p[k] for k in ("measurement", "channel_a", "comparator", "channel_b", "reference_value", "reference_channel", "unit")}) for p in d["predicates"]]
    return ClaimProgram(d["connective"], preds, parse_status=d.get("parse_status", "OK"))


def _run_prod(prog: ClaimProgram, available: list, fs, data: dict) -> tuple[str, str]:
    validated = validate_program(prog, available)
    if validated.parse_status != "OK":
        return "UNVERIFIABLE", "INSUFFICIENT_EVIDENCE"
    tvs, statuses = [], []
    for pred in validated.predicates:
        tv, ev = predicate_truth(pred, data, fs)
        tvs.append(tv)
        m = ev.get("measurement")
        if m is not None and hasattr(m, "status"):
            statuses.append(m.status)
        elif tv == "UNKNOWN":
            statuses.append("INSUFFICIENT_EVIDENCE")
        else:
            statuses.append("OK")
    verdict = verdict_from_tv(compose(validated.connective, tvs))
    st = statuses[0] if statuses else "INSUFFICIENT_EVIDENCE"
    mapping = {
        "OK": "VALID",
        "MISSING_CHANNEL": "MISSING_CHANNEL",
        "INSUFFICIENT_EVIDENCE": "INSUFFICIENT_EVIDENCE",
        "INVALID_METADATA": "INVALID_METADATA",
        "UNSUPPORTED": "UNSUPPORTED",
        "INVALID_MEASUREMENT": "INVALID_MEASUREMENT",
    }
    return verdict, mapping.get(st, "INSUFFICIENT_EVIDENCE")


def evaluate_split(rows: list[dict], label: str) -> dict:
    refuse_legacy_p3_stress()
    recs = []
    for i, row in enumerate(rows, 1):
        if i == 1 or i % 200 == 0 or i == len(rows):
            print(f"  {label} {i}/{len(rows)}", flush=True)
        data = {k: (None if v is None else np.asarray(v, dtype=float)) for k, v in row["channels_data"].items()}
        prog = _prog(row["program"])
        named = {n: data.get(n) for n in required_names(row["primitive"], row["channel_a"], row["channel_b"])}
        # never use production to label; gold already stored; recompute as check
        st = gold_status(row["primitive"], named, row["fs"])
        pv, pstatus = _run_prod(prog, row["available_channels"], row["fs"], {k: v for k, v in data.items() if v is not None})
        gold_v = row["gold_verdict"]
        gold_s = row["gold_status"]
        committed = pv in ("SUPPORTED", "CONTRADICTED")
        recs.append({
            "claim_id": row["claim_id"],
            "family": row["family"],
            "hard_invalid": row.get("hard_invalid"),
            "primitive": row["primitive"],
            "gold_status": gold_s,
            "gold_reason": row.get("gold_reason"),
            "oracle_status_recheck": st["status"],
            "prod_status": pstatus,
            "gold_verdict": gold_v,
            "pred_verdict": pv,
            "false_commitment": gold_s != VALID and committed,
            "unknown_correct": gold_s != VALID and pv == "UNVERIFIABLE",
            "false_abstention": gold_s == VALID and pv == "UNVERIFIABLE",
            "verdict_ok": gold_s == VALID and pv == gold_v,
        })
    return _summarize(label, recs)


def _ci(k, n):
    return proportion(k, n) if n else {"p": None, "n": 0, "k": 0}


def _summarize(label, recs):
    inv = [r for r in recs if r["gold_status"] != VALID]
    val = [r for r in recs if r["gold_status"] == VALID]
    hard = [r for r in recs if r.get("hard_invalid") and r["gold_status"] != VALID]
    by_fam = _group(recs, "family")
    by_pr = _group(recs, "primitive")
    hard_by = defaultdict(list)
    for r in recs:
        if r["family"] in ("missing_required_channel", "invalid_fs", "missing_fs", "insufficient_n"):
            hard_by[r["family"]].append(r)
    return {
        "label": label,
        "n": len(recs),
        "n_invalid": len(inv),
        "n_valid": len(val),
        "status_accuracy": _ci(sum(r["prod_status"] == r["gold_status"] or (r["gold_status"] == VALID and r["prod_status"] == "VALID") for r in recs), len(recs)),
        "invalidated": {
            "n": len(inv),
            "false_commitment": _ci(sum(r["false_commitment"] for r in inv), len(inv)),
            "unknown_recall": _ci(sum(r["unknown_correct"] for r in inv), len(inv)),
        },
        "valid": {
            "n": len(val),
            "false_abstention": _ci(sum(r["false_abstention"] for r in val), len(val)),
            "verdict_accuracy": _ci(sum(r["verdict_ok"] for r in val), len(val)),
            "reference_agreement": _ci(sum(r["pred_verdict"] == r["gold_verdict"] for r in val), len(val)),
        },
        "hard_invalid": {
            "n": len(hard),
            "false_commitment": _ci(sum(r["false_commitment"] for r in hard), len(hard)),
        },
        "hard_categories": {k: {"n": len(xs), "fcr": _ci(sum(r["false_commitment"] for r in xs), len(xs))} for k, xs in hard_by.items()},
        "by_family": by_fam,
        "by_primitive": by_pr,
        "status_confusion": {f"{a}->{b}": n for (a, b), n in Counter((r["gold_status"], r["prod_status"]) for r in recs).items()},
        "records": recs,
    }


def _group(recs, key):
    g = defaultdict(list)
    for r in recs:
        g[r[key]].append(r)
    out = {}
    for k, xs in g.items():
        inv = [r for r in xs if r["gold_status"] != VALID]
        val = [r for r in xs if r["gold_status"] == VALID]
        out[k] = {
            "n": len(xs),
            "n_invalid": len(inv),
            "n_valid": len(val),
            "fcr": _ci(sum(r["false_commitment"] for r in inv), len(inv)) if inv else {"p": None, "n": 0, "k": 0},
            "unknown_recall": _ci(sum(r["unknown_correct"] for r in inv), len(inv)) if inv else None,
            "false_abstention": _ci(sum(r["false_abstention"] for r in val), len(val)) if val else None,
            "valid_verdict": _ci(sum(r["verdict_ok"] for r in val), len(val)) if val else None,
        }
    return out


def slim(m: dict) -> dict:
    d = {k: v for k, v in m.items() if k != "records"}
    d["n_records"] = len(m.get("records") or [])
    return d


def load_rows(name: str) -> list:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))
