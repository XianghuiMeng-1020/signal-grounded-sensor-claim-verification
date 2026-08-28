"""Score frozen programs on clean and degraded waveforms via production oracle."""
from __future__ import annotations

from collections import Counter, defaultdict

from p2r.pipeline import run_oracle

from .config import PERTURBATIONS
from .degrade import apply

VERDICTS = ("SUPPORTED", "CONTRADICTED", "UNVERIFIABLE")


def _verdict(program, available, fs, channels) -> str:
    rec = run_oracle(program, available, fs, channels)
    return rec["verdict"]


def score_item(item: dict) -> dict:
    prog = item["program"]
    avail = item["available_channels"]
    fs = item["fs"]
    clean = item["channels"]
    named = item["named_channels"]
    clean_v = _verdict(prog, avail, fs, clean)
    out = {
        "item_id": item["item_id"],
        "dataset": item["dataset"],
        "op": item["op"],
        "clean_verdict": clean_v,
        "threshold": item["threshold"],
        "clean_value": item["clean_value"],
        "perturbed": {},
    }
    for name in PERTURBATIONS:
        dirty = apply(name, clean, named, item["item_id"])
        out["perturbed"][name] = {
            "verdict": _verdict(prog, avail, fs, dirty),
            "evidence_invalid_by_construction": name == "dropout_10pct",
        }
    return out


def _empty_matrix() -> dict[str, dict[str, int]]:
    return {a: {b: 0 for b in VERDICTS} for a in VERDICTS}


def summarize(records: list[dict]) -> dict:
    per = {}
    for name in PERTURBATIONS:
        mat = _empty_matrix()
        dirty_counts = Counter()
        n = 0
        n_clean_sup = 0
        n_sup_kept = 0
        n_invalid = 0
        n_commit_on_invalid = 0
        n_unknown = 0
        for rec in records:
            clean_v = rec["clean_verdict"]
            cell = rec["perturbed"][name]
            dirty_v = cell["verdict"]
            mat[clean_v][dirty_v] += 1
            dirty_counts[dirty_v] += 1
            n += 1
            if dirty_v == "UNVERIFIABLE":
                n_unknown += 1
            if clean_v == "SUPPORTED":
                n_clean_sup += 1
                if dirty_v == "SUPPORTED":
                    n_sup_kept += 1
            if cell["evidence_invalid_by_construction"]:
                n_invalid += 1
                if dirty_v in ("SUPPORTED", "CONTRADICTED"):
                    n_commit_on_invalid += 1
        per[name] = {
            "n": n,
            "transition": mat,
            "unknown_rate": n_unknown / n if n else None,
            "supported_preservation": n_sup_kept / n_clean_sup if n_clean_sup else None,
            "n_clean_supported": n_clean_sup,
            "false_commitment_rate": (
                n_commit_on_invalid / n_invalid if n_invalid else None
            ),
            "n_invalid_by_construction": n_invalid,
            "dirty_verdict_counts": dict(dirty_counts),
        }
    clean_counts = Counter(r["clean_verdict"] for r in records)
    by_op = defaultdict(lambda: Counter())
    for r in records:
        by_op[r["op"]][r["clean_verdict"]] += 1
    return {
        "n_items": len(records),
        "clean_verdict_counts": dict(clean_counts),
        "clean_by_op": {k: dict(v) for k, v in by_op.items()},
        "by_perturbation": per,
    }
