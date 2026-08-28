"""Shared P3 evaluation. Holdouts refused. PRIMARY prompt v2 frozen."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from p2.stats_ci import mcnemar_paired, proportion
from p2r.eval_p2r import _gold_program
from p2r.extractor import extract_b6_baseline
from p2r.lag_canon import canonicalize_program
from p2r.ollama_adapter import extract_ollama
from p2r.pipeline import run_oracle, run_pipeline
from p2r.schema import ClaimProgram
from p2r.validator import from_legacy
import run_p2r_lm_local as loc

from .config import PRIMARY_MODEL, PRIMARY_PROMPT, RESULTS
from .guard import refuse_holdout
from .io_util import program_from_dict
from .semantic_canon import programs_canonically_equal


def _gold(row) -> ClaimProgram:
    return from_legacy(row.get("semantic_program") or {"connective": "SINGLE", "predicates": []}, row["available_channels"])


def field_raw(pred, gold):
    return loc.field_scores(canonicalize_program(pred), canonicalize_program(gold))


def evaluate_extractor(rows, extractor, label: str, ckpt_name: str | None = None):
    refuse_holdout(str(rows[0].get("split") if rows else ""))
    ckpt = RESULTS / ckpt_name if ckpt_name else None
    done = {}
    if ckpt and ckpt.exists():
        done = {r["claim_id"]: r for r in json.loads(ckpt.read_text(encoding="utf-8"))}
        print(f"  {label} resume {len(done)} cached records", flush=True)
    recs = []
    field_hits = Counter()
    n = 0
    for i, row in enumerate(rows, 1):
        if i == 1 or i % 50 == 0 or i == len(rows):
            print(f"  {label} {i}/{len(rows)}", flush=True)
        if row["claim_id"] in done:
            rec = done[row["claim_id"]]
        else:
            gold = _gold(row)
            pred = extractor(row["surface_text"], row["available_channels"], row.get("fs"))
            ch = {k: __import__("numpy").asarray(v, dtype=float) for k, v in row["channels_data"].items() if v is not None}
            out = run_pipeline(row["surface_text"], row["available_channels"], row["fs"], ch, lambda t, c, f, _p=pred: _p)
            fs = field_raw(pred, gold)
            gv = row["gold_composed_verdict"]
            pv = out["verdict"]
            rec = {
                "claim_id": row["claim_id"],
                "dataset": row.get("dataset") or row.get("source_dataset"),
                "source": row.get("source"),
                "band": row.get("band"),
                "primitive": (gold.predicates[0].measurement if gold.predicates else None),
                "gold_verdict": gv,
                "pred_verdict": pv,
                "correct": pv == gv,
                "exact": bool(fs["exact"]),
                "canonical": programs_canonically_equal(pred, gold),
                "pred_program": pred.to_dict(),
                "error": loc.classify_error(canonicalize_program(pred), canonicalize_program(gold)),
            }
        recs.append(rec)
        n += 1
        gold = _gold(row)
        pred = program_from_dict(rec["pred_program"]) if rec.get("pred_program") else gold
        fs = field_raw(pred, gold) if rec.get("pred_program") and rec["pred_program"].get("parse_status") else {}
        for k, v in fs.items():
            field_hits[k] += int(v)
        if ckpt and (i % 25 == 0 or i == len(rows)):
            RESULTS.mkdir(parents=True, exist_ok=True)
            ckpt.write_text(json.dumps(recs, ensure_ascii=False), encoding="utf-8")
    return _summarize(label, recs, field_hits, n)


def evaluate_primary(rows, label: str, ckpt_name: str | None = None):
    def ext(text, chs, fs):
        prog, _ = extract_ollama(text, chs, fs, model=PRIMARY_MODEL, prompt_version=PRIMARY_PROMPT)
        return prog
    return evaluate_extractor(rows, ext, label, ckpt_name=ckpt_name)


def evaluate_secondary(rows, model: str, label: str):
    def ext(text, chs, fs):
        prog, _ = extract_ollama(text, chs, fs, model=model, prompt_version=PRIMARY_PROMPT)
        return ext if False else prog  # keep type
    def ext2(text, chs, fs):
        prog, _ = extract_ollama(text, chs, fs, model=model, prompt_version=PRIMARY_PROMPT)
        return prog
    return evaluate_extractor(rows, ext2, label)


def evaluate_b6(rows, label: str):
    def ext(text, chs, fs):
        return extract_b6_baseline(text, chs, fs)
    return evaluate_extractor(rows, ext, label)


def evaluate_oracle(rows, label: str):
    recs = []
    field_hits = Counter()
    n = 0
    for row in rows:
        gold = _gold(row)
        ch = {k: __import__("numpy").asarray(v, dtype=float) for k, v in row["channels_data"].items() if v is not None}
        out = run_oracle(gold, row["available_channels"], row["fs"], ch)
        n += 1
        field_hits["exact"] += 1
        recs.append({
            "claim_id": row["claim_id"],
            "dataset": row.get("dataset"),
            "gold_verdict": row["gold_composed_verdict"],
            "pred_verdict": out["verdict"],
            "correct": out["verdict"] == row["gold_composed_verdict"],
            "exact": True,
            "canonical": True,
            "pred_program": gold.to_dict(),
            "error": "none",
        })
    return _summarize(label, recs, field_hits, n)


def evaluate_forced_binary(rows, base_recs, label: str):
    """Map UNVERIFIABLE predictions to CONTRADICTED (forced commitment)."""
    recs = []
    for r in base_recs:
        pv = r["pred_verdict"]
        if pv == "UNVERIFIABLE":
            pv = "CONTRADICTED"
        recs.append({**r, "pred_verdict": pv, "correct": pv == r["gold_verdict"]})
    field_hits = Counter(exact=sum(int(r["exact"]) for r in recs))
    return _summarize(label, recs, field_hits, len(recs))


def _summarize(label, recs, field_hits, n):
    labels = ("SUPPORTED", "CONTRADICTED", "UNVERIFIABLE")
    f1s = []
    per = {}
    for lab in labels:
        tp = sum(1 for r in recs if r["pred_verdict"] == lab and r["gold_verdict"] == lab)
        fp = sum(1 for r in recs if r["pred_verdict"] == lab and r["gold_verdict"] != lab)
        fn = sum(1 for r in recs if r["pred_verdict"] != lab and r["gold_verdict"] == lab)
        prec = tp / (tp + fp) if tp + fp else 0.0
        reca = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * reca / (prec + reca) if prec + reca else 0.0
        per[lab] = {"precision": prec, "recall": reca, "f1": f1}
        f1s.append(f1)
    unv = [r for r in recs if r["gold_verdict"] == "UNVERIFIABLE"]
    ans = [r for r in recs if r["gold_verdict"] != "UNVERIFIABLE"]
    committed = sum(1 for r in unv if r["pred_verdict"] in ("SUPPORTED", "CONTRADICTED"))
    false_abs = sum(1 for r in ans if r["pred_verdict"] == "UNVERIFIABLE")
    fields = {k: field_hits[k] / n for k in field_hits} if n else {}
    return {
        "label": label,
        "n": n,
        "exact_program": fields.get("exact", sum(r["exact"] for r in recs) / n if n else None),
        "canonical_semantic": sum(r.get("canonical", False) for r in recs) / n if n else None,
        "primitive": fields.get("measurement"),
        "channel": fields.get("channel"),
        "comparator": fields.get("comparator"),
        "value": fields.get("value"),
        "connective": fields.get("connective"),
        "verdict_accuracy": proportion(sum(r["correct"] for r in recs), n) if n else None,
        "macro_f1": float(np.mean(f1s)) if f1s else None,
        "per_class": per,
        "false_commitment": proportion(committed, len(unv)) if unv else {"p": 0.0, "n": 0, "k": 0},
        "false_abstention": proportion(false_abs, len(ans)) if ans else None,
        "answerable_coverage": proportion(len(ans) - false_abs, len(ans)) if ans else None,
        "records": recs,
        "by_source": _by(recs, "source"),
        "by_dataset": _by(recs, "dataset"),
        "by_band": _by(recs, "band"),
    }


def _by(recs, key):
    from collections import defaultdict
    g = defaultdict(list)
    for r in recs:
        if r.get(key) is None:
            continue
        g[r[key]].append(r)
    out = {}
    for k, xs in g.items():
        out[k] = {
            "n": len(xs),
            "exact": sum(r["exact"] for r in xs) / len(xs),
            "canonical": sum(r.get("canonical", False) for r in xs) / len(xs),
            "verdict": sum(r["correct"] for r in xs) / len(xs),
        }
    return out


def slim(m: dict) -> dict:
    d = {k: v for k, v in m.items() if k != "records"}
    d["n_records"] = len(m.get("records") or [])
    return d


def paired_verdict(a_recs, b_recs):
    by = {r["claim_id"]: r for r in b_recs}
    paired = [(x["correct"], by[x["claim_id"]]["correct"]) for x in a_recs if x["claim_id"] in by]
    return {
        "n": len(paired),
        "a": sum(p[0] for p in paired) / len(paired) if paired else None,
        "b": sum(p[1] for p in paired) / len(paired) if paired else None,
        "mcnemar": mcnemar_paired([p[0] for p in paired], [p[1] for p in paired]) if paired else None,
    }
