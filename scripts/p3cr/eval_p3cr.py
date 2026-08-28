"""P3C-R evaluation. Prompt version is an argument. Checkpointed."""
from __future__ import annotations

import json
from collections import Counter, defaultdict

import numpy as np

from p2.stats_ci import proportion
from p2r.ollama_adapter import extract_ollama
from p2r.pipeline import run_pipeline
from p3.eval_common import _gold, field_raw

from p3c.strict_semantic import programs_strictly_equivalent

from .config import PRIMARY_MODEL, RESULTS


def evaluate_rows(rows, label: str, ckpt_name: str, prompt_version: str) -> dict:
    ckpt = RESULTS / ckpt_name
    done = {}
    if ckpt.exists():
        done = {r["claim_id"]: r for r in json.loads(ckpt.read_text(encoding="utf-8"))}
        print(f"  {label} resume {len(done)}", flush=True)
    recs = []
    field_hits = Counter()
    for i, row in enumerate(rows, 1):
        if i == 1 or i % 50 == 0 or i == len(rows):
            print(f"  {label} {i}/{len(rows)}", flush=True)
        if row["claim_id"] in done:
            rec = done[row["claim_id"]]
        else:
            gold = _gold(row)
            pred, _ = extract_ollama(
                row["surface_text"], row["available_channels"], row.get("fs"),
                model=PRIMARY_MODEL, prompt_version=prompt_version,
            )
            ch = {k: np.asarray(v, dtype=float) for k, v in row["channels_data"].items() if v is not None}
            out = run_pipeline(row["surface_text"], row["available_channels"], row["fs"], ch, lambda t, c, f, _p=pred: _p)
            fs = field_raw(pred, gold)
            rec = {
                "claim_id": row["claim_id"],
                "source": row.get("source"),
                "family": row.get("family"),
                "band": row.get("band"),
                "condition": row.get("condition"),
                "primitive": row.get("primitive") or (gold.predicates[0].measurement if gold.predicates else None),
                "connective": row.get("connective") or gold.connective,
                "n_pred": row.get("n_pred") if row.get("n_pred") is not None else len(gold.predicates),
                "gold_verdict": row["gold_composed_verdict"],
                "pred_verdict": out["verdict"],
                "exact": bool(fs.get("exact")),
                "strict": programs_strictly_equivalent(pred, gold, row["available_channels"]),
                "pred_program": pred.to_dict(),
                "field": {k: bool(fs.get(k)) for k in ("measurement", "channel", "comparator", "value", "connective")},
                "prompt_version": prompt_version,
            }
        recs.append(rec)
        for k, v in (rec.get("field") or {}).items():
            field_hits[k] += int(v)
        if i % 25 == 0 or i == len(rows):
            RESULTS.mkdir(parents=True, exist_ok=True)
            ckpt.write_text(json.dumps(recs, ensure_ascii=False), encoding="utf-8")
    return _summarize(label, recs, field_hits, prompt_version)


def _ci(k, n):
    return proportion(k, n) if n else {"p": None, "n": 0, "k": 0}


def _summarize(label, recs, field_hits, prompt_version):
    n = len(recs)
    labels = ("SUPPORTED", "CONTRADICTED", "UNVERIFIABLE")
    per = {}
    f1s = []
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
    fcr = sum(1 for r in unv if r["pred_verdict"] in ("SUPPORTED", "CONTRADICTED"))
    fa = sum(1 for r in ans if r["pred_verdict"] == "UNVERIFIABLE")

    def _by(key):
        g = defaultdict(list)
        for r in recs:
            if r.get(key) is None:
                continue
            g[r[key]].append(r)
        out = {}
        for k, xs in g.items():
            a = [x for x in xs if x["gold_verdict"] != "UNVERIFIABLE"]
            u = [x for x in xs if x["gold_verdict"] == "UNVERIFIABLE"]
            out[str(k)] = {
                "n": len(xs),
                "exact": sum(x["exact"] for x in xs) / len(xs),
                "strict": sum(bool(x.get("strict")) for x in xs) / len(xs),
                "verdict": sum(x["pred_verdict"] == x["gold_verdict"] for x in xs) / len(xs),
                "primitive": sum((x.get("field") or {}).get("measurement", False) for x in xs) / len(xs),
                "fcr": _ci(sum(x["pred_verdict"] in ("SUPPORTED", "CONTRADICTED") for x in u), len(u)) if u else {"p": None, "n": 0, "k": 0},
                "fa": _ci(sum(x["pred_verdict"] == "UNVERIFIABLE" for x in a), len(a)) if a else None,
            }
        return out

    return {
        "label": label,
        "prompt_version": prompt_version,
        "n": n,
        "exact_program": sum(r["exact"] for r in recs) / n if n else None,
        "strict_semantic": sum(bool(r.get("strict")) for r in recs) / n if n else None,
        "primitive": field_hits["measurement"] / n if n else None,
        "channel": field_hits["channel"] / n if n else None,
        "comparator": field_hits["comparator"] / n if n else None,
        "value": field_hits["value"] / n if n else None,
        "connective": field_hits["connective"] / n if n else None,
        "verdict_accuracy": _ci(sum(r["pred_verdict"] == r["gold_verdict"] for r in recs), n),
        "macro_f1": float(np.mean(f1s)) if f1s else None,
        "per_class": per,
        "false_commitment": _ci(fcr, len(unv)) if unv else {"p": 0.0, "n": 0, "k": 0},
        "false_abstention": _ci(fa, len(ans)) if ans else None,
        "by_source": _by("source"),
        "by_connective": _by("connective"),
        "by_n_pred": _by("n_pred"),
        "by_primitive": _by("primitive"),
        "by_band": _by("band"),
        "by_condition": _by("condition"),
        "by_family": _by("family"),
        "records": recs,
    }


def slim(m):
    d = {k: v for k, v in m.items() if k != "records"}
    d["n_records"] = len(m.get("records") or [])
    return d
