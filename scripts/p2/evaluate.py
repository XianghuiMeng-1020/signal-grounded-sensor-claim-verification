"""Evaluate DEVELOPMENT and CHALLENGE only. Refuse the sealed holdout."""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .config import (
    ABSTENTION_CANDIDATES,
    BENCH_P2,
    EVALUABLE_SPLITS,
    HOLDOUT_NAME,
    MARGIN_BANDS,
    PILOT_R6_SCORE,
    PILOT_R7_RAW,
    PILOT_V1_RAW,
    RESULTS_P2,
    ROOT,
    UNVERIFIABLE_TARGET_FCR,
)
from .extractor_deterministic import extract
from .independent_adjudicator import adjudicate as ref_adjudicate
from .independent_dsp import MeasurementError, measure as ref_measure
from .stats_ci import fmt_ci, mcnemar_paired, proportion, wilson_interval

sys.path.insert(0, str(ROOT / "scripts"))
from f_round8_composer import adjudicate as prod_adjudicate  # noqa: E402


def _refuse_holdout(split: str) -> None:
    if split == HOLDOUT_NAME or "holdout" in split.lower() and split != "challenge":
        if split == HOLDOUT_NAME:
            raise RuntimeError("P2 FORBIDDEN: attempted to evaluate the final sealed holdout.")


def load_split(split: str, bench_root=None) -> list[dict]:
    _refuse_holdout(split)
    if split not in EVALUABLE_SPLITS:
        raise RuntimeError(f"split {split} is not evaluable in P2")
    root = bench_root if bench_root is not None else BENCH_P2
    gold_path = root / "splits" / f"{split}.gold.jsonl"
    inf_path = root / "splits" / f"{split}.inference.jsonl"
    gold = [json.loads(line) for line in gold_path.read_text(encoding="utf-8").splitlines() if line]
    inf = [json.loads(line) for line in inf_path.read_text(encoding="utf-8").splitlines() if line]
    by_id = {r["claim_id"]: r for r in gold}
    rows = []
    for item in inf:
        g = by_id[item["claim_id"]]
        # inference fields only for extractor
        rows.append({**g, "inference": item})
    return rows


def _rep(row: dict) -> dict:
    return {"channels": row["channels_data"], "fs": row["fs"]}


def _programs_equal(a: Optional[dict], b: Optional[dict]) -> bool:
    if not a or not b:
        return False
    if (a.get("connective") or "SINGLE") != (b.get("connective") or "SINGLE"):
        return False
    pa, pb = a.get("predicates") or [], b.get("predicates") or []
    if len(pa) != len(pb):
        return False
    for x, y in zip(pa, pb):
        if x.get("op") != y.get("op"):
            return False
        if x.get("mode") != y.get("mode"):
            return False
        if (x.get("channels") or []) != (y.get("channels") or []):
            return False
        if x.get("compare_channel") != y.get("compare_channel"):
            return False
        if x.get("relation") != y.get("relation"):
            return False
        if x.get("mode") == "vs_value":
            av, bv = x.get("asserted_value"), y.get("asserted_value")
            if av is None or bv is None:
                return av == bv
            if abs(float(av) - float(bv)) > 1e-3 * max(1.0, abs(float(bv))):
                return False
        if x.get("mode") == "vs_threshold":
            av, bv = x.get("threshold"), y.get("threshold")
            if av is None or bv is None:
                return False
            if abs(float(av) - float(bv)) > 1e-3 * max(1.0, abs(float(bv))):
                return False
    return True


def _extract_row(row: dict) -> dict:
    inf = row["inference"]
    return extract(inf["surface_text"], inf["available_channels"], inf.get("fs"))


def run_condition(rows: list[dict], name: str) -> list[dict]:
    out = []
    for row in rows:
        gold_v = row["gold_composed_verdict"]
        gold_st = row.get("semantic_program") or {"connective": "SINGLE", "predicates": []}
        if name == "B4_oracle_production":
            extracted = gold_st
            pred = prod_adjudicate(_rep(row), extracted)
            verdict = pred["verdict"]
        elif name == "B5_forced_binary":
            extracted = gold_st
            pred = prod_adjudicate(_rep(row), extracted)
            verdict = pred["verdict"]
            if verdict == "UNVERIFIABLE":
                verdict = "CONTRADICTED"
        elif name == "B6_regex_structured":
            extracted = _extract_row(row)
            if extracted.get("unverifiable") or not extracted.get("predicates"):
                verdict = "UNVERIFIABLE"
                pred = {"verdict": verdict, "predicate_truths": [], "evidence": []}
            else:
                pred = prod_adjudicate(_rep(row), extracted)
                verdict = pred["verdict"]
        else:
            raise KeyError(name)
        out.append({
            "claim_id": row["claim_id"],
            "split": row["split"],
            "dataset": row["source_dataset"],
            "connective": row.get("connective"),
            "generation_family": row.get("generation_family"),
            "unverifiable_family": row.get("unverifiable_family"),
            "margin_band": row.get("margin_band"),
            "gold": gold_v,
            "pred": verdict,
            "correct": verdict == gold_v,
            "extracted": extracted,
            "gold_program": gold_st,
            "exact_program": _programs_equal(extracted, gold_st),
            "primitive_ok": _primitive_ok(extracted, gold_st),
            "channel_ok": _channel_ok(extracted, gold_st),
            "connective_ok": (extracted or {}).get("connective") == (gold_st or {}).get("connective"),
            "comparator_ok": _comparator_ok(extracted, gold_st),
        })
    return out


def _primitive_ok(ex, gold) -> bool:
    if not ex or not gold:
        return False
    a = [p.get("op") for p in (ex.get("predicates") or [])]
    b = [p.get("op") for p in (gold.get("predicates") or [])]
    return a == b


def _channel_ok(ex, gold) -> bool:
    if not ex or not gold:
        return False
    def chs(st):
        out = []
        for p in st.get("predicates") or []:
            out.extend(p.get("channels") or [])
            if p.get("compare_channel"):
                out.append(p["compare_channel"])
        return out
    return chs(ex) == chs(gold)


def _comparator_ok(ex, gold) -> bool:
    if not ex or not gold:
        return False
    pa, pb = ex.get("predicates") or [], gold.get("predicates") or []
    if len(pa) != len(pb):
        return False
    for x, y in zip(pa, pb):
        if x.get("mode") != y.get("mode"):
            return False
        if x.get("relation") != y.get("relation"):
            return False
        if x.get("mode") == "vs_value":
            av, bv = x.get("asserted_value"), y.get("asserted_value")
            if av is None or bv is None:
                return False
            if abs(float(av) - float(bv)) > 1e-3 * max(1.0, abs(float(bv))):
                return False
        if x.get("mode") == "vs_threshold":
            av, bv = x.get("threshold"), y.get("threshold")
            if av is None or bv is None:
                return False
            if abs(float(av) - float(bv)) > 1e-3 * max(1.0, abs(float(bv))):
                return False
    return True


def _macro_f1(rows: list[dict]) -> dict:
    labels = ("SUPPORTED", "CONTRADICTED", "UNVERIFIABLE")
    f1s = []
    per = {}
    for lab in labels:
        tp = sum(1 for r in rows if r["pred"] == lab and r["gold"] == lab)
        fp = sum(1 for r in rows if r["pred"] == lab and r["gold"] != lab)
        fn = sum(1 for r in rows if r["pred"] != lab and r["gold"] == lab)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per[lab] = {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
        f1s.append(f1)
    return {"macro_f1": float(np.mean(f1s)), "per_class": per}


def summarize(rows: list[dict], tag: str) -> dict:
    n = len(rows)
    acc = proportion(sum(r["correct"] for r in rows), n) if n else wilson_interval(0, 0)
    mf = _macro_f1(rows)
    unv = [r for r in rows if r["gold"] == "UNVERIFIABLE"]
    ans = [r for r in rows if r["gold"] != "UNVERIFIABLE"]
    committed_on_unv = [r for r in unv if r["pred"] in ("SUPPORTED", "CONTRADICTED")]
    fcr = proportion(len(committed_on_unv), len(unv)) if unv else wilson_interval(0, 0)
    coverage = proportion(sum(1 for r in ans if r["pred"] != "UNVERIFIABLE"), len(ans)) if ans else wilson_interval(0, 0)
    ans_acc = proportion(sum(r["correct"] for r in ans), len(ans)) if ans else wilson_interval(0, 0)
    unv_prec = mf["per_class"]["UNVERIFIABLE"]["precision"]
    unv_rec = mf["per_class"]["UNVERIFIABLE"]["recall"]
    # FPR: SUPPORTED predicted among gold CONTRADICTED? Standard: false SUPPORTED among gold not-SUPPORTED answerable
    gold_con = [r for r in rows if r["gold"] == "CONTRADICTED"]
    gold_sup = [r for r in rows if r["gold"] == "SUPPORTED"]
    supported_fpr = proportion(sum(1 for r in gold_con if r["pred"] == "SUPPORTED"), len(gold_con)) if gold_con else wilson_interval(0, 0)
    contradicted_fpr = proportion(sum(1 for r in gold_sup if r["pred"] == "CONTRADICTED"), len(gold_sup)) if gold_sup else wilson_interval(0, 0)
    recall_con = proportion(sum(1 for r in gold_con if r["pred"] == "CONTRADICTED"), len(gold_con)) if gold_con else wilson_interval(0, 0)
    by_conn, by_ds, by_prim = {}, {}, {}
    for key, getter in (
        ("connective", lambda r: r.get("connective") or "NA"),
        ("dataset", lambda r: r.get("dataset")),
        ("family", lambda r: str(r.get("generation_family"))),
    ):
        groups = defaultdict(list)
        for r in rows:
            groups[getter(r)].append(r)
        bucket = {k: {"n": len(v), "acc": proportion(sum(x["correct"] for x in v), len(v))} for k, v in groups.items()}
        if key == "connective":
            by_conn = bucket
        elif key == "dataset":
            by_ds = bucket
        else:
            by_prim = bucket
    return {
        "tag": tag,
        "n": n,
        "accuracy": acc,
        "accuracy_fmt": fmt_ci(acc),
        "macro_f1": mf["macro_f1"],
        "per_class": mf["per_class"],
        "exact_program_acc": proportion(sum(r["exact_program"] for r in rows), n) if n else wilson_interval(0, 0),
        "primitive_acc": proportion(sum(r["primitive_ok"] for r in rows), n) if n else wilson_interval(0, 0),
        "channel_acc": proportion(sum(r["channel_ok"] for r in rows), n) if n else wilson_interval(0, 0),
        "connective_acc": proportion(sum(r["connective_ok"] for r in rows), n) if n else wilson_interval(0, 0),
        "comparator_acc": proportion(sum(r["comparator_ok"] for r in rows), n) if n else wilson_interval(0, 0),
        "unverifiable_precision": unv_prec,
        "unverifiable_recall": unv_rec,
        "false_commitment": fcr,
        "false_commitment_fmt": fmt_ci(fcr),
        "answerable_coverage": coverage,
        "answerable_accuracy": ans_acc,
        "supported_fpr_on_contradicted": supported_fpr,
        "contradicted_fpr_on_supported": contradicted_fpr,
        "recall_contradicted": recall_con,
        "by_connective": by_conn,
        "by_dataset": by_ds,
        "by_family": by_prim,
    }


def choose_abstention(dev_rows: list[dict]) -> float:
    """Pre-registered selection: among candidates, FCR<=5% then max answerable coverage on DEV."""
    best = None
    for m in ABSTENTION_CANDIDATES:
        fcr_n = fcr_d = cov_n = cov_d = 0
        for row in dev_rows:
            gold = row["gold_composed_verdict"]
            st = row.get("semantic_program") or {"connective": "SINGLE", "predicates": []}
            pred = prod_adjudicate(_rep(row), st)["verdict"]
            margin = row.get("margin")
            if pred != "UNVERIFIABLE" and margin is not None and margin <= m:
                pred = "UNVERIFIABLE"
            if gold == "UNVERIFIABLE":
                fcr_d += 1
                if pred in ("SUPPORTED", "CONTRADICTED"):
                    fcr_n += 1
            else:
                cov_d += 1
                if pred != "UNVERIFIABLE":
                    cov_n += 1
        fcr = fcr_n / fcr_d if fcr_d else 0.0
        cov = cov_n / cov_d if cov_d else 0.0
        rec = {"margin": m, "fcr": fcr, "coverage": cov}
        if best is None:
            best = rec
            continue
        legal = fcr <= UNVERIFIABLE_TARGET_FCR
        best_legal = best["fcr"] <= UNVERIFIABLE_TARGET_FCR
        if legal and not best_legal:
            best = rec
        elif legal == best_legal and cov > best["coverage"]:
            best = rec
        elif legal == best_legal and cov == best["coverage"] and m < best["margin"]:
            best = rec
    return float(best["margin"]) if best else 0.0


def classify_error(row: dict, pred_row: dict) -> str:
    if pred_row["correct"]:
        return "none"
    gold_st = pred_row["gold_program"] or {}
    ex = pred_row["extracted"] or {}
    if row.get("gold_composed_verdict") == "UNVERIFIABLE" and pred_row["pred"] in ("SUPPORTED", "CONTRADICTED"):
        return "evidence_availability_error"
    if not pred_row["primitive_ok"]:
        return "primitive_selection_error"
    if not pred_row["channel_ok"]:
        return "channel_resolution_error"
    if not pred_row["comparator_ok"] or not pred_row["connective_ok"] or not pred_row["exact_program"]:
        if not pred_row["connective_ok"]:
            return "logical_composition_error"
        return "extraction_error"
    # extraction matched; check measurement disagreement
    try:
        prod_v = prod_adjudicate(_rep(row), gold_st)["verdict"]
        ref_v = ref_adjudicate(_rep(row), gold_st)["verdict"]
    except Exception:
        return "measurement_error"
    if prod_v != ref_v:
        return "measurement_error"
    if row.get("gold_composed_verdict") != ref_v:
        return "gold_label_ambiguity"
    if pred_row["pred"] != gold_st and pred_row["exact_program"]:
        return "logical_composition_error"
    return "other"


def error_table(rows: list[dict], pred_rows: list[dict]) -> list[dict]:
    by_id = {r["claim_id"]: r for r in rows}
    out = []
    for p in pred_rows:
        if p["correct"]:
            continue
        row = by_id[p["claim_id"]]
        cls = classify_error(row, p)
        # production vs reference measurements for first predicate if possible
        prod_m = ref_m = None
        st = p["gold_program"] or {}
        if st.get("predicates"):
            pred0 = st["predicates"][0]
            try:
                ref_m = ref_measure(pred0["op"], {c: row["channels_data"][c] for c in pred0.get("channels") or [] if c in row["channels_data"]}, row["fs"])
            except Exception:
                ref_m = None
            try:
                sys.path.insert(0, str(ROOT / "scripts"))
                from f_round6_operators import compute as prod_compute
                chmap = {c: row["channels_data"][c] for c in pred0.get("channels") or [] if c in row["channels_data"]}
                if chmap:
                    prod_m = prod_compute(pred0["op"], chmap, row["fs"])
            except Exception:
                prod_m = None
        out.append({
            "claim_id": p["claim_id"],
            "claim": row["surface_text"],
            "intended_semantic_program": st,
            "extracted_semantic_program": p["extracted"],
            "reference_measurement": ref_m,
            "production_measurement": prod_m,
            "gold_verdict": p["gold"],
            "predicted_verdict": p["pred"],
            "failure_classification": cls,
            "dataset": p["dataset"],
            "split": p["split"],
        })
    return out


# -------------------- signal robustness --------------------
PERTURBATIONS = (
    ("additive_noise_20db", "not_theoretically_invariant"),
    ("amplitude_scale_2x", "expected_equivariant"),
    ("temporal_shift_8", "expected_invariant_approx"),
    ("downsample_2", "expected_equivariant"),
    ("quantize_8bit", "expected_invariant_approx"),
    ("channel_dropout", "not_theoretically_invariant"),
    ("missing_samples", "not_theoretically_invariant"),
)


def _snr_noise(x, snr_db, rng):
    arr = np.asarray(x, dtype=float)
    p_sig = np.mean(arr ** 2) + 1e-12
    p_n = p_sig / (10 ** (snr_db / 10))
    return (arr + rng.normal(0, np.sqrt(p_n), arr.shape)).tolist()


def apply_perturbation(row: dict, name: str, rng) -> dict:
    chs = {k: list(v) for k, v in row["channels_data"].items()}
    fs = float(row["fs"])
    if name == "additive_noise_20db":
        chs = {k: _snr_noise(v, 20, rng) for k, v in chs.items()}
    elif name == "amplitude_scale_2x":
        chs = {k: (np.asarray(v) * 2.0).tolist() for k, v in chs.items()}
    elif name == "temporal_shift_8":
        chs = {k: np.roll(np.asarray(v), 8).tolist() for k, v in chs.items()}
    elif name == "downsample_2":
        chs = {k: np.asarray(v)[::2].tolist() for k, v in chs.items()}
        fs = fs / 2.0
    elif name == "quantize_8bit":
        def q(v):
            a = np.asarray(v, dtype=float)
            lo, hi = a.min(), a.max()
            if hi <= lo:
                return a.tolist()
            u = np.round((a - lo) / (hi - lo) * 255.0)
            return (u / 255.0 * (hi - lo) + lo).tolist()
        chs = {k: q(v) for k, v in chs.items()}
    elif name == "channel_dropout":
        first = next(iter(chs))
        chs[first] = []
    elif name == "missing_samples":
        first = next(iter(chs))
        arr = np.asarray(chs[first], dtype=float)
        if arr.size:
            arr = arr.copy()
            arr[arr.size // 2] = np.nan
            chs[first] = arr.tolist()
    return {"channels": chs, "fs": fs}


def expected_behavior(op: str, pert: str) -> str:
    if pert in ("channel_dropout", "missing_samples"):
        return "should_abstain_or_fail"
    if pert == "amplitude_scale_2x":
        if op in ("rms_amplitude", "peak_amplitude", "signal_range"):
            return "expected_equivariant"
        if op in ("dominant_frequency", "periodicity_strength", "spectral_energy_ratio_low", "trend_ratio", "cross_channel_lag_ms"):
            return "expected_invariant"
        return "not_theoretically_invariant"
    if pert == "downsample_2":
        if op == "dominant_frequency":
            return "expected_invariant"  # if fs is updated
        return "not_theoretically_invariant"
    if pert in ("temporal_shift_8", "quantize_8bit"):
        return "expected_invariant_approx"
    return "not_theoretically_invariant"


def run_signal_robustness(rows: list[dict], n_per=40) -> dict:
    rng = np.random.default_rng(20270823)
    # use oracle schema on a subset of answerable SINGLE claims
    pool = [r for r in rows if r.get("generation_family") == "SINGLE_VS_VALUE" and r["gold_composed_verdict"] != "UNVERIFIABLE"]
    if len(pool) > n_per:
        idx = rng.choice(len(pool), size=n_per, replace=False)
        pool = [pool[i] for i in idx]
    results = []
    for row in pool:
        st = row["semantic_program"]
        op = (st.get("predicates") or [{}])[0].get("op")
        base = prod_adjudicate(_rep(row), st)
        for pert, _ in PERTURBATIONS:
            new_rep = apply_perturbation(row, pert, rng)
            try:
                new = prod_adjudicate(new_rep, st)
                pred = new["verdict"]
                err = None
            except Exception as exc:  # noqa: BLE001
                pred, err = "UNVERIFIABLE", str(exc)
                new = {"verdict": pred, "evidence": []}
            exp = expected_behavior(op, pert)
            stable = pred == base["verdict"]
            results.append({
                "claim_id": row["claim_id"],
                "op": op,
                "perturbation": pert,
                "expected_class": exp,
                "base_verdict": base["verdict"],
                "new_verdict": pred,
                "stable": stable,
                "error": err,
                "dataset": row["source_dataset"],
            })
    # summaries
    by = defaultdict(list)
    for r in results:
        by[(r["perturbation"], r["expected_class"])].append(r)
    summary = {}
    for (pert, cls), recs in by.items():
        if cls.startswith("expected_invariant"):
            summary[f"{pert}|{cls}"] = {
                "n": len(recs),
                "stability": proportion(sum(x["stable"] for x in recs), len(recs)),
            }
        elif cls == "should_abstain_or_fail":
            n_abs = sum(1 for x in recs if x["new_verdict"] == "UNVERIFIABLE")
            summary[f"{pert}|{cls}"] = {"n": len(recs), "abstain_rate": proportion(n_abs, len(recs))}
        else:
            summary[f"{pert}|{cls}"] = {"n": len(recs), "changed_rate": proportion(sum(not x["stable"] for x in recs), len(recs))}
    return {"n_rows": len(results), "summary": summary, "rows": results}


def historical_baselines() -> dict:
    out = {
        "llm_available": False,
        "B1_direct_llm_judge": "NOT_RUN_on_v2_no_api",
        "B2_tool_agent": "NOT_RUN_on_v2_no_api",
        "B3_llm_over_precomputed": "NOT_RUN_on_v2_no_api",
        "cached_r6": None,
        "cached_r7_note": None,
        "pilot_v1_llm_extraction_rescored": None,
    }
    if PILOT_R6_SCORE.exists():
        out["cached_r6"] = json.loads(PILOT_R6_SCORE.read_text(encoding="utf-8"))
    if PILOT_V1_RAW.exists() and (BENCH_P2 / "pilot_v1_independent_relabel.json").exists():
        raw = json.loads(PILOT_V1_RAW.read_text(encoding="utf-8"))
        rel = json.loads((BENCH_P2 / "pilot_v1_independent_relabel.json").read_text(encoding="utf-8"))
        gold = {r["id"]: r for r in rel["rows"]}
        n = corr_prod = corr_ind = 0
        for row in raw:
            g = gold.get(row["id"])
            if not g:
                continue
            n += 1
            corr_prod += int(row.get("pred_verdict") == g["production_gold"])
            corr_ind += int(row.get("pred_verdict") == g["independent_gold"])
        out["pilot_v1_llm_extraction_rescored"] = {
            "n": n,
            "vs_production_gold": proportion(corr_prod, n) if n else None,
            "vs_independent_gold": proportion(corr_ind, n) if n else None,
        }
    if PILOT_R7_RAW.exists():
        out["cached_r7_note"] = (
            "Round-7 stronger tool-agent headline 71.1% FPR / 76.5% recall is taken from "
            "the certification memo; raw JSON is preserved but was not re-derived as a "
            "confirmatory independent-gold number in P2."
        )
    return out


def run_all_evaluations() -> dict:
    RESULTS_P2.mkdir(parents=True, exist_ok=True)
    dev = load_split("development")
    chal = load_split("challenge")
    # freeze abstention on DEV only
    abst = choose_abstention(dev)
    (RESULTS_P2 / "frozen_abstention_margin.json").write_text(
        json.dumps({"chosen_on": "development", "margin": abst, "candidates": ABSTENTION_CANDIDATES}, indent=2),
        encoding="utf-8",
    )

    conditions = ("B4_oracle_production", "B5_forced_binary", "B6_regex_structured")
    all_pred = {}
    summaries = {}
    for split, rows in (("development", dev), ("challenge", chal)):
        for cond in conditions:
            preds = run_condition(rows, cond)
            all_pred[f"{split}:{cond}"] = preds
            summaries[f"{split}:{cond}"] = summarize(preds, f"{split}:{cond}")

    # language robustness = challenge B6, answerable+all, plus extraction breakdown
    lang = summaries["challenge:B6_regex_structured"]
    # UNVERIFIABLE first-class on challenge B6
    unv_focus = summarize(
        [p for p in all_pred["challenge:B6_regex_structured"]],
        "challenge_unv",
    )
    # margin on challenge oracle (isolates measurement, not extraction)
    margin_rows = [p for p in all_pred["challenge:B4_oracle_production"]]
    by_band = defaultdict(list)
    gold_by = {r["claim_id"]: r for r in chal}
    for p in margin_rows:
        band = gold_by[p["claim_id"]].get("margin_band") or "unknown"
        by_band[band].append(p)
    margin_summary = {b: summarize(v, b) for b, v in by_band.items()}

    # apply frozen abstention to challenge oracle (not retuned)
    chal_abs = []
    for row, p in zip(chal, all_pred["challenge:B4_oracle_production"]):
        pred = p["pred"]
        m = row.get("margin")
        if pred != "UNVERIFIABLE" and m is not None and m <= abst:
            pred = "UNVERIFIABLE"
        chal_abs.append({**p, "pred": pred, "correct": pred == p["gold"]})
    abst_summary = summarize(chal_abs, "challenge:B4_abstention_frozen")

    # signal robustness on challenge
    sig = run_signal_robustness(chal)

    # cross-dataset from challenge B6 and B4
    cross = {
        "B6": summaries["challenge:B6_regex_structured"]["by_dataset"],
        "B4": summaries["challenge:B4_oracle_production"]["by_dataset"],
    }

    # errors: all challenge B6 mistakes
    errors = error_table(chal, all_pred["challenge:B6_regex_structured"])
    err_counts = Counter(e["failure_classification"] for e in errors)

    # McNemar B4 vs B6 and B4 vs B5 on challenge
    b4 = all_pred["challenge:B4_oracle_production"]
    b5 = all_pred["challenge:B5_forced_binary"]
    b6 = all_pred["challenge:B6_regex_structured"]
    tests = {
        "B4_vs_B6_challenge": mcnemar_paired([r["correct"] for r in b4], [r["correct"] for r in b6]),
        "B4_vs_B5_challenge": mcnemar_paired([r["correct"] for r in b4], [r["correct"] for r in b5]),
        "label": "exploratory_paired_tests",
    }

    hist = historical_baselines()

    payload = {
        "summaries": summaries,
        "language_robustness_challenge_B6": lang,
        "unverifiable_challenge_B6": unv_focus,
        "margin_challenge_B4": margin_summary,
        "abstention_frozen": {"margin": abst, "challenge_B4": abst_summary},
        "signal_robustness": {"n_rows": sig["n_rows"], "summary": sig["summary"]},
        "cross_dataset": cross,
        "error_counts": dict(err_counts),
        "n_errors_challenge_B6": len(errors),
        "paired_tests": tests,
        "historical": hist,
        "benchmark_n": {"development": len(dev), "challenge": len(chal)},
    }
    (RESULTS_P2 / "evaluation_summaries.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    (RESULTS_P2 / "error_decomposition.json").write_text(json.dumps(errors, indent=1, default=str), encoding="utf-8")
    (RESULTS_P2 / "signal_robustness.json").write_text(json.dumps(sig, indent=1, default=str), encoding="utf-8")
    return payload
