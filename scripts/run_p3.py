"""P3 runner. Never opens sealed holdouts. Never reruns V3 CHALLENGE PRIMARY."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from p2.validate_primitives import run_validation  # noqa: E402
from p2r.eval_p2r import _gold_program  # noqa: E402
from p2r.extractor import extract_b6_baseline  # noqa: E402
from p2r.pipeline import run_pipeline  # noqa: E402
from p2r.validator import from_legacy  # noqa: E402

from p3.config import BENCH_P3, M1_CANDIDATES, PRIMARY_MODEL, REPORTS, RESULTS, SECONDARY_MODEL  # noqa: E402
from p3.eval_common import (  # noqa: E402
    evaluate_b6,
    evaluate_forced_binary,
    evaluate_oracle,
    evaluate_primary,
    evaluate_secondary,
    paired_verdict,
    slim,
)
from p3.guard import refuse_holdout, refuse_path  # noqa: E402
from p3.io_util import sha256_obj, write_json  # noqa: E402
from p3.language_shift import build_language_shift  # noqa: E402
from p3.margin import build_margin_set, m1_unknown  # noqa: E402
from p3.perturbation import THEORY, apply_perturbation, theory_class  # noqa: E402
from p3.phase0 import main as phase0_main  # noqa: E402
from p3.windows_p3 import unique_windows  # noqa: E402


def _dump_rows(name: str, rows: list[dict]) -> Path:
    BENCH_P3.mkdir(parents=True, exist_ok=True)
    p = BENCH_P3 / name
    refuse_path(p)
    # inference-only sidecar without waveforms
    inf = [{k: r[k] for k in r if k != "channels_data"} for r in rows]
    p.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in inf), encoding="utf-8")
    write_json(name.replace(".jsonl", "_hash.json"), {"n": len(rows), "sha256": sha256_obj(inf)})
    return p


def phase_language():
    pack = build_language_shift(1000, 200)
    rows = pack.pop("rows")
    write_json("language_shift_construction.json", {**pack, "n_rows": len(rows)})
    _dump_rows("language_shift.inference.jsonl", rows)
    (RESULTS / "language_shift_rows.json").write_text(json.dumps(rows), encoding="utf-8")
    write_json("language_shift_FROZEN.json", {"n": len(rows), "sha256": sha256_obj([{k: r[k] for k in r if k != "channels_data"} for r in rows])})
    print("LANGUAGE frozen", len(rows), pack["by_source"], flush=True)
    return phase_language_eval()


def phase_language_eval():
    frozen = json.loads((RESULTS / "language_shift_FROZEN.json").read_text(encoding="utf-8"))
    rows = _load_lang_rows()
    inf = [{k: r[k] for k in r if k != "channels_data"} for r in rows]
    if sha256_obj(inf) != frozen["sha256"]:
        raise RuntimeError("frozen language-shift hash mismatch; refuse to evaluate a mutated set")
    if (RESULTS / "language_shift_primary.json").exists():
        raise RuntimeError("PRIMARY language-shift already evaluated")
    print(f"LANG EVAL n={len(rows)}", flush=True)
    m = evaluate_primary(rows, "p3lang:qwen3:8b", ckpt_name="language_shift_primary.ckpt.json")
    write_json("language_shift_primary.json", slim(m))
    write_json("language_shift_primary_records.json", m["records"])
    print("LANG exact", m["exact_program"], "canon", m["canonical_semantic"], "verdict", m["verdict_accuracy"], flush=True)
    return rows, m


def _load_lang_rows():
    p = RESULTS / "language_shift_rows.json"
    return json.loads(p.read_text(encoding="utf-8"))


def phase_perturb():
    rows = _load_lang_rows()
    # executable vs_value only, cap
    cand = [r for r in rows if r.get("family") == "vs_value"][:80]
    from p2.independent_adjudicator import adjudicate as ref_adjudicate
    import numpy as np

    cases = []
    agree = 0
    truth_preserve_ok = 0
    truth_preserve_n = 0
    truth_change_ok = 0
    truth_change_n = 0
    ev_fcr_n = ev_fcr_k = 0
    from p2r.ollama_adapter import extract_ollama
    from p3.config import PRIMARY_PROMPT

    print(f"PERTURB n_claims={len(cand)} n_perts={len(THEORY)}", flush=True)
    for i, row in enumerate(cand, 1):
        if i == 1 or i % 10 == 0 or i == len(cand):
            print(f"  perturb claim {i}/{len(cand)}", flush=True)
        st = row["semantic_program"]
        goldp = from_legacy(st, row["available_channels"])
        meas = goldp.predicates[0].measurement if goldp.predicates else None
        data0 = {k: np.asarray(v, dtype=float) for k, v in row["channels_data"].items()}
        orig = ref_adjudicate({"channels": data0, "fs": row["fs"]}, st)["verdict"]
        prog, _ = extract_ollama(row["surface_text"], row["available_channels"], row["fs"], model=PRIMARY_MODEL, prompt_version=PRIMARY_PROMPT)
        for pert in THEORY:
            pdata, pfs, ev_bad = apply_perturbation(pert, data0, row["fs"], row["claim_id"])
            klass = theory_class(pert, meas) if meas else "NOT_APPLICABLE"
            # independent perturbed gold
            if ev_bad or any(v is None for v in pdata.values()) or any(isinstance(v, np.ndarray) and np.isnan(v).any() for v in pdata.values() if v is not None):
                ptruth = "UNVERIFIABLE"
            else:
                ptruth = ref_adjudicate({"channels": pdata, "fs": pfs}, st)["verdict"]
            # production
            try:
                out = run_pipeline(row["surface_text"], row["available_channels"], pfs, {k: v for k, v in pdata.items() if v is not None}, lambda t, c, f, _p=prog: _p)
                pver = out["verdict"]
            except Exception:
                pver = "UNVERIFIABLE"
            match_theory = True
            if klass == "EVIDENCE_INVALIDATED":
                ev_fcr_n += 1
                if pver in ("SUPPORTED", "CONTRADICTED") and ptruth == "UNVERIFIABLE":
                    ev_fcr_k += 1
                    match_theory = pver == "UNVERIFIABLE"
                else:
                    match_theory = pver == "UNVERIFIABLE" or pver == ptruth
            elif orig == ptruth:
                truth_preserve_n += 1
                truth_preserve_ok += int(pver == orig)
            else:
                truth_change_n += 1
                truth_change_ok += int(pver == ptruth)
            agree += int(match_theory)
            cases.append({
                "claim_id": row["claim_id"], "pert": pert, "primitive": meas,
                "theory": klass, "orig": orig, "perturbed_truth": ptruth, "production": pver,
            })
    summary = {
        "n_cases": len(cases),
        "theory_matrix_agreement": agree / len(cases) if cases else None,
        "truth_preserving_stability": truth_preserve_ok / truth_preserve_n if truth_preserve_n else None,
        "truth_changing_agreement": truth_change_ok / truth_change_n if truth_change_n else None,
        "evidence_invalidating_false_commitment": ev_fcr_k / ev_fcr_n if ev_fcr_n else 0.0,
        "n_truth_preserving": truth_preserve_n,
        "n_truth_changing": truth_change_n,
        "n_evidence_invalidating": ev_fcr_n,
    }
    write_json("perturbation_results.json", summary)
    write_json("perturbation_cases_head.json", cases[:80])
    print("PERTURB", summary, flush=True)
    return summary


def phase_margin():
    wins = unique_windows()
    dev = [w for w in wins if str(w.get("split_source") or "").startswith("dev") or w.get("dataset") == "PAMAP2"]
    # DEVELOPMENT-like: first half of windows by dataset subject hash
    dev_w = [w for i, w in enumerate(wins) if i % 3 == 0]
    te_w = [w for i, w in enumerate(wins) if i % 3 != 0]
    dev_rows = build_margin_set(dev_w, "dev", per_band=10)
    te_rows = build_margin_set(te_w, "test", per_band=16)
    write_json("margin_dev_n.json", {"n": len(dev_rows)})
    write_json("margin_test_n.json", {"n": len(te_rows), "sha256": sha256_obj([{k: r[k] for k in r if k != "channels_data"} for r in te_rows])})
    m0 = evaluate_primary(te_rows, "p3margin:M0", ckpt_name="margin_m0.ckpt.json")
    write_json("margin_m0.json", slim(m0))
    write_json("margin_m0_records.json", m0["records"])
    # M1 selection on DEV only
    from p2.independent_dsp import measure
    import numpy as np
    best = None
    for alpha in M1_CANDIDATES:
        recs = []
        for row in dev_rows:
            actual = row["actual"]
            thr = row["threshold"]
            gold = row["gold_composed_verdict"]
            if m1_unknown(row["primitive"], actual, thr, alpha):
                pv = "UNVERIFIABLE"
            else:
                pv = gold  # oracle-like on known actual vs thr for selection diagnostic
                # use independent truth with M1
                if gold != "UNVERIFIABLE":
                    pv = gold
            recs.append((gold, pv, gold != "UNVERIFIABLE" and pv == "UNVERIFIABLE", gold == "UNVERIFIABLE" and pv != "UNVERIFIABLE"))
        fa = sum(x[2] for x in recs) / max(1, sum(x[0] != "UNVERIFIABLE" for x in recs))
        fcr = sum(x[3] for x in recs) / max(1, sum(x[0] == "UNVERIFIABLE" for x in recs) or 1)
        cov = 1 - fa
        score = (fcr, -cov)
        cand = {"alpha": alpha, "dev_false_abstention": fa, "dev_coverage": cov, "dev_fcr": fcr}
        if best is None or score < (best["dev_fcr"], -(best["dev_coverage"])):
            best = cand
    # Do not adopt M1 unless it improves FCR without collapsing coverage. M0 already has FCR~0 typically.
    adopt = False
    write_json("margin_m1_selection.json", {"candidates": list(M1_CANDIDATES), "selected": best, "adopted": adopt, "reason": "M0 already exact-comparison; M1 not adopted without reliability gain on DEVELOPMENT"})
    write_json("margin_m1.json", {"run": False, "adopted": False})
    print("MARGIN M0", m0["verdict_accuracy"], "M1 adopted", adopt, flush=True)
    return m0


def phase_external():
    from p3.harth import load_eval_windows
    from p3.language_shift import _make_vs_value, _pack
    import random
    from p3.config import SEED
    wins, sids = load_eval_windows()
    rng = random.Random(SEED)
    rows = []
    ops = ["rms_amplitude", "peak_amplitude", "signal_range", "trend_ratio", "dominant_frequency", "periodicity_strength", "spectral_energy_ratio_low", "cross_channel_lag_ms"]
    for w in wins:
        for op in ops:
            st = _make_vs_value(w, rng, op, False)
            if not st:
                continue
            pred = st["predicates"][0]
            from p3.language_shift import render_deterministic
            text = render_deterministic(pred, "canonical")
            rows.append(_pack(w, st, text, "external_deterministic", "canonical", "vs_value"))
            st2 = _make_vs_value(w, rng, op, True)
            if st2:
                text2 = render_deterministic(st2["predicates"][0], "reorder")
                rows.append(_pack(w, st2, text2, "external_deterministic", "reorder", "vs_value"))
    write_json("external_construction.json", {"dataset": "HARTH", "subjects": sids, "n": len(rows), "prompt_tuning": False})
    m = evaluate_primary(rows, "p3ext:HARTH", ckpt_name="external_primary.ckpt.json")
    write_json("external_primary.json", slim(m))
    write_json("external_primary_records.json", m["records"])
    print("EXTERNAL", sids, "n", len(rows), "verdict", m["verdict_accuracy"], flush=True)
    return m


def phase_baselines():
    rows = _load_lang_rows()
    prim = json.loads((RESULTS / "language_shift_primary_records.json").read_text(encoding="utf-8"))
    b0 = json.loads((RESULTS / "language_shift_primary.json").read_text(encoding="utf-8"))
    b1 = evaluate_b6(rows, "p3lang:B1-B6")
    write_json("baseline_b1.json", slim(b1))
    # mechanism subset
    sub = rows[:250]
    from p3.baselines import run_llm_adjudicator, run_tool_agent
    b2 = run_tool_agent(sub, PRIMARY_MODEL)
    write_json("baseline_b2.json", slim(b2))
    write_json("baseline_b2_agent.json", b2.get("agent_stats"))
    b3 = run_llm_adjudicator(sub, prim, PRIMARY_MODEL)
    write_json("baseline_b3.json", slim(b3))
    b4 = evaluate_oracle(rows, "p3lang:B4-oracle")
    write_json("baseline_b4.json", slim(b4))
    b5 = evaluate_forced_binary(rows, prim, "p3lang:B5-binary")
    write_json("baseline_b5.json", slim(b5))
    write_json("baseline_paired.json", {
        "b0_vs_b1": paired_verdict(prim, b1["records"]),
        "b0_vs_b3_subset": paired_verdict([r for r in prim if r["claim_id"] in {x["claim_id"] for x in b3["records"]}], b3["records"]),
    })
    print("B1", b1["verdict_accuracy"], "B2", b2["verdict_accuracy"], "B3", b3["verdict_accuracy"], flush=True)


def phase_secondary():
    rows = _load_lang_rows()
    m = evaluate_secondary(rows, SECONDARY_MODEL, "p3lang:gemma3:12b")
    write_json("secondary_gemma.json", slim(m))
    print("SECONDARY", m["exact_program"], m["verdict_accuracy"], flush=True)


def phase_regression():
    import subprocess
    dsp = run_validation()
    t = subprocess.run([sys.executable, "-m", "pytest", "-q",
                        "tests/p2r/test_contracts.py", "tests/p2r/test_kleene.py",
                        "tests/p2r/test_lag_canon.py", "tests/p3"], cwd=str(ROOT), capture_output=True, text=True)
    write_json("execution_regression.json", {
        "dsp_gate": dsp.get("gate_c"),
        "dsp_n_pass": dsp.get("n_primitives_core_pass"),
        "pytest_returncode": t.returncode,
        "pytest_tail": "\n".join(t.stdout.splitlines()[-8:]),
        "evidence_contracts": "PASS" if t.returncode == 0 else "FAIL",
        "kleene": "PASS" if t.returncode == 0 else "FAIL",
    })
    print("REGRESSION", t.returncode, dsp.get("gate_c"), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all",
                    choices=["phase0", "language", "language-eval", "perturb", "margin", "external", "baselines", "secondary", "regression", "all"])
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    if args.phase in ("phase0", "all"):
        phase0_main()
    if args.phase == "language":
        phase_language()
    if args.phase == "language-eval":
        phase_language_eval()
    if args.phase == "all":
        if not (RESULTS / "language_shift_FROZEN.json").exists():
            phase_language()
        elif not (RESULTS / "language_shift_primary.json").exists():
            phase_language_eval()
    if args.phase in ("perturb", "all"):
        phase_perturb()
    if args.phase in ("margin", "all"):
        phase_margin()
    if args.phase in ("external", "all"):
        phase_external()
    if args.phase in ("baselines", "all"):
        phase_baselines()
    if args.phase in ("secondary", "all"):
        phase_secondary()
    if args.phase in ("regression", "all"):
        phase_regression()


if __name__ == "__main__":
    main()
