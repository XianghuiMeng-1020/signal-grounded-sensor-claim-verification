"""P3C runner. Frozen system. No holdout. No old-set tuning."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from p2.validate_primitives import run_validation
from p3c.audit_frozen_ls import run_audit
from p3c.config import RESULTS
from p3c.eval_p3c import evaluate_rows, slim
from p3c.harth_construct import construct as harth_construct
from p3c.ls_construct import construct as ls_construct
from p3c.margin_construct import construct as margin_construct


def _load(name):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def phase_audit():
    return run_audit()


def phase_ls_build():
    return ls_construct()


def phase_ls_eval():
    rows = _load("ls_closure_rows.json")
    m = evaluate_rows(rows, "p3c:LS-CLOSURE", "ls_closure.ckpt.json")
    (RESULTS / "ls_closure_primary.json").write_text(json.dumps(slim(m), indent=2), encoding="utf-8")
    (RESULTS / "ls_closure_records.json").write_text(json.dumps(m["records"]), encoding="utf-8")
    (RESULTS / "ls_closure_run.json").write_text(json.dumps({"LS_CLOSURE_PRIMARY_RUN_COUNT": 1}), encoding="utf-8")
    print("LS_EVAL", {k: m[k] for k in ("n", "exact_program", "strict_semantic", "verdict_accuracy", "false_commitment")}, flush=True)
    return m


def phase_harth_build():
    return harth_construct()


def phase_harth_eval():
    rows = _load("harth_closure_rows.json")
    m = evaluate_rows(rows, "p3c:HARTH-CLOSURE", "harth_closure.ckpt.json")
    (RESULTS / "harth_closure_primary.json").write_text(json.dumps(slim(m), indent=2), encoding="utf-8")
    (RESULTS / "harth_closure_records.json").write_text(json.dumps(m["records"]), encoding="utf-8")
    (RESULTS / "harth_closure_run.json").write_text(json.dumps({"HARTH_CLOSURE_PRIMARY_RUN_COUNT": 1}), encoding="utf-8")
    print("HARTH_EVAL", {k: m[k] for k in ("n", "exact_program", "strict_semantic", "verdict_accuracy", "false_commitment", "macro_f1")}, flush=True)
    return m


def phase_margin_build():
    return margin_construct()


def phase_margin_eval():
    rows = _load("margin_closure_rows.json")
    m = evaluate_rows(rows, "p3c:MARGIN-CLOSURE", "margin_closure.ckpt.json")
    (RESULTS / "margin_closure_primary.json").write_text(json.dumps(slim(m), indent=2), encoding="utf-8")
    (RESULTS / "margin_closure_records.json").write_text(json.dumps(m["records"]), encoding="utf-8")
    print("MARGIN_EVAL", {k: m[k] for k in ("n", "verdict_accuracy", "false_commitment", "false_abstention")}, flush=True)
    return m


def phase_regression():
    dsp = run_validation()
    t = subprocess.run(
        [sys.executable, "-m", "pytest", "-q",
         "tests/p2r/test_contracts.py", "tests/p2r/test_kleene.py", "tests/p3r_ec", "tests/p3"],
        cwd=str(ROOT), capture_output=True, text=True,
    )
    payload = {
        "dsp_n_pass": dsp.get("n_primitives_core_pass"),
        "dsp_gate": dsp.get("gate_c"),
        "pytest_returncode": t.returncode,
        "pytest_tail": "\n".join((t.stdout or "").splitlines()[-10:]),
        "evidence_contracts": "PASS" if t.returncode == 0 else "FAIL",
        "property_tests": "PASS" if t.returncode == 0 else "FAIL",
        "kleene": "PASS" if t.returncode == 0 else "FAIL",
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "execution_regression.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("REGRESSION", payload, flush=True)
    if t.returncode != 0:
        print(t.stdout, t.stderr, flush=True)
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True,
                    choices=["audit", "ls-build", "ls-eval", "harth-build", "harth-eval",
                             "margin-build", "margin-eval", "regression"])
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    {
        "audit": phase_audit,
        "ls-build": phase_ls_build,
        "ls-eval": phase_ls_eval,
        "harth-build": phase_harth_build,
        "harth-eval": phase_harth_eval,
        "margin-build": phase_margin_build,
        "margin-eval": phase_margin_eval,
        "regression": phase_regression,
    }[args.phase]()


if __name__ == "__main__":
    main()
