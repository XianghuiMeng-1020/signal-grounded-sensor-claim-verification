"""P3R-EC runner. No LLM. Does not touch the frozen P3 1040 set."""
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
from p3r_ec.config import RESULTS
from p3r_ec.construct import construct_both
from p3r_ec.evaluate import evaluate_split, load_rows, slim
from p3r_ec.guard import refuse_legacy_p3_stress


def phase_construct():
    refuse_legacy_p3_stress()
    return construct_both()


def phase_eval(split: str):
    refuse_legacy_p3_stress()
    name = "ec_dev_rows.json" if split == "dev" else "ec_blind_rows.json"
    rows = load_rows(name)
    m = evaluate_split(rows, "EC-DEV" if split == "dev" else "EC-BLIND")
    out = "ec_dev_results.json" if split == "dev" else "ec_blind_results.json"
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / out).write_text(json.dumps(slim(m), indent=2), encoding="utf-8")
    rec_name = "ec_dev_records.json" if split == "dev" else "ec_blind_records.json"
    (RESULTS / rec_name).write_text(json.dumps(m["records"]), encoding="utf-8")
    print(split.upper(), json.dumps({k: m[k] for k in m if k != "records"}, default=str)[:2000], flush=True)
    return m


def phase_regression():
    dsp = run_validation()
    t = subprocess.run(
        [sys.executable, "-m", "pytest", "-q",
         "tests/p2r/test_contracts.py", "tests/p2r/test_kleene.py", "tests/p2r/test_boundaries.py",
         "tests/p3r_ec"],
        cwd=str(ROOT), capture_output=True, text=True,
    )
    payload = {
        "dsp_gate": dsp.get("gate_c"),
        "dsp_n_pass": dsp.get("n_primitives_core_pass"),
        "pytest_returncode": t.returncode,
        "pytest_tail": "\n".join((t.stdout or "").splitlines()[-12:]),
        "evidence_contracts": "PASS" if t.returncode == 0 else "FAIL",
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
    ap.add_argument("--phase", required=True, choices=["construct", "eval-dev", "eval-blind", "regression"])
    args = ap.parse_args()
    if args.phase == "construct":
        phase_construct()
    elif args.phase == "eval-dev":
        phase_eval("dev")
    elif args.phase == "eval-blind":
        phase_eval("blind")
    else:
        phase_regression()


if __name__ == "__main__":
    main()
