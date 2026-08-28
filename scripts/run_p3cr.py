"""P3C-R runner. SEM sets must exist before prompt-v3 is added."""
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
from p3cr.config import RESULTS
from p3cr.eval_p3cr import evaluate_rows, slim
from p3cr.margin2_construct import construct as margin2_construct
from p3cr.sem_construct import construct as sem_construct


def _load(name):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def _dump(name, obj):
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / name).write_text(json.dumps(obj, indent=2), encoding="utf-8")


def phase_sem_build():
    return sem_construct()


def phase_margin2_build():
    return margin2_construct()


def phase_sem_dev(prompt_version: str):
    rows = _load("sem_dev_rows.json")
    m = evaluate_rows(rows, f"p3cr:SEM-DEV:{prompt_version}", f"sem_dev_{prompt_version}.ckpt.json", prompt_version)
    _dump(f"sem_dev_{prompt_version}.json", slim(m))
    print("SEM_DEV", prompt_version, {k: m[k] for k in ("n", "exact_program", "strict_semantic", "primitive", "verdict_accuracy", "macro_f1", "false_commitment")}, flush=True)
    return m


def phase_sem_blind(prompt_version: str):
    rows = _load("sem_blind_rows.json")
    m = evaluate_rows(rows, f"p3cr:SEM-BLIND:{prompt_version}", "sem_blind.ckpt.json", prompt_version)
    _dump("sem_blind_primary.json", slim(m))
    _dump("sem_blind_run.json", {"SEM_BLIND_PRIMARY_RUN_COUNT": 1, "prompt_version": prompt_version})
    print("SEM_BLIND", {k: m[k] for k in ("n", "strict_semantic", "primitive", "verdict_accuracy", "macro_f1", "false_commitment")}, flush=True)
    return m


def phase_margin2_eval(prompt_version: str):
    rows = _load("margin2_blind_rows.json")
    m = evaluate_rows(rows, "p3cr:MARGIN2-BLIND", "margin2_blind.ckpt.json", prompt_version)
    _dump("margin2_blind_primary.json", slim(m))
    _dump("margin2_blind_run.json", {"MARGIN2_BLIND_RUN_COUNT": 1, "prompt_version": prompt_version})
    print("MARGIN2", {k: m[k] for k in ("n", "verdict_accuracy", "false_commitment")}, flush=True)
    return m


def phase_regression():
    dsp = run_validation()
    t = subprocess.run(
        [sys.executable, "-m", "pytest", "-q",
         "tests/p2r/test_contracts.py", "tests/p2r/test_kleene.py", "tests/p3r_ec", "tests/p3",
         "tests/p3cr"],
        cwd=str(ROOT), capture_output=True, text=True,
    )
    payload = {
        "dsp_n_pass": dsp.get("n_primitives_core_pass"),
        "dsp_gate": dsp.get("gate_c"),
        "pytest_returncode": t.returncode,
        "pytest_tail": "\n".join((t.stdout or "").splitlines()[-12:]),
        "evidence_contracts": "PASS" if t.returncode == 0 else "FAIL",
        "property_tests": "PASS" if t.returncode == 0 else "FAIL",
        "kleene": "PASS" if t.returncode == 0 else "FAIL",
    }
    _dump("execution_regression.json", payload)
    print("REGRESSION", payload, flush=True)
    if t.returncode != 0:
        print(t.stdout, t.stderr, flush=True)
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True,
                    choices=["sem-build", "sem-dev", "sem-blind", "margin2-build", "margin2-eval", "regression"])
    ap.add_argument("--prompt", default="v2")
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    {
        "sem-build": phase_sem_build,
        "sem-dev": lambda: phase_sem_dev(args.prompt),
        "sem-blind": lambda: phase_sem_blind(args.prompt),
        "margin2-build": phase_margin2_build,
        "margin2-eval": lambda: phase_margin2_eval(args.prompt),
        "regression": phase_regression,
    }[args.phase]()


if __name__ == "__main__":
    main()
