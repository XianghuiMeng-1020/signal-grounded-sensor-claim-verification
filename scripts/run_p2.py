"""P2 pipeline entry point. Evaluate DEV/CHALLENGE only. Never open the sealed holdout."""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from p2.config import RESULTS_P2, ROOT as CFG_ROOT  # noqa: E402
from p2.build_benchmarks import build_independent_gold_v2, relabel_pilot_v1  # noqa: E402
from p2.evaluate import run_all_evaluations  # noqa: E402
from p2.validate_primitives import run_validation  # noqa: E402
from p2.write_reports import write_all  # noqa: E402


def snapshot_env() -> dict:
    import numpy
    import pandas
    import scipy

    info = {
        "python": sys.version,
        "executable": sys.executable,
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "pandas": pandas.__version__,
        "root": str(CFG_ROOT),
        "llm_openai": False,
        "llm_openrouter": False,
        "extraction_model_new_runs": None,
        "temperature": None,
        "seed": 20270823,
    }
    RESULTS_P2.mkdir(parents=True, exist_ok=True)
    (RESULTS_P2 / "environment.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def main() -> None:
    RESULTS_P2.mkdir(parents=True, exist_ok=True)
    env = snapshot_env()
    print("P2 env", env["python"].split()[0], "numpy", env["numpy"], "scipy", env["scipy"], flush=True)

    print("=== Phase 2 primitive validation ===", flush=True)
    prim = run_validation()
    print("  Gate C", prim["gate_c"], f"{prim['n_primitives_core_pass']}/8", flush=True)

    print("=== Phase 3 relabel pilot_v1 ===", flush=True)
    rel = relabel_pilot_v1()
    print("  flips", rel.get("n_flips"), "/", rel.get("n"), flush=True)

    print("=== Phase 3–4 build independent_gold_v2 + splits ===", flush=True)
    built = build_independent_gold_v2()
    print("  claims", built["n_claims"], "splits", {k: v.get("n") for k, v in built["splits"].items()}, flush=True)

    print("=== Phase 5–11 evaluate DEV+CHALLENGE (holdout sealed) ===", flush=True)
    ev = run_all_evaluations()
    print("  evaluated", ev.get("benchmark_n"), flush=True)

    print("=== Write reports ===", flush=True)
    meta = {
        "root": str(CFG_ROOT),
        "branch": "research/project-f-p2-scientific-escalation",
        "starting_commit": "UNVERSIONED_AT_AUDIT_START",
        "final_commit": "PENDING_GIT",
        "tag": "PENDING",
        "working_tree": "dirty_during_run",
    }
    write_all(meta)
    print("P2 pipeline complete. Reports in reports/P2_SCIENTIFIC_ESCALATION/", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
