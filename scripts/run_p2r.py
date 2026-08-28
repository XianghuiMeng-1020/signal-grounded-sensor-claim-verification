"""P2R runner. Never loads the sealed holdout."""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from p2r.eval_p2r import run_all  # noqa: E402
from p2r.write_reports import write_all  # noqa: E402


def main() -> None:
    print("P2R start", flush=True)
    ev = run_all()
    print("eval written", ev.get("dsp_validation"), "llm", ev.get("llm", {}).get("available"), flush=True)
    gates = write_all({
        "root": str(ROOT),
        "branch": "research/project-f-p2r-semantic-evidence-repair",
        "start": "afd2aa559b42a8039e46b5db42f28cd23f5c6498",
        "final": "PENDING",
        "tag": "PENDING",
    })
    print("gates", gates, flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
