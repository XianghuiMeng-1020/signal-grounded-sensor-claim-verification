"""Phase 3 Evidence Trust entry. Order: Module 3 -> 1 -> 2. Stop on gate failure."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from p3_evidence_trust.m1_evaluate import score as m1_score  # noqa: E402
from p3_evidence_trust.m1_write import write as m1_write  # noqa: E402
from p3_evidence_trust.m2_evaluate import run as m2_run  # noqa: E402
from p3_evidence_trust.m2_write import write as m2_write  # noqa: E402
from p3_evidence_trust.m3_numerical import run as m3_run  # noqa: E402
from p3_evidence_trust.m3_write import write as m3_write  # noqa: E402


def main() -> int:
    m3 = m3_run()
    print("M3", m3["summary"]["decision"], m3_write(m3))
    if m3["summary"]["decision"] != "PASS":
        print("STOP after Module 3")
        return 2
    m1 = m1_score()
    print("M1", m1["summary"]["decision"], m1_write(m1))
    if m1["summary"]["decision"] != "PASS":
        print("STOP after Module 1")
        return 3
    m2 = m2_run()
    print("M2", m2["summary"]["decision"], m2_write(m2))
    if m2["summary"]["decision"] != "PASS":
        print("STOP after Module 2")
        return 4
    print(json.dumps({"phase3": "PASS", "m3": "PASS", "m1": "PASS", "m2": "PASS"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
