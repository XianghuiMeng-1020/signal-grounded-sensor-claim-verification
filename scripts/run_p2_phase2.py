"""Phase 2 Experiment 1 entry point. Waveform degradation only. Run once."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from p2_phase2.config import EXPERIMENT_ID, RESULTS, SEED  # noqa: E402
from p2_phase2.construct import build_items  # noqa: E402
from p2_phase2.evaluate import score_item, summarize  # noqa: E402
from p2_phase2.write_report import write  # noqa: E402


def _slim(rec: dict) -> dict:
    return {
        "item_id": rec["item_id"],
        "dataset": rec["dataset"],
        "op": rec["op"],
        "clean_verdict": rec["clean_verdict"],
        "threshold": rec["threshold"],
        "clean_value": rec["clean_value"],
        "perturbed": {
            k: {"verdict": v["verdict"]} for k, v in rec["perturbed"].items()
        },
    }


def main() -> int:
    items = build_items()
    records = [score_item(it) for it in items]
    summary = summarize(records)
    datasets = dict(Counter(it["dataset"] for it in items))
    n_windows = len({it["window_id"] for it in items})
    meta = {
        "experiment_id": EXPERIMENT_ID,
        "seed": SEED,
        "n_windows": n_windows,
        "n_items": len(items),
        "datasets": datasets,
        "run_count": 1,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    payload = {"meta": meta, "summary": summary, "records": [_slim(r) for r in records]}
    out = RESULTS / "degradation_run.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report = write(summary, meta)
    print(json.dumps({"wrote": str(out), "report": report, "summary": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
