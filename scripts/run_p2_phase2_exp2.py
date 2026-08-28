"""Phase 2 Experiment 2 entry point. Lag time-base only. Run once."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from p2_phase2.lag_config import EXPERIMENT_ID, RESULTS, SEED  # noqa: E402
from p2_phase2.lag_construct import build_items  # noqa: E402
from p2_phase2.lag_evaluate import score_item, summarize  # noqa: E402
from p2_phase2.lag_write_report import write  # noqa: E402


def _slim(rec: dict) -> dict:
    keep = (
        "item_id",
        "window_id",
        "dataset",
        "fs_native",
        "fs",
        "n",
        "delay_ms",
        "delay_samples",
        "true_lag_ms",
        "in_sample_box",
        "length_ok",
        "sample_bound_ms",
        "physical_bound_ms",
        "mode_a",
        "mode_b",
        "contract_status",
        "contract_reason",
        "measured_lag_ms",
        "measurement_faithful",
        "invalid_length",
        "invalid_outside_search",
        "invalid_by_construction",
    )
    return {k: rec[k] for k in keep}


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
    out = RESULTS / "lag_timebase_run.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report = write(summary, meta)
    print(
        json.dumps(
            {
                "wrote": str(out),
                "report": report,
                "n_items": summary["n_items"],
                "n_measurable": summary["n_measurable"],
                "consistency_valid_length": summary["consistency_valid_length"],
                "false_commitment": summary["false_commitment"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
