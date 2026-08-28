"""HAR → claim baseline. Run once. No prompt/kernel/threshold search."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from p4_har_claim.config import EXPERIMENT_ID, RESULTS, SEED  # noqa: E402
from p4_har_claim.construct import (  # noqa: E402
    assert_no_holdout,
    build_items,
    load_labeled_unused,
    pool_audit,
    split_windows,
)
from p4_har_claim.dictionary import DICTIONARY_ID, dictionary_sha256  # noqa: E402
from p4_har_claim.evaluate import metrics, recognition_accuracy, score_rows  # noqa: E402
from p4_har_claim.har_model import fit_rf, model_card, predict_activities, predict_families  # noqa: E402
from p4_har_claim.write_report import write  # noqa: E402


def _slim(row: dict) -> dict:
    return {k: row[k] for k in row if k != "feature_names"}


def main() -> int:
    windows = load_labeled_unused()
    assert_no_holdout(windows)
    train, test = split_windows(windows)
    if len(train) < 8 or len(test) < 4:
        raise RuntimeError(f"too few windows train={len(train)} test={len(test)}")
    clf, order = fit_rf(train)
    items = build_items(test)
    pred_fam = predict_families(clf, order, items)
    rows = score_rows(items, pred_fam)
    pred_act = predict_activities(clf, order, test)
    summary = metrics(rows)
    audit = {
        "all_unused_mappable": pool_audit(windows),
        "train_development": pool_audit(train),
        "test_challenge": pool_audit(test),
    }
    model = model_card(order)
    meta = {
        "experiment_id": EXPERIMENT_ID,
        "seed": SEED,
        "dictionary_id": DICTIONARY_ID,
        "dictionary_sha256": dictionary_sha256(),
        "n_train_windows": len(train),
        "n_test_windows": len(test),
        "n_items": len(rows),
        "recognition_accuracy_challenge": recognition_accuracy(test, pred_act),
        "run_count": 1,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "har_claim_run.json").write_text(
        json.dumps({"meta": meta, "metrics": summary, "records": [_slim(r) for r in rows]}, indent=2),
        encoding="utf-8",
    )
    write(meta, summary, audit, model)
    print(json.dumps({"meta": meta, "metrics": {k: {"fcr": v.get("fcr"), "unknown_rate": v.get("unknown_rate")} for k, v in summary.items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
