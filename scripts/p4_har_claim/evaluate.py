"""Score HAR class-match verdicts vs contract+DSP+Kleene on the same claims."""
from __future__ import annotations

from collections import Counter

from p2r.pipeline import run_oracle

from .dictionary import make_program
from .features import feature_names

VERDICTS = ("SUPPORTED", "CONTRADICTED", "UNVERIFIABLE")


def har_verdict(pred_family: str | None, posed_family: str) -> str:
    """Recognition-as-verification: class match is SUPPORTED, else CONTRADICTED.

    There is no UNKNOWN. Missing or unmapped predictions still commit.
    """
    if pred_family is not None and pred_family == posed_family:
        return "SUPPORTED"
    return "CONTRADICTED"


def proposed_verdict(family: str, channel_a: str, available: list[str], fs: float, channels: dict) -> str:
    prog = make_program(family, channel_a)
    rec = run_oracle(prog, available, fs, channels)
    return rec["verdict"]


def fcr(rows: list[dict], pred_key: str) -> float | None:
    unv = [r for r in rows if r["gold"] == "UNVERIFIABLE"]
    if not unv:
        return None
    committed = sum(1 for r in unv if r[pred_key] in ("SUPPORTED", "CONTRADICTED"))
    return committed / len(unv)


def _slice_metrics(rows: list[dict], pred_key: str) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0, "verdict_accuracy": None, "unknown_rate": None, "supported_recall": None, "contradicted_recall": None}
    correct = sum(1 for r in rows if r[pred_key] == r["gold"])
    unknown_rate = sum(1 for r in rows if r[pred_key] == "UNVERIFIABLE") / n
    sup = [r for r in rows if r["gold"] == "SUPPORTED"]
    con = [r for r in rows if r["gold"] == "CONTRADICTED"]
    return {
        "n": n,
        "verdict_accuracy": correct / n,
        "unknown_rate": unknown_rate,
        "supported_recall": (sum(1 for r in sup if r[pred_key] == "SUPPORTED") / len(sup)) if sup else None,
        "contradicted_recall": (sum(1 for r in con if r[pred_key] == "CONTRADICTED") / len(con)) if con else None,
        "gold_counts": dict(Counter(r["gold"] for r in rows)),
        "pred_counts": dict(Counter(r[pred_key] for r in rows)),
    }


def metrics(rows: list[dict]) -> dict:
    legal = [r for r in rows if r.get("slice") == "legal"]
    illegal = [r for r in rows if r.get("slice") == "illegal"]
    out = {}
    for name in ("har", "proposed"):
        out[name] = {
            "fcr": fcr(rows, name),
            "unknown_rate": (sum(1 for r in rows if r[name] == "UNVERIFIABLE") / len(rows)) if rows else None,
            "legal": _slice_metrics(legal, name),
            "illegal": _slice_metrics(illegal, name),
            "all": _slice_metrics(rows, name),
        }
    return out


def score_rows(items: list[dict], predicted_family: dict[str, str | None]) -> list[dict]:
    """Attach HAR and proposed verdicts. Gold is always the proposed stack."""
    rows = []
    for it in items:
        gold = proposed_verdict(
            it["posed_family"],
            it["channel_a"],
            it["available_channels"],
            it["fs"],
            it["channels"],
        )
        pred_fam = predicted_family.get(it["item_id"])
        har = har_verdict(pred_fam, it["posed_family"])
        rows.append({
            "item_id": it["item_id"],
            "window_id": it["window_id"],
            "dataset": it["dataset"],
            "subject": it["subject"],
            "slice": it["slice"],
            "invalidation": it.get("invalidation"),
            "gold_activity": it["gold_activity"],
            "posed_family": it["posed_family"],
            "pred_family": pred_fam,
            "gold": gold,
            "har": har,
            "proposed": gold,
            "feature_names": feature_names(it.get("train_channels") or list(it["channels"])),
        })
    return rows


def recognition_accuracy(windows: list[dict], predicted_activity: dict[str, object]) -> float | None:
    if not windows:
        return None
    hit = sum(1 for w in windows if str(predicted_activity.get(w["window_id"])) == str(w["activity"]))
    return hit / len(windows)


# keep extract_features import used by runner via this module
__all__ = [
    "har_verdict",
    "proposed_verdict",
    "fcr",
    "metrics",
    "score_rows",
    "recognition_accuracy",
    "extract_features",
]
