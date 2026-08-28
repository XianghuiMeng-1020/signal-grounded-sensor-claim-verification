"""Score Mode A (sample-domain) vs Mode B (physical-time) on frozen kernels."""
from __future__ import annotations

from collections import Counter, defaultdict

from p2r.contracts import OK
from p2r.executor import execute_predicate_measurement
from p2r.pipeline import run_oracle
from p2r.schema import Predicate

from .lag_config import (
    DELAYS_MS,
    EVAL_PHYSICAL,
    EVAL_SAMPLE,
    FS_GRID,
    MIN_N_LAG,
    PHYSICAL_BOUND_MS,
    SAMPLE_BOUND,
    faithful_tol_ms,
)


def _measure(item: dict) -> dict:
    a, b = item["named_channels"]
    pred = Predicate(
        measurement="cross_channel_lag_ms",
        channel_a=a,
        comparator="lt",
        channel_b=b,
        reference_value=0.0,
        unit="ms",
    )
    res = execute_predicate_measurement(pred, item["channels"], item["fs"])
    measured = float(res.value) if res.status == OK and res.value is not None else None
    faithful = False
    if measured is not None:
        # Production correlate(a, roll(a,+k)) reports −k samples. Compare magnitudes.
        faithful = abs(abs(measured) - abs(float(item["true_lag_ms"]))) <= faithful_tol_ms(
            item["fs"]
        )
    short = item["n"] < MIN_N_LAG
    outside = not item["in_sample_box"]
    return {
        "contract_status": res.status,
        "contract_reason": (res.diagnostics or {}).get("reason"),
        "measured_lag_ms": measured,
        "measurement_faithful": faithful,
        "invalid_length": short,
        "invalid_outside_search": outside and not short,
        "invalid_by_construction": short or outside,
    }


def score_item(item: dict) -> dict:
    diag = _measure(item)
    avail = item["available_channels"]
    fs = item["fs"]
    ch = item["channels"]
    a = run_oracle(item["program_sample"], avail, fs, ch, eval_mode=EVAL_SAMPLE)
    b = run_oracle(item["program_physical"], avail, fs, ch, eval_mode=EVAL_PHYSICAL)
    return {
        "item_id": item["item_id"],
        "window_id": item["window_id"],
        "dataset": item["dataset"],
        "fs_native": item["fs_native"],
        "fs": fs,
        "n": item["n"],
        "delay_ms": item["delay_ms"],
        "delay_samples": item["delay_samples"],
        "true_lag_ms": item["true_lag_ms"],
        "in_sample_box": item["in_sample_box"],
        "length_ok": item["length_ok"],
        "sample_bound_ms": item["sample_bound_ms"],
        "physical_bound_ms": item["physical_bound_ms"],
        "mode_a": {"verdict": a["verdict"], "eval_mode": EVAL_SAMPLE},
        "mode_b": {"verdict": b["verdict"], "eval_mode": EVAL_PHYSICAL},
        **diag,
    }


def _rates(rows: list[dict], key: str) -> dict:
    c = Counter(r[key]["verdict"] if key in ("mode_a", "mode_b") else r[key] for r in rows)
    n = len(rows)
    n_s = c.get("SUPPORTED", 0)
    n_c = c.get("CONTRADICTED", 0)
    n_u = c.get("UNVERIFIABLE", 0)
    return {
        "n": n,
        "SUPPORTED": n_s,
        "CONTRADICTED": n_c,
        "UNVERIFIABLE": n_u,
        "support_rate": n_s / n if n else None,
        "contradict_rate": n_c / n if n else None,
        "unknown_rate": n_u / n if n else None,
    }


def _commit_on_invalid(rows: list[dict], mode_key: str) -> dict:
    inv = [r for r in rows if r["invalid_by_construction"]]
    n = len(inv)
    n_commit = sum(1 for r in inv if r[mode_key]["verdict"] in ("SUPPORTED", "CONTRADICTED"))
    return {
        "n_invalid": n,
        "n_commit": n_commit,
        "false_commitment_rate": n_commit / n if n else None,
    }


def _abs_faithful(rec: dict) -> bool:
    m = rec.get("measured_lag_ms")
    if m is None:
        return False
    return abs(abs(float(m)) - abs(float(rec["true_lag_ms"]))) <= faithful_tol_ms(rec["fs"])


def _consistency_in_box_all_rates(records: list[dict], grouped: dict) -> dict:
    """Same window + delay; delay is inside ±30 samples at 20, 50, and 100 Hz."""

    def _run(mode: str) -> dict:
        n = 0
        n_same = 0
        for (wid, d_ms), cell in grouped.items():
            trio = []
            ok = True
            for fs in FS_GRID:
                recs = [
                    r
                    for r in records
                    if r["window_id"] == wid and r["delay_ms"] == d_ms and r["fs"] == fs
                ]
                if not recs or not recs[0]["length_ok"] or not recs[0]["in_sample_box"]:
                    ok = False
                    break
                trio.append(cell[mode][fs])
            if not ok:
                continue
            n += 1
            n_same += int(len(set(trio)) == 1)
        return {"n_groups": n, "n_consistent": n_same, "rate": n_same / n if n else None}

    return {"mode_a": _run("mode_a"), "mode_b": _run("mode_b")}


def summarize(records: list[dict]) -> dict:
    by_delay_fs = {}
    for d in DELAYS_MS:
        by_delay_fs[str(d)] = {}
        for fs in FS_GRID:
            sub = [r for r in records if r["delay_ms"] == float(d) and r["fs"] == fs]
            by_delay_fs[str(d)][str(int(fs))] = {
                "n": len(sub),
                "n_length_ok": sum(1 for r in sub if r["length_ok"]),
                "n_in_box": sum(1 for r in sub if r["in_sample_box"]),
                "n_faithful": sum(1 for r in sub if _abs_faithful(r)),
                "mode_a": _rates(sub, "mode_a"),
                "mode_b": _rates(sub, "mode_b"),
            }

    valid = [r for r in records if r["length_ok"]]
    measurable = [r for r in valid if r["in_sample_box"]]
    unmeasurable = [r for r in valid if not r["in_sample_box"]]
    short = [r for r in records if not r["length_ok"]]

    # Same physical delay, same window, three rates.
    grouped = defaultdict(lambda: {"mode_a": {}, "mode_b": {}})
    for r in records:
        key = (r["window_id"], r["delay_ms"])
        grouped[key]["mode_a"][r["fs"]] = r["mode_a"]["verdict"]
        grouped[key]["mode_b"][r["fs"]] = r["mode_b"]["verdict"]

    def _consistency(mode: str, require_valid_length: bool) -> dict:
        n = 0
        n_same = 0
        for (wid, d_ms), cell in grouped.items():
            vs = cell[mode]
            if require_valid_length:
                trio = []
                ok = True
                for fs in FS_GRID:
                    recs = [
                        r
                        for r in records
                        if r["window_id"] == wid and r["delay_ms"] == d_ms and r["fs"] == fs
                    ]
                    if not recs or not recs[0]["length_ok"]:
                        ok = False
                        break
                    trio.append(vs[fs])
                if not ok:
                    continue
                n += 1
                n_same += int(len(set(trio)) == 1)
            else:
                n += 1
                n_same += int(len(set(vs.values())) == 1)
        return {"n_groups": n, "n_consistent": n_same, "rate": n_same / n if n else None}

    # Cross-mode agreement on measurable in-box items.
    n_agree = sum(1 for r in measurable if r["mode_a"]["verdict"] == r["mode_b"]["verdict"])
    n_meas = len(measurable)

    # Support preservation: delays physically inside 300 ms and measurable.
    inside_phys = [r for r in measurable if abs(r["true_lag_ms"]) < PHYSICAL_BOUND_MS]
    at_or_over_phys = [r for r in measurable if abs(r["true_lag_ms"]) >= PHYSICAL_BOUND_MS]

    def _preserve(rows: list[dict], mode_key: str, expect: str) -> dict:
        n = len(rows)
        n_ok = sum(1 for r in rows if r[mode_key]["verdict"] == expect)
        return {"n": n, "n_match": n_ok, "rate": n_ok / n if n else None}

    return {
        "n_items": len(records),
        "n_valid_length": len(valid),
        "n_short": len(short),
        "n_measurable": n_meas,
        "n_unmeasurable": len(unmeasurable),
        "n_faithful": sum(1 for r in records if _abs_faithful(r)),
        "faithful_rate_measurable": (
            sum(1 for r in measurable if _abs_faithful(r)) / n_meas if n_meas else None
        ),
        "by_delay_fs": by_delay_fs,
        "mode_a_all": _rates(records, "mode_a"),
        "mode_b_all": _rates(records, "mode_b"),
        "mode_a_measurable": _rates(measurable, "mode_a"),
        "mode_b_measurable": _rates(measurable, "mode_b"),
        "consistency_all_rates": {
            "mode_a": _consistency("mode_a", False),
            "mode_b": _consistency("mode_b", False),
        },
        "consistency_valid_length": {
            "mode_a": _consistency("mode_a", True),
            "mode_b": _consistency("mode_b", True),
        },
        "consistency_in_box_all_rates": _consistency_in_box_all_rates(records, grouped),
        "cross_mode_agree_measurable": {
            "n": n_meas,
            "n_agree": n_agree,
            "rate": n_agree / n_meas if n_meas else None,
        },
        "support_if_physically_inside": {
            "mode_a": _preserve(inside_phys, "mode_a", "SUPPORTED"),
            "mode_b": _preserve(inside_phys, "mode_b", "SUPPORTED"),
        },
        "contradict_if_physically_outside_and_measurable": {
            "mode_a": _preserve(at_or_over_phys, "mode_a", "CONTRADICTED"),
            "mode_b": _preserve(at_or_over_phys, "mode_b", "CONTRADICTED"),
        },
        "false_commitment": {
            "mode_a_all_invalid": _commit_on_invalid(records, "mode_a"),
            "mode_b_all_invalid": _commit_on_invalid(records, "mode_b"),
            "mode_a_short": _commit_on_invalid(short, "mode_a"),
            "mode_b_short": _commit_on_invalid(short, "mode_b"),
            "mode_a_outside_search": _commit_on_invalid(unmeasurable, "mode_a"),
            "mode_b_outside_search": _commit_on_invalid(unmeasurable, "mode_b"),
        },
        "bounds": {
            "sample_bound_samples": SAMPLE_BOUND,
            "physical_bound_ms": PHYSICAL_BOUND_MS,
        },
    }
