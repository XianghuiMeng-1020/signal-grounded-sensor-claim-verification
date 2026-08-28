"""HAR baseline never emits UNKNOWN / UNVERIFIABLE."""
from p4_har_claim.evaluate import fcr, har_verdict, metrics


def test_har_verdict_is_binary_class_match():
    assert har_verdict("walk", "walk") == "SUPPORTED"
    assert har_verdict("run", "walk") == "CONTRADICTED"
    assert har_verdict(None, "walk") == "CONTRADICTED"


def test_har_verdict_never_unverifiable():
    for pred in ("static", "walk", "run", None, "other"):
        assert har_verdict(pred, "static") in ("SUPPORTED", "CONTRADICTED")


def test_fcr_is_commitment_on_gold_unknown():
    rows = [
        {"gold": "UNVERIFIABLE", "har": "SUPPORTED", "proposed": "UNVERIFIABLE"},
        {"gold": "UNVERIFIABLE", "har": "CONTRADICTED", "proposed": "UNVERIFIABLE"},
        {"gold": "SUPPORTED", "har": "SUPPORTED", "proposed": "SUPPORTED"},
    ]
    assert fcr(rows, "har") == 1.0
    assert fcr(rows, "proposed") == 0.0


def test_secondary_metrics_separate_legal_and_unknown():
    rows = [
        {"gold": "SUPPORTED", "har": "SUPPORTED", "proposed": "SUPPORTED", "slice": "legal"},
        {"gold": "CONTRADICTED", "har": "SUPPORTED", "proposed": "CONTRADICTED", "slice": "legal"},
        {"gold": "UNVERIFIABLE", "har": "SUPPORTED", "proposed": "UNVERIFIABLE", "slice": "illegal"},
    ]
    m = metrics(rows)
    assert m["har"]["unknown_rate"] == 0.0
    assert m["proposed"]["unknown_rate"] > 0.0
    assert m["har"]["fcr"] == 1.0
    assert m["proposed"]["fcr"] == 0.0
