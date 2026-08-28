"""Frozen activity → family → measurement claim.

Written from textbook locomotion / IMU practice, not from verifier output,
not from a threshold search, and not from an LLM.

Only scale-free primitives are used so PAMAP2 (m/s^2), WISDM (g), and
MHEALTH do not require a fitted amplitude scale.

HARTH is not in this dictionary: it has no existing p2 window loader.
"""
from __future__ import annotations

import hashlib
import json

from p2r.schema import ClaimProgram, Predicate

DICTIONARY_ID = "p4_har_claim_dictionary_v1"

# Claim families. Thresholds are physical / fractional, not dataset-fitted.
# static: quasi-static windows concentrate energy below 3 Hz.
# walk:   step-like dominant rate (adult walk ~1.5–2 Hz; 0.8 Hz is a low cut).
# run:    faster cyclic rate (jog/run typically >2 Hz).
FAMILY_CLAIMS = {
    "static": {
        "measurement": "spectral_energy_ratio_low",
        "comparator": "gt",
        "reference_value": 0.50,
        "unit": "fraction",
    },
    "walk": {
        "measurement": "dominant_frequency",
        "comparator": "gt",
        "reference_value": 0.8,
        "unit": "Hz",
    },
    "run": {
        "measurement": "dominant_frequency",
        "comparator": "gt",
        "reference_value": 2.0,
        "unit": "Hz",
    },
}

# (dataset, activity_code) → family. Codes match p2.windows majority labels.
ACTIVITY_FAMILY: dict[tuple[str, str], str] = {
    # PAMAP2 protocol IDs
    ("PAMAP2", "1"): "static",   # lying
    ("PAMAP2", "2"): "static",   # sitting
    ("PAMAP2", "3"): "static",   # standing
    ("PAMAP2", "4"): "walk",     # walking
    ("PAMAP2", "5"): "run",      # running
    ("PAMAP2", "6"): "walk",     # cycling (cadence ~1 Hz class)
    ("PAMAP2", "7"): "walk",     # Nordic walking
    ("PAMAP2", "12"): "walk",    # ascending stairs
    ("PAMAP2", "13"): "walk",    # descending stairs
    ("PAMAP2", "24"): "run",     # rope jumping
    # WISDM watch letters
    ("WISDM", "A"): "walk",      # walking
    ("WISDM", "B"): "run",       # jogging
    ("WISDM", "C"): "walk",      # stairs
    ("WISDM", "D"): "static",    # sitting
    ("WISDM", "E"): "static",    # standing
    # MHEALTH IDs
    ("MHEALTH", "1"): "static",  # standing still
    ("MHEALTH", "2"): "static",  # sitting
    ("MHEALTH", "3"): "static",  # lying
    ("MHEALTH", "4"): "walk",    # walking
    ("MHEALTH", "5"): "walk",    # climbing stairs
    ("MHEALTH", "9"): "walk",    # cycling
    ("MHEALTH", "10"): "run",    # jogging
    ("MHEALTH", "11"): "run",    # running
    ("MHEALTH", "12"): "run",    # jump front/back
}


def activity_key(dataset: str, activity) -> tuple[str, str]:
    return (dataset, str(activity).strip())


def family_of(dataset: str, activity) -> str | None:
    return ACTIVITY_FAMILY.get(activity_key(dataset, activity))


def mappable(dataset: str, activity) -> bool:
    return family_of(dataset, activity) is not None


def claim_spec(family: str) -> dict:
    return dict(FAMILY_CLAIMS[family])


def make_program(family: str, channel_a: str) -> ClaimProgram:
    spec = FAMILY_CLAIMS[family]
    pred = Predicate(
        measurement=spec["measurement"],
        channel_a=channel_a,
        comparator=spec["comparator"],
        channel_b=None,
        reference_value=float(spec["reference_value"]),
        reference_channel=None,
        unit=spec["unit"],
    )
    return ClaimProgram("SINGLE", [pred], parse_status="OK")


def dictionary_sha256() -> str:
    payload = {
        "id": DICTIONARY_ID,
        "families": FAMILY_CLAIMS,
        "activity_family": {f"{d}|{a}": fam for (d, a), fam in sorted(ACTIVITY_FAMILY.items())},
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()
