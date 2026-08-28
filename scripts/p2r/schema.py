"""Frozen executable ClaimProgram schema. Inference never carries gold/split/family IDs."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

MEASUREMENTS = (
    "dominant_frequency",
    "rms_amplitude",
    "peak_amplitude",
    "signal_range",
    "trend_ratio",
    "cross_channel_lag_ms",
    "periodicity_strength",
    "spectral_energy_ratio_low",
)

CONNECTIVES = ("SINGLE", "AND", "OR", "IF_THEN")
# Comparator is the asserted comparison, not a gold truth bit.
COMPARATORS = ("eq", "gt", "lt", "similar", "different")
UNITS = (
    "Hz",
    "ms",
    "s",
    "raw",
    "ratio",
    "fraction",
    "percent",
    "score_0_1",
)
FORBIDDEN_INFERENCE_KEYS = (
    "gt_verdict",
    "gold_composed_verdict",
    "gold_program",
    "paraphrase_family_id",
    "generation_family",
    "structure_type",
    "template_id",
    "source_generator",
    "split",
    "semantic_program_id",
    "claim_family",
)

SCHEMA_VERSION = "p2r_claimprogram_v1"


@dataclass
class Predicate:
    measurement: str
    channel_a: str
    comparator: str
    channel_b: Optional[str] = None
    reference_value: Optional[float] = None
    reference_channel: Optional[str] = None
    unit: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ClaimProgram:
    connective: str
    predicates: list[Predicate] = field(default_factory=list)
    parse_status: str = "OK"  # OK | AMBIGUOUS | UNPARSEABLE
    parse_reason: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "connective": self.connective,
            "predicates": [p.to_dict() for p in self.predicates],
            "parse_status": self.parse_status,
            "parse_reason": self.parse_reason,
        }


def schema_catalog() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "connectives": list(CONNECTIVES),
        "measurements": list(MEASUREMENTS),
        "comparators": list(COMPARATORS),
        "units": list(UNITS),
        "predicate_fields": {
            "measurement": "required enum MEASUREMENTS",
            "channel_a": "required string; must resolve to an available channel",
            "channel_b": "optional; required iff measurement==cross_channel_lag_ms",
            "comparator": "required enum COMPARATORS",
            "reference_value": "optional float; required for eq/gt/lt vs a numeric reference",
            "reference_channel": "optional; required for gt/lt/similar/different vs another channel",
            "unit": "optional enum UNITS",
        },
        "parse_status": ["OK", "AMBIGUOUS", "UNPARSEABLE"],
        "forbidden_inference_keys": list(FORBIDDEN_INFERENCE_KEYS),
    }


def schema_hash() -> str:
    payload = json.dumps(schema_catalog(), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def assert_no_leakage(payload: dict) -> None:
    extra = set(payload) & set(FORBIDDEN_INFERENCE_KEYS)
    if extra:
        raise ValueError(f"inference payload contains forbidden keys: {sorted(extra)}")
