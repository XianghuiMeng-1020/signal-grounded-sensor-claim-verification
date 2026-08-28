"""Adversarial extraction items with independently defined programs. No holdout windows."""
from __future__ import annotations

import json
from pathlib import Path

from .schema import ClaimProgram, Predicate

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "benchmarks" / "p2r" / "adversarial_extraction.json"


def build() -> list[dict]:
    avail = ["hand_accel", "chest_accel"]
    items = []

    def add(text, program: ClaimProgram, gold_verdict_role: str, note: str):
        items.append({
            "surface_text": text,
            "available_channels": avail,
            "fs": 100.0,
            "gold_program": program.to_dict(),
            "expected_parse": program.parse_status,
            "role": gold_verdict_role,  # executable | non_executable
            "note": note,
            "dataset": "ADVERSARIAL_SYNTH",
        })

    p_rms = ClaimProgram("SINGLE", [Predicate("rms_amplitude", "hand_accel", "eq", reference_value=1.85, unit="raw")])
    add("The RMS amplitude of the hand channel is approximately 1.85 raw units.", p_rms, "executable", "direct")
    add("Approximately 1.85 raw units is the RMS amplitude on the hand channel.", p_rms, "executable", "clause_reorder")
    add("The quadratic-mean amplitude of the hand stream is about 1.85 raw units.", p_rms, "executable", "synonym")
    add("Hand RMS is about 1.85 raw units.", p_rms, "executable", "channel_alias")
    add("Ignoring battery notes, hand RMS is about 1.85 raw units.", p_rms, "executable", "distractor")
    add("The session lasted 12 minutes; hand RMS is about 1.85 raw units.", p_rms, "executable", "irrelevant_number")
    add("Between 0.10 and 9.99, the operative hand RMS claim is 1.85 raw units.", p_rms, "executable", "two_numbers")
    add("Hand RMS sits around 1.85 raw units.", p_rms, "executable", "colloquial")
    add("The hand channel RMS amplitude is 1.85 raw units.", p_rms, "executable", "passive")

    p_gt = ClaimProgram("SINGLE", [Predicate("rms_amplitude", "hand_accel", "gt", reference_channel="chest_accel")])
    add("The hand channel has higher RMS amplitude than the chest channel.", p_gt, "executable", "compare")
    add("The chest channel has lower RMS amplitude than the hand channel.", 
        ClaimProgram("SINGLE", [Predicate("rms_amplitude", "chest_accel", "lt", reference_channel="hand_accel")]),
        "executable", "reversed_wording_different_program")

    p_hz = ClaimProgram("SINGLE", [Predicate("dominant_frequency", "hand_accel", "eq", reference_value=4.0, unit="Hz")])
    add("The hand channel oscillates at roughly 4.0 cycles per second.", p_hz, "executable", "mixed_unit")

    add("The hand channel does not have an RMS near 1.85 raw units.",
        ClaimProgram("SINGLE", [], parse_status="UNPARSEABLE", parse_reason="negation_not_in_schema"),
        "non_executable", "negation")
    add("It is larger than the other one.",
        ClaimProgram("SINGLE", [], parse_status="AMBIGUOUS", parse_reason="pronoun"),
        "non_executable", "pronoun_ambiguity")
    add("Heart rate was above 130 bpm.",
        ClaimProgram("SINGLE", [], parse_status="UNPARSEABLE", parse_reason="unsupported_measurement"),
        "non_executable", "unsupported")
    add("The RMS is high.",
        ClaimProgram("SINGLE", [], parse_status="UNPARSEABLE", parse_reason="malformed"),
        "non_executable", "malformed")
    add("The gyroscope z-axis RMS is 1.2 raw units.",
        ClaimProgram("SINGLE", [], parse_status="UNPARSEABLE", parse_reason="missing_channel"),
        "non_executable", "missing_channel")
    add("The hand and ankle channels have similar RMS.",
        ClaimProgram("SINGLE", [], parse_status="UNPARSEABLE", parse_reason="contradictory_channel"),
        "non_executable", "contradictory_names")

    p_and = ClaimProgram("AND", [
        Predicate("spectral_energy_ratio_low", "hand_accel", "gt", reference_channel="chest_accel"),
        Predicate("rms_amplitude", "hand_accel", "similar", reference_channel="chest_accel"),
    ])
    add("The hand channel contains more low-frequency spectral energy than the chest channel, while the two channels have similar RMS amplitude.",
        p_and, "executable", "and")
    return items


def save() -> list[dict]:
    items = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(items, indent=2), encoding="utf-8")
    return items
