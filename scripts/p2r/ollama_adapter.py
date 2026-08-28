"""Ollama adapter for the P2R semantic compiler.

Role is unchanged: natural-language claim → ClaimProgram.
Never inspects waveforms, never computes DSP, never assigns a verdict.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .schema import (
    COMPARATORS,
    CONNECTIVES,
    FORBIDDEN_INFERENCE_KEYS,
    MEASUREMENTS,
    UNITS,
    ClaimProgram,
    Predicate,
    assert_no_leakage,
    schema_catalog,
    schema_hash,
)
from .validator import validate_program

ROOT = Path(__file__).resolve().parents[2]
CACHE_ROOT = ROOT / "results" / "p2r_lm_local" / "cache"
PROMPT_ID = "p2r_lm_local_v1"
SEED = 20270823
DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434")
TEMPERATURE = 0.0
TIMEOUT_S = 180
RETRIES = 1

FORBIDDEN_PAYLOAD_KEYS = tuple(FORBIDDEN_INFERENCE_KEYS) + (
    "gold_program",
    "gold_composed_verdict",
    "gold_predicate_truth",
    "semantic_program",
    "template_id",
    "paraphrase_family_id",
    "generation_family",
    "difficulty",
    "primitive",
    "source_dataset",
    "claim_id",
    "filename",
    "source_window_id",
    "subject",
    "session",
    "window_index",
    "benchmark_version",
    "provenance",
    "margin",
    "margin_band",
    "surface_style",
    "unverifiable_family",
    "threshold_or_value",
    "reference_measurement",
    "activity",
    "channels",
    "channels_data",
)

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "connective": {"type": "string", "enum": list(CONNECTIVES)},
        "predicates": {
            "type": "array",
            "maxItems": 3,
            "items": {
                "type": "object",
                "properties": {
                    "measurement": {"type": "string", "enum": list(MEASUREMENTS)},
                    "channel_a": {"type": "string"},
                    "channel_b": {"type": ["string", "null"]},
                    "comparator": {"type": "string", "enum": list(COMPARATORS)},
                    "reference_value": {"type": ["number", "null"]},
                    "reference_channel": {"type": ["string", "null"]},
                    "unit": {"type": ["string", "null"], "enum": list(UNITS) + [None]},
                },
                "required": ["measurement", "channel_a", "comparator"],
            },
        },
        "parse_status": {"type": "string", "enum": ["OK", "AMBIGUOUS", "UNPARSEABLE"]},
        "parse_reason": {"type": ["string", "null"]},
    },
    "required": ["connective", "predicates", "parse_status"],
}

SYSTEM_V1 = """You are a semantic compiler for wearable IMU claims.
Your only job is to translate one natural-language sentence into an executable ClaimProgram JSON object.
You do NOT see any waveform. You do NOT compute DSP measurements. You do NOT judge whether the claim is true.
You must NOT output a final truth label of any kind.

Return ONLY a JSON object with keys:
  connective: SINGLE | AND | OR | IF_THEN
  predicates: 0-3 predicate objects
  parse_status: OK | AMBIGUOUS | UNPARSEABLE
  parse_reason: string or null

Each predicate has:
  measurement: one of dominant_frequency, rms_amplitude, peak_amplitude, signal_range, trend_ratio, cross_channel_lag_ms, periodicity_strength, spectral_energy_ratio_low
  channel_a: an available channel name (or a name that uniquely resolves to one)
  channel_b: required only for cross_channel_lag_ms
  comparator: eq | gt | lt | similar | different
  reference_value: number when comparing to a numeric reference
  reference_channel: available channel name when comparing two channels
  unit: Hz | ms | s | raw | ratio | fraction | percent | score_0_1 or null

Legal comparators:
  eq + reference_value = the sentence asserts a specific numeric value
  gt/lt + reference_value = vs a numeric threshold
  gt/lt/similar/different + reference_channel = vs another channel

Channel representation:
  Use the exact available channel names. Ordinary placement words resolve only when unique:
  hand → hand_accel; chest → chest_accel; ankle → ankle_accel; x / x-axis → x_accel; y / y-axis → y_accel.
  If two available channels remain compatible with the wording, parse_status=AMBIGUOUS and predicates=[].

General schema interpretation (not dataset-specific hacks):
  RMS / root-mean-square → rms_amplitude
  peak amplitude → peak_amplitude
  peak-to-peak / range → signal_range
  dominant frequency / spectral peak frequency → dominant_frequency
  cross-channel timing lag / lag → cross_channel_lag_ms (needs two channels)
  periodicity strength → periodicity_strength
  low-frequency spectral energy fraction / low-band energy fraction → spectral_energy_ratio_low
  energy increasing / rising across the window → trend_ratio, comparator=gt, reference_value=1.0, unit=ratio
  energy decreasing or flat across the window → trend_ratio, comparator=lt, reference_value=1.0, unit=ratio
  Copy numeric literals from the sentence. Do not invent extra precision.

Connectives:
  one predicate → SINGLE
  both clauses required → AND
  either clause sufficient → OR
  if A then B → IF_THEN with exactly two predicates in order (antecedent, consequent)

Set parse_status=UNPARSEABLE and predicates=[] when the claim is outside the supported executable schema, including:
  unsupported measurement (entropy, jerk, emotion, heart rate, physiology, battery)
  unavailable or missing channel that is not in the available-channel list
  unknown/invalid sampling-rate metadata stated in the sentence
  stated corrupt/missing evidence that makes the measurement non-executable
  nested logic beyond AND / OR / IF_THEN
  more than 3 predicates
  purely qualitative wording with no executable measurement
  window too short / insufficient samples stated as the reason the measurement cannot be trusted

Set parse_status=AMBIGUOUS and predicates=[] when a required slot is not uniquely determined, including:
  ambiguous channel reference
  comparator that is simultaneously greater and smaller / either direction

If parse_status is not OK, predicates MUST be [].
Do not include any keys other than the schema keys.
Think only if required internally; the user-visible answer must be the JSON object alone.
"""

FEWSHOT_V1 = [
    {
        "sentence": "The dominant frequency of the chest channel is approximately 2.95 Hz.",
        "channels": ["hand_accel", "chest_accel"],
        "fs": 100.0,
        "output": {
            "connective": "SINGLE",
            "predicates": [{
                "measurement": "dominant_frequency",
                "channel_a": "chest_accel",
                "channel_b": None,
                "comparator": "eq",
                "reference_value": 2.95,
                "reference_channel": None,
                "unit": "Hz",
            }],
            "parse_status": "OK",
            "parse_reason": None,
        },
    },
    {
        "sentence": "The hand channel has a higher RMS amplitude than the chest channel.",
        "channels": ["hand_accel", "chest_accel"],
        "fs": 100.0,
        "output": {
            "connective": "SINGLE",
            "predicates": [{
                "measurement": "rms_amplitude",
                "channel_a": "hand_accel",
                "channel_b": None,
                "comparator": "gt",
                "reference_value": None,
                "reference_channel": "chest_accel",
                "unit": "raw",
            }],
            "parse_status": "OK",
            "parse_reason": None,
        },
    },
    {
        "sentence": "Energy in the x channel is increasing across the window.",
        "channels": ["x_accel", "y_accel"],
        "fs": 20.0,
        "output": {
            "connective": "SINGLE",
            "predicates": [{
                "measurement": "trend_ratio",
                "channel_a": "x_accel",
                "channel_b": None,
                "comparator": "gt",
                "reference_value": 1.0,
                "reference_channel": None,
                "unit": "ratio",
            }],
            "parse_status": "OK",
            "parse_reason": None,
        },
    },
    {
        "sentence": "The cross-channel timing lag of the x and y channels is approximately 50 ms.",
        "channels": ["x_accel", "y_accel"],
        "fs": 20.0,
        "output": {
            "connective": "SINGLE",
            "predicates": [{
                "measurement": "cross_channel_lag_ms",
                "channel_a": "x_accel",
                "channel_b": "y_accel",
                "comparator": "eq",
                "reference_value": 50.0,
                "reference_channel": None,
                "unit": "ms",
            }],
            "parse_status": "OK",
            "parse_reason": None,
        },
    },
    {
        "sentence": "The hand channel has a lower RMS amplitude than the chest channel, and the hand channel has a lower dominant frequency than the chest channel.",
        "channels": ["hand_accel", "chest_accel"],
        "fs": 100.0,
        "output": {
            "connective": "AND",
            "predicates": [
                {
                    "measurement": "rms_amplitude",
                    "channel_a": "hand_accel",
                    "comparator": "lt",
                    "reference_channel": "chest_accel",
                    "reference_value": None,
                    "channel_b": None,
                    "unit": "raw",
                },
                {
                    "measurement": "dominant_frequency",
                    "channel_a": "hand_accel",
                    "comparator": "lt",
                    "reference_channel": "chest_accel",
                    "reference_value": None,
                    "channel_b": None,
                    "unit": "Hz",
                },
            ],
            "parse_status": "OK",
            "parse_reason": None,
        },
    },
    {
        "sentence": "If the x channel has a higher periodicity strength than the y channel, then the x channel has a higher low-frequency spectral energy fraction than the y channel.",
        "channels": ["x_accel", "y_accel"],
        "fs": 20.0,
        "output": {
            "connective": "IF_THEN",
            "predicates": [
                {
                    "measurement": "periodicity_strength",
                    "channel_a": "x_accel",
                    "comparator": "gt",
                    "reference_channel": "y_accel",
                    "reference_value": None,
                    "channel_b": None,
                    "unit": "score_0_1",
                },
                {
                    "measurement": "spectral_energy_ratio_low",
                    "channel_a": "x_accel",
                    "comparator": "gt",
                    "reference_channel": "y_accel",
                    "reference_value": None,
                    "channel_b": None,
                    "unit": "fraction",
                },
            ],
            "parse_status": "OK",
            "parse_reason": None,
        },
    },
    {
        "sentence": "Jerk entropy of the recording exceeds 4.2 nats.",
        "channels": ["hand_accel", "chest_accel"],
        "fs": 100.0,
        "output": {
            "connective": "SINGLE",
            "predicates": [],
            "parse_status": "UNPARSEABLE",
            "parse_reason": "unsupported_measurement",
        },
    },
    {
        "sentence": "The sensor has higher RMS than the other sensor.",
        "channels": ["hand_accel", "chest_accel"],
        "fs": 100.0,
        "output": {
            "connective": "SINGLE",
            "predicates": [],
            "parse_status": "AMBIGUOUS",
            "parse_reason": "ambiguous_channel",
        },
    },
]

SYSTEM_V2 = SYSTEM_V1 + """

Additional general interpretation rules:
- "near X" / "approximately X" / "X is the <measurement>" is comparator=eq with reference_value=X, even if the sentence adds "rather than a markedly different value". That phrase does NOT mean comparator=similar.
- comparator=similar or different is legal ONLY when two channels are being compared. Never use similar/different against a number.
- cross_channel_lag_ms always needs channel_a and channel_b set to the two named available channels, in sentence order. A lag claim with only one channel filled is invalid; fill both instead of abstaining.
- "half-window energy ratio" is trend_ratio, not spectral_energy_ratio_low. The unit is ratio. A trailing "x" after that number is ratio, not a channel.
- Do not guess a channel when the wording is only "the sensor" / "the other sensor" / "the recording" and more than one channel is available: parse_status=AMBIGUOUS.
- If the sentence states that the file/evidence is corrupt or missing, or that the sampling rate is unknown, return UNPARSEABLE even if a number is also mentioned.
"""

FEWSHOT_V2 = FEWSHOT_V1 + [
    {
        "sentence": "The chest channel shows dominant frequency near 3.10 Hz, rather than a markedly different value.",
        "channels": ["hand_accel", "chest_accel"],
        "fs": 100.0,
        "output": {
            "connective": "SINGLE",
            "predicates": [{
                "measurement": "dominant_frequency",
                "channel_a": "chest_accel",
                "channel_b": None,
                "comparator": "eq",
                "reference_value": 3.10,
                "reference_channel": None,
                "unit": "Hz",
            }],
            "parse_status": "OK",
            "parse_reason": None,
        },
    },
    {
        "sentence": "Approximately 0.80 x is the half-window energy ratio measured on the y channel.",
        "channels": ["x_accel", "y_accel"],
        "fs": 20.0,
        "output": {
            "connective": "SINGLE",
            "predicates": [{
                "measurement": "trend_ratio",
                "channel_a": "y_accel",
                "channel_b": None,
                "comparator": "eq",
                "reference_value": 0.80,
                "reference_channel": None,
                "unit": "ratio",
            }],
            "parse_status": "OK",
            "parse_reason": None,
        },
    },
    {
        "sentence": "The file is corrupt; nonetheless the chest RMS is 4.0 raw units.",
        "channels": ["hand_accel", "chest_accel"],
        "fs": 100.0,
        "output": {
            "connective": "SINGLE",
            "predicates": [],
            "parse_status": "UNPARSEABLE",
            "parse_reason": "corrupt_or_missing_evidence",
        },
    },
]

SYSTEM_V3A = SYSTEM_V2 + """

Semantic compilation procedure (apply on every sentence):
1. Segment the sentence into independent measurement clauses before choosing a connective.
2. Map each clause to exactly one frozen primitive. Do not merge two measurements into one predicate.
3. The number of extracted predicates MUST equal the number of independent clauses (1, 2, or 3). AND / OR / IF_THEN never drop a clause.
4. Connective: one clause → SINGLE; every clause required → AND; any clause sufficient → OR; condition then consequence → IF_THEN with predicates in that order (antecedent first).
5. Surface order of clauses is the logical order unless a clear if/then or whenever/then marker reverses it — then follow the logical marker, not the first noun.

Primitive discrimination (general, not lexical lookup):
- RMS / root-mean-square / quadratic-mean level → rms_amplitude
- peak / maximum excursion / largest absolute deviation → peak_amplitude
- range / span / peak-to-trough / peak-to-peak → signal_range
- dominant / spectral-peak / leading Fourier frequency → dominant_frequency
- inter-channel delay / cross-sensor offset / timing lag → cross_channel_lag_ms (always two channels)
- periodicity / cyclic regularity / repeatability coefficient → periodicity_strength
- sub-3 Hz / low-band energy share or occupancy → spectral_energy_ratio_low
- late-to-early or second-half versus first-half energy quotient → trend_ratio
These are schema definitions. Do not invent a ninth measurement.

Channels: bind every named placement to an available channel. Lag: channel_a then channel_b in the order the two sites are named.
Comparators: a stated number with equals/is/reads → eq; greater/exceeds → gt; less/below → lt. similar/different only for two channels.
Unsupported measurement, unknown sampling rate, corrupt evidence, exclusive-or / nested logic beyond AND/OR/IF_THEN, or a named channel that is not available → UNPARSEABLE, predicates=[].
A required slot that is not unique (the sensor / both greater and less) → AMBIGUOUS, predicates=[].
"""

FEWSHOT_V3A = FEWSHOT_V2 + [
    {
        "sentence": "Two measurements are required together: ankle dominant frequency is 1.75 Hz and chest RMS is 0.640 raw units.",
        "channels": ["ankle_accel", "chest_accel"],
        "fs": 100.0,
        "output": {
            "connective": "AND",
            "predicates": [
                {
                    "measurement": "dominant_frequency",
                    "channel_a": "ankle_accel",
                    "channel_b": None,
                    "comparator": "eq",
                    "reference_value": 1.75,
                    "reference_channel": None,
                    "unit": "Hz",
                },
                {
                    "measurement": "rms_amplitude",
                    "channel_a": "chest_accel",
                    "channel_b": None,
                    "comparator": "eq",
                    "reference_value": 0.640,
                    "reference_channel": None,
                    "unit": "raw",
                },
            ],
            "parse_status": "OK",
            "parse_reason": None,
        },
    },
    {
        "sentence": "If the hand peak-to-peak range exceeds 2.20 raw units, then the hand peak amplitude exceeds 1.10 raw units.",
        "channels": ["hand_accel", "chest_accel"],
        "fs": 100.0,
        "output": {
            "connective": "IF_THEN",
            "predicates": [
                {
                    "measurement": "signal_range",
                    "channel_a": "hand_accel",
                    "channel_b": None,
                    "comparator": "gt",
                    "reference_value": 2.20,
                    "reference_channel": None,
                    "unit": "raw",
                },
                {
                    "measurement": "peak_amplitude",
                    "channel_a": "hand_accel",
                    "channel_b": None,
                    "comparator": "gt",
                    "reference_value": 1.10,
                    "reference_channel": None,
                    "unit": "raw",
                },
            ],
            "parse_status": "OK",
            "parse_reason": None,
        },
    },
    {
        "sentence": "The x periodicity strength is 0.410, the y low-band energy fraction is 0.330, and the x-to-y timing lag is 12.5 ms.",
        "channels": ["x_accel", "y_accel"],
        "fs": 20.0,
        "output": {
            "connective": "AND",
            "predicates": [
                {
                    "measurement": "periodicity_strength",
                    "channel_a": "x_accel",
                    "channel_b": None,
                    "comparator": "eq",
                    "reference_value": 0.410,
                    "reference_channel": None,
                    "unit": "score_0_1",
                },
                {
                    "measurement": "spectral_energy_ratio_low",
                    "channel_a": "y_accel",
                    "channel_b": None,
                    "comparator": "eq",
                    "reference_value": 0.330,
                    "reference_channel": None,
                    "unit": "fraction",
                },
                {
                    "measurement": "cross_channel_lag_ms",
                    "channel_a": "x_accel",
                    "channel_b": "y_accel",
                    "comparator": "eq",
                    "reference_value": 12.5,
                    "reference_channel": None,
                    "unit": "ms",
                },
            ],
            "parse_status": "OK",
            "parse_reason": None,
        },
    },
]

SYSTEM_V3B = SYSTEM_V2 + """

Work clause-by-clause. First count independent measurement claims in the sentence (N=1..3). Return exactly N predicates. If you cannot fill every slot legally, do not return a shorter OK program — use UNPARSEABLE or AMBIGUOUS.

Connectives after the count:
- N=1 → SINGLE
- N=2 and both required / joint / in addition / likewise / both of the following → AND
- N=2 and either / otherwise / at least one → OR
- N=2 and whenever/if/antecedent then consequent → IF_THEN (predicate 1 = condition)
- N=3 → AND unless the sentence is a 3-way disjunction

Primitive families are mutually exclusive:
amplitude shape: rms_amplitude vs peak_amplitude vs signal_range
spectrum: dominant_frequency vs spectral_energy_ratio_low
time structure: trend_ratio vs periodicity_strength vs cross_channel_lag_ms
Never substitute one family member for another.

Copy numbers as written. Bind each placement word to one available channel. Lag needs two channels.
XOR / exactly-one-never-both / nested mixtures / heart-rate / entropy / unknown fs / missing named site → UNPARSEABLE.
Non-unique channel or simultaneous gt and lt → AMBIGUOUS.
"""

FEWSHOT_V3B = FEWSHOT_V2 + [
    {
        "sentence": "At least one claim is sufficient: ankle peak-to-peak range exceeds 3.5 raw units, or chest RMS is below 0.55 raw units.",
        "channels": ["ankle_accel", "chest_accel"],
        "fs": 100.0,
        "output": {
            "connective": "OR",
            "predicates": [
                {
                    "measurement": "signal_range",
                    "channel_a": "ankle_accel",
                    "channel_b": None,
                    "comparator": "gt",
                    "reference_value": 3.5,
                    "reference_channel": None,
                    "unit": "raw",
                },
                {
                    "measurement": "rms_amplitude",
                    "channel_a": "chest_accel",
                    "channel_b": None,
                    "comparator": "lt",
                    "reference_value": 0.55,
                    "reference_channel": None,
                    "unit": "raw",
                },
            ],
            "parse_status": "OK",
            "parse_reason": None,
        },
    },
    {
        "sentence": "If the hand dominant frequency is below 8.0 Hz then the hand half-window energy ratio is below 1.0.",
        "channels": ["hand_accel", "chest_accel"],
        "fs": 100.0,
        "output": {
            "connective": "IF_THEN",
            "predicates": [
                {
                    "measurement": "dominant_frequency",
                    "channel_a": "hand_accel",
                    "channel_b": None,
                    "comparator": "lt",
                    "reference_value": 8.0,
                    "reference_channel": None,
                    "unit": "Hz",
                },
                {
                    "measurement": "trend_ratio",
                    "channel_a": "hand_accel",
                    "channel_b": None,
                    "comparator": "lt",
                    "reference_value": 1.0,
                    "reference_channel": None,
                    "unit": "ratio",
                },
            ],
            "parse_status": "OK",
            "parse_reason": None,
        },
    },
    {
        "sentence": "Chest periodicity strength equals 0.22, chest peak amplitude equals 0.91 raw units, and the chest-to-hand timing lag equals -4.0 ms.",
        "channels": ["hand_accel", "chest_accel"],
        "fs": 100.0,
        "output": {
            "connective": "AND",
            "predicates": [
                {
                    "measurement": "periodicity_strength",
                    "channel_a": "chest_accel",
                    "channel_b": None,
                    "comparator": "eq",
                    "reference_value": 0.22,
                    "reference_channel": None,
                    "unit": "score_0_1",
                },
                {
                    "measurement": "peak_amplitude",
                    "channel_a": "chest_accel",
                    "channel_b": None,
                    "comparator": "eq",
                    "reference_value": 0.91,
                    "reference_channel": None,
                    "unit": "raw",
                },
                {
                    "measurement": "cross_channel_lag_ms",
                    "channel_a": "chest_accel",
                    "channel_b": "hand_accel",
                    "comparator": "eq",
                    "reference_value": -4.0,
                    "reference_channel": None,
                    "unit": "ms",
                },
            ],
            "parse_status": "OK",
            "parse_reason": None,
        },
    },
]

PROMPTS = {
    "v1": {
        "id": "p2r_lm_local_v1",
        "system": SYSTEM_V1,
        "fewshot": FEWSHOT_V1,
    },
    "v2": {
        "id": "p2r_lm_local_v2",
        "system": SYSTEM_V2,
        "fewshot": FEWSHOT_V2,
    },
    "v3a": {
        "id": "p2r_lm_local_v3a",
        "system": SYSTEM_V3A,
        "fewshot": FEWSHOT_V3A,
    },
    "v3b": {
        "id": "p2r_lm_local_v3b",
        "system": SYSTEM_V3B,
        "fewshot": FEWSHOT_V3B,
    },
}


def prompt_bundle(prompt_version: str = "v1") -> dict[str, Any]:
    if prompt_version not in PROMPTS:
        raise KeyError(prompt_version)
    return PROMPTS[prompt_version]


def prompt_hash(prompt_version: str = "v1") -> str:
    bundle = prompt_bundle(prompt_version)
    payload = json.dumps(
        {"id": bundle["id"], "system": bundle["system"], "fewshot": bundle["fewshot"], "schema": schema_catalog()},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_inference_payload(row: dict[str, Any]) -> dict[str, Any]:
    """Deployment-legal extractor input. Gold/split/family keys are dropped."""
    inf = row.get("inference") if isinstance(row.get("inference"), dict) else {}
    surface = inf.get("surface_text") or row.get("surface_text")
    channels = inf.get("available_channels") or row.get("available_channels")
    fs = inf.get("fs") if inf.get("fs") is not None else row.get("fs")
    payload = {
        "surface_text": surface,
        "available_channels": list(channels or []),
        "fs": fs,
    }
    assert_no_leakage(payload)
    extra = set(payload) & set(FORBIDDEN_PAYLOAD_KEYS)
    if extra:
        raise ValueError(f"payload leaked keys: {sorted(extra)}")
    return payload


def _user_block(sentence: str, channels: list[str], fs: Any) -> str:
    catalog = json.dumps(schema_catalog(), indent=2)
    return (
        f"Available channels: {', '.join(channels)}\n"
        f"Sampling rate fs_Hz: {fs}\n"
        f"Sentence: {sentence}\n\n"
        f"Frozen executable schema catalog:\n{catalog}\n"
    )


def build_chat_messages(row: dict[str, Any], prompt_version: str = "v1") -> list[dict[str, str]]:
    bundle = prompt_bundle(prompt_version)
    payload = build_inference_payload(row)
    messages = [{"role": "system", "content": bundle["system"]}]
    for ex in bundle["fewshot"]:
        messages.append({
            "role": "user",
            "content": _user_block(ex["sentence"], ex["channels"], ex["fs"]),
        })
        messages.append({
            "role": "assistant",
            "content": json.dumps(ex["output"], ensure_ascii=False),
        })
    messages.append({
        "role": "user",
        "content": _user_block(payload["surface_text"], payload["available_channels"], payload["fs"]),
    })
    return messages


def payload_leakage_audit(row: dict[str, Any], prompt_version: str = "v1") -> dict[str, Any]:
    payload = build_inference_payload(row)
    messages = build_chat_messages(row, prompt_version=prompt_version)
    blob = " ".join(m["content"] for m in messages)
    present = [k for k in FORBIDDEN_PAYLOAD_KEYS if k in payload]
    canaries = []
    for key in FORBIDDEN_PAYLOAD_KEYS:
        val = row.get(key)
        if val is None or key in ("surface_text", "available_channels", "fs"):
            continue
        token = str(val)
        if token and token in blob and token not in payload["surface_text"]:
            canaries.append(key)
    return {
        "pass": not present and not canaries,
        "forbidden_keys_present": present,
        "canaries_in_prompt": canaries,
        "payload_keys": sorted(payload),
        "prompt_hash": prompt_hash(prompt_version),
        "schema_hash": schema_hash(),
    }


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _extract_json_object(raw: str) -> Optional[dict]:
    if not raw:
        return None
    text = _THINK_RE.sub("", raw).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _as_float(v: Any) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def program_from_obj(obj: dict, available: list[str]) -> ClaimProgram:
    leak = {k: obj[k] for k in obj if k in FORBIDDEN_INFERENCE_KEYS}
    if leak:
        raise ValueError(f"model object carried forbidden keys {sorted(leak)}")
    status = str(obj.get("parse_status") or "OK").upper()
    if status in ("AMBIGUOUS", "UNPARSEABLE"):
        return ClaimProgram("SINGLE", [], parse_status=status, parse_reason=obj.get("parse_reason"))
    if status != "OK":
        return ClaimProgram("SINGLE", [], parse_status="UNPARSEABLE", parse_reason=f"bad_parse_status:{status}")
    conn = obj.get("connective") or "SINGLE"
    if conn not in CONNECTIVES:
        return ClaimProgram("SINGLE", [], parse_status="UNPARSEABLE", parse_reason="bad_connective")
    preds: list[Predicate] = []
    for raw in obj.get("predicates") or []:
        if not isinstance(raw, dict):
            return ClaimProgram(conn, [], parse_status="UNPARSEABLE", parse_reason="bad_predicate")
        meas = raw.get("measurement") or raw.get("op")
        if meas not in MEASUREMENTS:
            return ClaimProgram(conn, [], parse_status="UNPARSEABLE", parse_reason=f"unknown_measurement:{meas}")
        pred = Predicate(
            measurement=meas,
            channel_a=str(raw.get("channel_a") or (raw.get("channels") or [""])[0]),
            comparator=str(raw.get("comparator") or raw.get("relation") or "eq"),
            channel_b=raw.get("channel_b") or (None if not raw.get("channels") or len(raw.get("channels") or []) < 2 else raw["channels"][1]),
            reference_value=_as_float(raw.get("reference_value", raw.get("asserted_value", raw.get("threshold")))),
            reference_channel=raw.get("reference_channel") or raw.get("compare_channel"),
            unit=raw.get("unit"),
        )
        preds.append(pred)
    return validate_program(ClaimProgram(conn, preds, parse_status="OK"), available)


def parse_model_output(raw: str, available: list[str]) -> dict[str, Any]:
    obj = _extract_json_object(raw)
    if obj is None:
        return {
            "program": ClaimProgram("SINGLE", [], parse_status="UNPARSEABLE", parse_reason="non_json"),
            "parsed": None,
            "malformed": True,
        }
    try:
        prog = program_from_obj(obj, available)
    except Exception as exc:  # noqa: BLE001
        return {
            "program": ClaimProgram("SINGLE", [], parse_status="UNPARSEABLE", parse_reason=f"parse_error:{exc}"),
            "parsed": obj,
            "malformed": True,
        }
    return {"program": prog, "parsed": obj, "malformed": False}


def _host_url(host: Optional[str] = None) -> str:
    h = host or DEFAULT_HOST
    if not h.startswith("http"):
        h = "http://" + h
    return h.rstrip("/")


def ollama_reachable(host: Optional[str] = None, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(_host_url(host) + "/api/tags", timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def list_ollama_models(host: Optional[str] = None) -> list[dict[str, Any]]:
    try:
        with urllib.request.urlopen(_host_url(host) + "/api/tags", timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return list(data.get("models") or [])
    except Exception:
        return []


def _post_chat(host: Optional[str], body: dict[str, Any]) -> tuple[dict[str, Any], Optional[str]]:
    req = urllib.request.Request(
        _host_url(host) + "/api/chat",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
            return json.loads(resp.read().decode("utf-8")), None
    except urllib.error.HTTPError as exc:
        return {"error": exc.read().decode("utf-8", errors="replace"), "http_status": exc.code}, f"http_{exc.code}"
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}, "request_failed"


def chat_ollama(
    model: str,
    messages: list[dict[str, str]],
    host: Optional[str] = None,
    seed: int = SEED,
    think: bool = False,
) -> dict[str, Any]:
    options = {"temperature": TEMPERATURE, "seed": seed}
    attempts = [
        {"model": model, "messages": messages, "stream": False, "think": think, "format": JSON_SCHEMA, "options": options},
        {"model": model, "messages": messages, "stream": False, "format": JSON_SCHEMA, "options": options},
        {"model": model, "messages": messages, "stream": False, "format": "json", "options": options},
    ]
    t0 = time.perf_counter()
    payload: dict[str, Any] = {}
    err: Optional[str] = "request_failed"
    for body in attempts:
        payload, err = _post_chat(host, body)
        if err is None:
            break
    dt = time.perf_counter() - t0
    msg = (payload.get("message") or {}) if isinstance(payload, dict) else {}
    raw = msg.get("content") or ""
    return {
        "raw": raw,
        "response": payload,
        "latency_s": dt,
        "error": err,
        "eval_count": (payload.get("eval_count") if isinstance(payload, dict) else None),
        "prompt_eval_count": (payload.get("prompt_eval_count") if isinstance(payload, dict) else None),
    }


def cache_key(model: str, prompt_version: str, sentence: str, channels: list[str], fs: Any, seed: int) -> str:
    blob = json.dumps(
        {
            "model": model,
            "prompt": prompt_hash(prompt_version),
            "schema": schema_hash(),
            "sentence": sentence,
            "channels": channels,
            "fs": fs,
            "seed": seed,
            "temperature": TEMPERATURE,
        },
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def extract_ollama(
    sentence: str,
    available_channels: list[str],
    fs: Optional[float] = None,
    model: str = "",
    prompt_version: str = "v1",
    host: Optional[str] = None,
    seed: int = SEED,
    use_cache: bool = True,
    cache_suffix: str = "",
) -> tuple[ClaimProgram, dict[str, Any]]:
    if not model:
        raise ValueError("model id required")
    row = {"surface_text": sentence, "available_channels": available_channels, "fs": fs}
    messages = build_chat_messages(row, prompt_version=prompt_version)
    key = cache_key(model, prompt_version, sentence, available_channels, fs, seed) + cache_suffix
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", model)
    cdir = CACHE_ROOT / safe
    cpath = cdir / f"{key}.json"
    if use_cache and cpath.exists():
        rec = json.loads(cpath.read_text(encoding="utf-8"))
        parsed = parse_model_output(rec.get("raw") or "", available_channels)
        rec["parsed_output"] = parsed["parsed"]
        rec["malformed"] = parsed["malformed"]
        rec["cache_hit"] = True
        return parsed["program"], rec

    last = None
    raw = ""
    retry_reason = None
    for attempt in range(RETRIES + 1):
        last = chat_ollama(model, messages, host=host, seed=seed, think=False)
        raw = last.get("raw") or ""
        parsed = parse_model_output(raw, available_channels)
        if not parsed["malformed"] or attempt == RETRIES:
            break
        retry_reason = parsed["program"].parse_reason
    parsed = parse_model_output(raw, available_channels)
    rec = {
        "provider": "ollama",
        "model": model,
        "prompt_version": prompt_version,
        "prompt_hash": prompt_hash(prompt_version),
        "schema_hash": schema_hash(),
        "temperature": TEMPERATURE,
        "seed": seed,
        "think": False,
        "retries": RETRIES,
        "retry_reason": retry_reason,
        "timeout_s": TIMEOUT_S,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sentence": sentence,
        "channels": available_channels,
        "fs": fs,
        "raw": raw,
        "parsed_output": parsed["parsed"],
        "malformed": parsed["malformed"],
        "latency_s": last.get("latency_s") if last else None,
        "error": last.get("error") if last else None,
        "eval_count": last.get("eval_count") if last else None,
        "prompt_eval_count": last.get("prompt_eval_count") if last else None,
        "cache_hit": False,
    }
    cdir.mkdir(parents=True, exist_ok=True)
    cpath.write_text(json.dumps(rec, indent=1, ensure_ascii=False), encoding="utf-8")
    return parsed["program"], rec


def make_extractor(model: str, prompt_version: str = "v1", host: Optional[str] = None, seed: int = SEED):
    def _extract(sentence: str, channels: list[str], fs: Optional[float] = None) -> ClaimProgram:
        prog, _meta = extract_ollama(
            sentence, channels, fs, model=model, prompt_version=prompt_version, host=host, seed=seed
        )
        return prog

    return _extract


def smoke_test(model: str, host: Optional[str] = None) -> dict[str, Any]:
    messages = [{"role": "user", "content": "Reply with the single word OK."}]
    out = chat_ollama(model, messages, host=host, seed=SEED, think=False)
    raw = (out.get("raw") or "").strip()
    return {
        "model": model,
        "ok": bool(raw),
        "nonempty": bool(raw),
        "raw_preview": raw[:120],
        "latency_s": out.get("latency_s"),
        "error": out.get("error"),
    }
