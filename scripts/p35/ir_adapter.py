"""IR interface only. Does not modify prompt v2. Not a v4 reasoning prompt."""
from __future__ import annotations

import json
from typing import Any

from p2r.ollama_adapter import _extract_json_object

from .compiler import compile_ir
from .config import IR_PROMPT_ID, PRIMARY_MODEL, SEED
from .ir_schema import ir_catalog, validate_ir
from .llm import cached_chat

SYSTEM_IR = """You convert one wearable-IMU sentence into a Semantic IR JSON object.
You do NOT see a waveform. You do NOT compute DSP. You do NOT assign a truth verdict.
You do NOT emit an executable ClaimProgram. You do NOT rewrite implications into OR.

Return ONLY JSON with a root "type" of:
  ATOMIC | COMPOSITE | CONDITIONAL | UNSUPPORTED_LANGUAGE | AMBIGUOUS_LANGUAGE | MISSING_REQUIRED_INFORMATION

ATOMIC fields: primitive, operator, threshold, unit, channel
  lag uses source_channel and target_channel instead of channel
  operator is EQ | GREATER_THAN | LESS_THAN | SIMILAR | DIFFERENT
  primitive is one of the eight frozen IMU measurements
  never set reference_channel on a lag ATOMIC

COMPOSITE: operator AND|OR and children[]
CONDITIONAL: antecedent and consequent (each typically ATOMIC). Do not convert to OR.

Use UNSUPPORTED_LANGUAGE for heart rate, entropy, XOR, or any non-ontology measurement.
Use AMBIGUOUS_LANGUAGE when a channel or comparator is not unique.
Use MISSING_REQUIRED_INFORMATION when a required channel or number is absent or sampling rate is stated unknown.
"""

# Invented interface examples. Not copied from evaluation sets.
_FEWSHOT = [
    {
        "sentence": "Chest RMS equals 0.42 raw units.",
        "channels": ["hand_accel", "chest_accel"],
        "fs": 100.0,
        "output": {
            "type": "ATOMIC",
            "primitive": "rms_amplitude",
            "operator": "EQ",
            "threshold": 0.42,
            "unit": "raw",
            "channel": "chest_accel",
        },
    },
    {
        "sentence": "If hand range exceeds 2.0 raw units then hand peak exceeds 1.0 raw units.",
        "channels": ["hand_accel", "chest_accel"],
        "fs": 100.0,
        "output": {
            "type": "CONDITIONAL",
            "antecedent": {
                "type": "ATOMIC",
                "primitive": "signal_range",
                "operator": "GREATER_THAN",
                "threshold": 2.0,
                "unit": "raw",
                "channel": "hand_accel",
            },
            "consequent": {
                "type": "ATOMIC",
                "primitive": "peak_amplitude",
                "operator": "GREATER_THAN",
                "threshold": 1.0,
                "unit": "raw",
                "channel": "hand_accel",
            },
        },
    },
    {
        "sentence": "Heart rate on chest is 72 bpm.",
        "channels": ["hand_accel", "chest_accel"],
        "fs": 100.0,
        "output": {"type": "UNSUPPORTED_LANGUAGE", "reason": "unsupported_measurement"},
    },
]


def _user(sentence: str, channels: list[str], fs: Any) -> str:
    return (
        f"Available channels: {', '.join(channels)}\n"
        f"Sampling rate fs_Hz: {fs}\n"
        f"Sentence: {sentence}\n\n"
        f"IR catalog:\n{json.dumps(ir_catalog(), indent=2)}\n"
    )


def extract_ir(sentence: str, channels: list[str], fs, model: str = PRIMARY_MODEL) -> dict:
    messages = [{"role": "system", "content": SYSTEM_IR}]
    for ex in _FEWSHOT:
        messages.append({"role": "user", "content": _user(ex["sentence"], ex["channels"], ex["fs"])})
        messages.append({"role": "assistant", "content": json.dumps(ex["output"], ensure_ascii=False)})
    messages.append({"role": "user", "content": _user(sentence, channels, fs)})
    rec = cached_chat(IR_PROMPT_ID, model, messages, seed=SEED, temperature=0.0, fmt="json")
    raw = rec.get("raw") or ""
    obj = _extract_json_object(raw)
    if obj is None:
        return {"type": "UNSUPPORTED_LANGUAGE", "reason": "non_json", "raw": raw}
    ir, err = validate_ir(obj)
    if err or ir is None:
        return {"type": "UNSUPPORTED_LANGUAGE", "reason": err or "invalid_ir", "raw_obj": obj}
    return ir


def extract_program_via_ir(sentence: str, channels: list[str], fs, model: str = PRIMARY_MODEL):
    ir = extract_ir(sentence, channels, fs, model=model)
    return compile_ir(ir, channels), ir


def ir_prompt_record() -> dict:
    return {
        "id": IR_PROMPT_ID,
        "role": "interface_change_only",
        "not_v4": True,
        "historical_v2_untouched": True,
        "system": SYSTEM_IR,
        "fewshot_n": len(_FEWSHOT),
    }
