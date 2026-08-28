"""Model-agnostic schema extractor.

Proposed front end: LLM restricted to ClaimProgram JSON.
B6 deterministic parser is a BASELINE only and is never labeled as the proposed system.

If no legitimate model is configured, extract_llm() returns parse_status=UNAVAILABLE
and does not fabricate programs.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .schema import MEASUREMENTS, ClaimProgram, schema_catalog, schema_hash
from .validator import from_legacy

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "results" / "p2r" / "llm_cache"
PROMPT_PATH = ROOT / "scripts" / "p2r" / "extraction_prompt.txt"


def prompt_text() -> str:
    cat = json.dumps(schema_catalog(), indent=2)
    return f"""You extract an executable IMU claim schema. You do NOT judge truth.
You do NOT compute signal features. You do NOT invent missing channels.

Return ONLY JSON with keys:
  connective: SINGLE | AND | OR | IF_THEN
  predicates: list of objects with keys
    measurement: one of {list(MEASUREMENTS)}
    channel_a: one of the available channel names (or a resolvable alias)
    channel_b: required only for cross_channel_lag_ms
    comparator: eq | gt | lt | similar | different
    reference_value: number when comparing to a numeric reference (eq/gt/lt vs value)
    reference_channel: channel name when comparing two channels
    unit: optional Hz|ms|s|raw|ratio|fraction|percent|score_0_1
  parse_status: OK | AMBIGUOUS | UNPARSEABLE
  parse_reason: string or null

eq = the sentence asserts a specific numeric value for the measurement.
gt/lt + reference_value = vs a numeric threshold.
gt/lt/similar/different + reference_channel = vs another channel.

If the sentence is not an executable IMU measurement (physiology, emotion, battery,
nested logic beyond AND/OR/IF_THEN with <=3 predicates, missing/ambiguous channel),
return parse_status UNPARSEABLE or AMBIGUOUS and predicates [].

Available channels: {{channels}}
Sampling rate fs_Hz: {{fs}}
Sentence: {{sentence}}

Schema catalog:
{cat}
"""


def prompt_hash() -> str:
    return hashlib.sha256(prompt_text().encode("utf-8")).hexdigest()


def _env_has_llm() -> tuple[bool, Optional[str], Optional[str]]:
    if os.environ.get("OPENROUTER_API_KEY"):
        return True, "openrouter", os.environ.get("P2R_MODEL", "qwen/qwen-2.5-72b-instruct")
    if os.environ.get("OPENAI_API_KEY"):
        return True, "openai", os.environ.get("P2R_MODEL", "gpt-4o-mini")
    from .ollama_adapter import list_ollama_models, ollama_reachable

    if ollama_reachable():
        requested = os.environ.get("P2R_MODEL") or os.environ.get("OLLAMA_MODEL")
        names = [m.get("name") for m in list_ollama_models() if m.get("name")]
        if requested and requested in names:
            return True, "ollama", requested
        if names:
            return True, "ollama", names[0]
    return False, None, None


def llm_status() -> dict[str, Any]:
    ok, provider, model = _env_has_llm()
    return {
        "available": ok,
        "provider": provider,
        "model": model,
        "temperature": 0,
        "schema_hash": schema_hash(),
        "prompt_hash": prompt_hash(),
        "date": datetime.now(timezone.utc).isoformat(),
    }


def extract_llm(sentence: str, available_channels: list[str], fs: Optional[float] = None) -> ClaimProgram:
    """Proposed extractor. Fabricates nothing when the model is absent."""
    ok, provider, model = _env_has_llm()
    if not ok:
        return ClaimProgram(
            "SINGLE",
            [],
            parse_status="UNAVAILABLE",
            parse_reason="no_legitimate_llm_configured",
        )
    if provider == "ollama":
        from .ollama_adapter import extract_ollama

        prog, _meta = extract_ollama(sentence, available_channels, fs, model=model or "")
        return prog
    # Live cloud path — only reached when a key exists. Cache raw output.
    try:
        from openai import OpenAI
    except Exception as exc:  # noqa: BLE001
        return ClaimProgram("SINGLE", [], parse_status="UNAVAILABLE", parse_reason=f"sdk:{exc}")
    base = "https://openrouter.ai/api/v1" if provider == "openrouter" else None
    key = os.environ["OPENROUTER_API_KEY"] if provider == "openrouter" else os.environ["OPENAI_API_KEY"]
    client = OpenAI(api_key=key, base_url=base) if base else OpenAI(api_key=key)
    user = prompt_text().format(channels=", ".join(available_channels), fs=fs, sentence=sentence)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": user}],
        temperature=0,
        max_tokens=800,
    )
    raw = resp.choices[0].message.content or ""
    CACHE.mkdir(parents=True, exist_ok=True)
    rec = {
        "provider": provider,
        "model": model,
        "temperature": 0,
        "prompt_hash": prompt_hash(),
        "schema_hash": schema_hash(),
        "date": datetime.now(timezone.utc).isoformat(),
        "sentence": sentence,
        "channels": available_channels,
        "fs": fs,
        "raw": raw,
    }
    digest = hashlib.sha256((sentence + "|" + ",".join(available_channels)).encode()).hexdigest()[:16]
    (CACHE / f"{digest}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    try:
        obj = json.loads(raw[raw.find("{") : raw.rfind("}") + 1])
    except Exception:
        return ClaimProgram("SINGLE", [], parse_status="UNPARSEABLE", parse_reason="non_json")
    return from_legacy(obj, available_channels)


def extract_b6_baseline(sentence: str, available_channels: list[str], fs: Optional[float] = None) -> ClaimProgram:
    """Deterministic baseline. Not the proposed system."""
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    from p2.extractor_deterministic import extract as b6

    raw = b6(sentence, available_channels, fs)
    return from_legacy(raw, available_channels)
