"""Ollama chat without the compiler JSON schema (used for generation and agents)."""
from __future__ import annotations

import hashlib
import json
import time
import urllib.request
from pathlib import Path
from typing import Any, Optional

from .config import CACHE, SEED

TIMEOUT_S = 180


def chat_plain(model: str, messages: list[dict[str, str]], seed: int = SEED, temperature: float = 0.0, fmt=None) -> dict[str, Any]:
    body = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": False,
        "options": {"temperature": temperature, "seed": seed},
    }
    if fmt is not None:
        body["format"] = fmt
    req = urllib.request.Request(
        "http://127.0.0.1:11434/api/chat",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        err = None
    except Exception as exc:  # noqa: BLE001
        payload, err = {"error": str(exc)}, str(exc)
    raw = ((payload.get("message") or {}).get("content") if isinstance(payload, dict) else "") or ""
    return {"raw": raw, "latency_s": time.perf_counter() - t0, "error": err, "response": payload}


def cached_chat(name: str, model: str, messages, seed=SEED, temperature=0.0, fmt=None) -> dict[str, Any]:
    key = hashlib.sha256(json.dumps({"n": name, "m": model, "msg": messages, "s": seed, "t": temperature}, sort_keys=True).encode()).hexdigest()
    path = CACHE / "llm" / f"{key}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        rec = json.loads(path.read_text(encoding="utf-8"))
        rec["cache_hit"] = True
        return rec
    rec = chat_plain(model, messages, seed=seed, temperature=temperature, fmt=fmt)
    rec["cache_hit"] = False
    path.write_text(json.dumps(rec, indent=1, ensure_ascii=False), encoding="utf-8")
    return rec
