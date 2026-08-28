from __future__ import annotations

import hashlib
import json
from pathlib import Path

from p2r.schema import ClaimProgram, Predicate

from .config import RESULTS
from .guard import refuse_path


def write_json(name: str, obj) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / name
    refuse_path(path)
    path.write_text(json.dumps(obj, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
    return path


def read_json(path: Path):
    refuse_path(path)
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    refuse_path(path)
    return sha256_bytes(path.read_bytes())


def sha256_obj(obj) -> str:
    return sha256_bytes(json.dumps(obj, sort_keys=True, default=str).encode())


def program_from_dict(d: dict) -> ClaimProgram:
    preds = []
    for raw in d.get("predicates") or []:
        preds.append(Predicate(
            measurement=raw.get("measurement"),
            channel_a=raw.get("channel_a"),
            comparator=raw.get("comparator"),
            channel_b=raw.get("channel_b"),
            reference_value=raw.get("reference_value"),
            reference_channel=raw.get("reference_channel"),
            unit=raw.get("unit"),
        ))
    return ClaimProgram(
        d.get("connective") or "SINGLE",
        preds,
        parse_status=d.get("parse_status") or "OK",
        parse_reason=d.get("parse_reason"),
    )
