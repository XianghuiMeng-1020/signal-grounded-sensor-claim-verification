"""Deterministic source-grounded channel resolution for SCV V2."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Optional

from .ontology import (
    CHANNEL_ALIASES,
    REGISTERED_CHANNELS,
    UNSUPPORTED_CHANNEL_NAMES,
)


@dataclass
class Mention:
    span: str
    start: int
    end: int
    candidates: tuple[str, ...]
    status: str  # VALID | AMBIGUOUS | UNSUPPORTED | UNRESOLVED


def _iter_alias_keys() -> list[str]:
    keys = list(CHANNEL_ALIASES.keys()) + list(REGISTERED_CHANNELS) + list(UNSUPPORTED_CHANNEL_NAMES)
    return sorted(set(keys), key=len, reverse=True)


def find_channel_mentions(text: str) -> list[Mention]:
    """Recover channel-bearing spans from source text. Longest match wins."""
    if not text:
        return []
    low = text.lower()
    taken = [False] * len(low)
    mentions: list[Mention] = []
    for key in _iter_alias_keys():
        pat = re.compile(r"(?<![\w])" + re.escape(key.lower()) + r"(?![\w])")
        for m in pat.finditer(low):
            if any(taken[i] for i in range(m.start(), m.end())):
                continue
            for i in range(m.start(), m.end()):
                taken[i] = True
            span = text[m.start() : m.end()]
            if key.lower() in {x.lower() for x in UNSUPPORTED_CHANNEL_NAMES}:
                mentions.append(Mention(span, m.start(), m.end(), (), "UNSUPPORTED"))
                continue
            cands = CHANNEL_ALIASES.get(key.lower())
            if cands is None and key in REGISTERED_CHANNELS:
                cands = (key,)
            if not cands:
                mentions.append(Mention(span, m.start(), m.end(), (), "UNRESOLVED"))
                continue
            uniq = tuple(dict.fromkeys(cands))
            status = "VALID" if len(uniq) == 1 else "AMBIGUOUS"
            mentions.append(Mention(span, m.start(), m.end(), uniq, status))
    mentions.sort(key=lambda x: x.start)
    return mentions


def split_clauses(text: str, connective: str) -> list[str]:
    if not text:
        return [""]
    if connective == "IF_THEN":
        parts = re.split(r"\bTHEN\b", text, maxsplit=1, flags=re.I)
        return parts if len(parts) == 2 else [text]
    if connective == "AND":
        parts = re.split(r"\bALSO\b|\bAND ALSO\b", text, maxsplit=1, flags=re.I)
        return parts if len(parts) == 2 else [text]
    if connective == "OR":
        parts = re.split(r"\bELSE\b|\bOR ELSE\b", text, maxsplit=1, flags=re.I)
        return parts if len(parts) == 2 else [text]
    return [text]


def resolve_mention_to_available(mention: Mention, available: list[str]) -> tuple[Optional[str], str]:
    """Map a source mention onto an available channel without substitution."""
    if mention.status == "UNSUPPORTED":
        return None, "UNSUPPORTED"
    if mention.status == "AMBIGUOUS":
        return None, "AMBIGUOUS"
    if mention.status != "VALID" or len(mention.candidates) != 1:
        return None, "UNRESOLVED"
    cid = mention.candidates[0]
    if cid not in available:
        return None, "UNRESOLVED"
    return cid, "VALID"


def mentions_as_dicts(mentions: list[Mention]) -> list[dict]:
    return [asdict(m) for m in mentions]
