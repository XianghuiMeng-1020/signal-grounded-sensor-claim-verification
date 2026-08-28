"""Strong Kleene three-valued logic. No LLM. No DSP. No raw language.

Values: TRUE, FALSE, UNKNOWN
IF_THEN is formal material implication: NOT antecedent OR consequent
under the same Kleene tables. This is not a theory of natural-language conditionals.
"""
from __future__ import annotations

from typing import Iterable, Optional

TRUE = "TRUE"
FALSE = "FALSE"
UNKNOWN = "UNKNOWN"
TV = (TRUE, FALSE, UNKNOWN)


def kleene_not(x: str) -> str:
    if x == TRUE:
        return FALSE
    if x == FALSE:
        return TRUE
    if x == UNKNOWN:
        return UNKNOWN
    raise ValueError(x)


def kleene_and(values: Iterable[str]) -> str:
    vals = list(values)
    if not vals:
        return UNKNOWN
    if any(v == FALSE for v in vals):
        return FALSE
    if all(v == TRUE for v in vals):
        return TRUE
    return UNKNOWN


def kleene_or(values: Iterable[str]) -> str:
    vals = list(values)
    if not vals:
        return UNKNOWN
    if any(v == TRUE for v in vals):
        return TRUE
    if all(v == FALSE for v in vals):
        return FALSE
    return UNKNOWN


def kleene_implies(antecedent: str, consequent: str) -> str:
    return kleene_or((kleene_not(antecedent), consequent))


def compose(connective: str, truths: list[str]) -> str:
    if connective == "SINGLE":
        return truths[0] if truths else UNKNOWN
    if connective == "AND":
        return kleene_and(truths)
    if connective == "OR":
        return kleene_or(truths)
    if connective == "IF_THEN":
        if len(truths) != 2:
            return UNKNOWN
        return kleene_implies(truths[0], truths[1])
    return UNKNOWN


def verdict_from_tv(tv: str) -> str:
    if tv == TRUE:
        return "SUPPORTED"
    if tv == FALSE:
        return "CONTRADICTED"
    return "UNVERIFIABLE"


def bool_or_none_to_tv(x: Optional[bool]) -> str:
    if x is True:
        return TRUE
    if x is False:
        return FALSE
    return UNKNOWN
