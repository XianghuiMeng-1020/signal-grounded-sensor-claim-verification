"""Exhaustive Strong Kleene truth tables."""
from scripts.p2r.kleene import (
    FALSE,
    TRUE,
    UNKNOWN,
    compose,
    kleene_and,
    kleene_implies,
    kleene_not,
    kleene_or,
    verdict_from_tv,
)


def test_not_table():
    assert kleene_not(TRUE) == FALSE
    assert kleene_not(FALSE) == TRUE
    assert kleene_not(UNKNOWN) == UNKNOWN


def test_and_table():
    rows = {
        (TRUE, TRUE): TRUE,
        (TRUE, FALSE): FALSE,
        (TRUE, UNKNOWN): UNKNOWN,
        (FALSE, TRUE): FALSE,
        (FALSE, FALSE): FALSE,
        (FALSE, UNKNOWN): FALSE,
        (UNKNOWN, TRUE): UNKNOWN,
        (UNKNOWN, FALSE): FALSE,
        (UNKNOWN, UNKNOWN): UNKNOWN,
    }
    for (a, b), y in rows.items():
        assert kleene_and((a, b)) == y, (a, b, y)


def test_or_table():
    rows = {
        (TRUE, TRUE): TRUE,
        (TRUE, FALSE): TRUE,
        (TRUE, UNKNOWN): TRUE,
        (FALSE, TRUE): TRUE,
        (FALSE, FALSE): FALSE,
        (FALSE, UNKNOWN): UNKNOWN,
        (UNKNOWN, TRUE): TRUE,
        (UNKNOWN, FALSE): UNKNOWN,
        (UNKNOWN, UNKNOWN): UNKNOWN,
    }
    for (a, b), y in rows.items():
        assert kleene_or((a, b)) == y, (a, b, y)


def test_implies_is_kleene_or_not():
    for a in (TRUE, FALSE, UNKNOWN):
        for b in (TRUE, FALSE, UNKNOWN):
            assert kleene_implies(a, b) == kleene_or((kleene_not(a), b))


def test_or_unknown_does_not_kill_true():
    assert compose("OR", [TRUE, UNKNOWN]) == TRUE
    assert verdict_from_tv(compose("OR", [TRUE, UNKNOWN])) == "SUPPORTED"


def test_and_unknown_is_not_false():
    assert compose("AND", [TRUE, UNKNOWN]) == UNKNOWN
    assert verdict_from_tv(compose("AND", [TRUE, UNKNOWN])) == "UNVERIFIABLE"


def test_if_then_false_antecedent_is_true():
    assert compose("IF_THEN", [FALSE, UNKNOWN]) == TRUE


def test_unknown_connective_and_empty():
    assert compose("XOR", [TRUE]) == UNKNOWN
    assert compose("SINGLE", []) == UNKNOWN
