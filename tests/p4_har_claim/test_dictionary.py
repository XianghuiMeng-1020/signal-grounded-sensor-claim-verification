"""Dictionary is frozen, manual, and scale-free. No LLM, no fitted amplitudes."""
from p4_har_claim.dictionary import (
    ACTIVITY_FAMILY,
    DICTIONARY_ID,
    FAMILY_CLAIMS,
    dictionary_sha256,
    family_of,
    make_program,
    mappable,
)


def test_dictionary_id_and_hash_are_stable():
    assert DICTIONARY_ID == "p4_har_claim_dictionary_v1"
    assert len(dictionary_sha256()) == 64
    assert dictionary_sha256() == dictionary_sha256()


def test_only_scale_free_primitives():
    allowed = {"dominant_frequency", "spectral_energy_ratio_low"}
    assert {c["measurement"] for c in FAMILY_CLAIMS.values()} <= allowed


def test_families_are_exactly_static_walk_run():
    assert set(FAMILY_CLAIMS) == {"static", "walk", "run"}


def test_mapped_codes_cover_three_corpora():
    datasets = {d for d, _ in ACTIVITY_FAMILY}
    assert datasets == {"PAMAP2", "WISDM", "MHEALTH"}
    assert "HARTH" not in datasets


def test_unmapped_activities_are_not_mappable():
    assert not mappable("PAMAP2", 17)  # ironing
    assert not mappable("WISDM", "F")  # typing
    assert not mappable("MHEALTH", 7)  # arm elevation
    assert family_of("HARTH", "walking") is None


def test_program_is_single_named_channel_claim():
    prog = make_program("walk", "hand_accel")
    assert prog.connective == "SINGLE"
    assert prog.parse_status == "OK"
    assert prog.predicates[0].measurement == "dominant_frequency"
    assert prog.predicates[0].reference_value == 0.8
    assert prog.predicates[0].channel_a == "hand_accel"
