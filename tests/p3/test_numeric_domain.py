from scripts.p2r.schema import Predicate
from scripts.p3.numeric_domain import vs_value_in_domain


def test_negative_spectral_value_is_invalid():
    p = Predicate("spectral_energy_ratio_low", "hand_accel", "eq", reference_value=-0.5, unit="fraction")
    ok, reason = vs_value_in_domain(p)
    assert not ok
    assert "below" in reason


def test_negative_spectral_threshold_is_valid():
    p = Predicate("spectral_energy_ratio_low", "hand_accel", "gt", reference_value=-0.1, unit="fraction")
    ok, _ = vs_value_in_domain(p)
    assert ok


def test_in_range_fraction_ok():
    p = Predicate("spectral_energy_ratio_low", "hand_accel", "eq", reference_value=0.4, unit="fraction")
    ok, _ = vs_value_in_domain(p)
    assert ok
