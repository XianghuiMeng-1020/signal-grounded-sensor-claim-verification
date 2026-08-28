"""HAR→claim pool must not include the sealed holdout."""
import pytest

from p4_har_claim.construct import assert_no_holdout


def test_assert_no_holdout_rejects_sealed_split():
    with pytest.raises(RuntimeError, match="holdout"):
        assert_no_holdout([{"window_id": "x", "split": "final_sealed_holdout"}])


def test_assert_no_holdout_allows_open_splits():
    assert_no_holdout([
        {"window_id": "a", "split": "development"},
        {"window_id": "b", "split": "challenge"},
    ])
