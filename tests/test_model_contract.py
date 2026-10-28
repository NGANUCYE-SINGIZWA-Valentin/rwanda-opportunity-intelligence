"""Small guardrail checks for the model's target/features contract."""
from backend.neet_model import FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES


def test_feature_groups_match_features():
    assert set(CATEGORICAL_FEATURES).union(NUMERIC_FEATURES) == set(FEATURES)
    assert set(CATEGORICAL_FEATURES).isdisjoint(NUMERIC_FEATURES)


def test_no_direct_target_or_current_status_leakage():
    forbidden = {
        "youngs", "neetyoung", "neetyouth", "young", "youth", "status1",
        "employed16", "UR1", "LFPR", "C01", "C02", "TVT", "TVT2",
    }
    assert forbidden.isdisjoint(FEATURES)
