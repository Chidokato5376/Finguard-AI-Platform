"""Unit tests cho src/data/synthetic_generator.py — trọng tâm là
exclude_patterns / generate_concept_drift_split (dùng cho KPI concept
drift, Mục 11).
"""

import pytest

from src.data.synthetic_generator import (
    FRAUD_PATTERN_NAMES,
    VNCalibrationConfig,
    generate_concept_drift_split,
    generate_synthetic_vn_dataset,
)


def _small_config() -> VNCalibrationConfig:
    return VNCalibrationConfig(num_normal_transactions=200, fraud_rate_assumed=0.03, random_seed=1)


def test_exclude_patterns_removes_pattern() -> None:
    df = generate_synthetic_vn_dataset(_small_config(), exclude_patterns=("account_takeover",))
    patterns = set(df["fraud_scenario_type"].dropna().unique())
    assert "account_takeover" not in patterns


def test_exclude_patterns_invalid_raises() -> None:
    with pytest.raises(ValueError):
        generate_synthetic_vn_dataset(_small_config(), exclude_patterns=("not_a_real_pattern",))


def test_generate_concept_drift_split_consistency() -> None:
    df_known, df_drift = generate_concept_drift_split(
        _small_config(), held_out_pattern="money_mule_chain"
    )
    known_patterns = set(df_known["fraud_scenario_type"].dropna().unique())
    drift_patterns = set(df_drift["fraud_scenario_type"].dropna().unique())

    assert "money_mule_chain" not in known_patterns
    assert "money_mule_chain" in drift_patterns
    # df_drift phải chứa mọi pattern khác cũng có trong df_known
    assert known_patterns.issubset(drift_patterns)


def test_generate_concept_drift_split_invalid_pattern_raises() -> None:
    with pytest.raises(ValueError):
        generate_concept_drift_split(_small_config(), held_out_pattern="unknown_pattern")


def test_fraud_pattern_names_cover_all_generators() -> None:
    # Bảo vệ chống lệch tên nếu ai đó thêm scenario mới nhưng quên cập nhật hằng số
    assert len(FRAUD_PATTERN_NAMES) == 3
