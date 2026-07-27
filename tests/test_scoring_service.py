"""Unit tests cho src/dashboard/scoring_service.py và data_service.py.

Trọng tâm: đảm bảo heuristic score có tín hiệu phân biệt thật (không phải
ngẫu nhiên) và luôn gắn nhãn `method` minh bạch — đúng cam kết trung thực
nêu trong docstring scoring_service.py.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.dashboard.scoring_service import (
    classify_tier,
    compute_heuristic_scores,
    try_load_drre_checkpoint,
)


def _build_synthetic_frame(n_normal: int = 200, n_anomalies: int = 20, seed: int = 0) -> pd.DataFrame:
    """Dựng một DataFrame tối giản có tín hiệu bất thường RÕ RÀNG, để kiểm
    tra heuristic có bắt được tín hiệu đó hay không (không phải test số
    tuyệt đối, mà test tính đơn điệu: bất thường phải có z_score/risk_score
    cao hơn bình thường).
    """
    rng = np.random.default_rng(seed)
    base_time = datetime(2026, 1, 1)
    rows = []
    account = "ACC_TEST"

    # Giao dịch bình thường: amount ổn định quanh 500,000
    for i in range(n_normal):
        rows.append({
            "account_id": account,
            "counterparty_id": f"CP_{i % 5}",  # lặp lại 5 counterparty quen thuộc
            "step": i,
            "amount": float(rng.normal(500_000, 50_000)),
            "label": False,
        })

    # Giao dịch bất thường: amount cao gấp ~50 lần, counterparty hoàn toàn mới
    for i in range(n_anomalies):
        rows.append({
            "account_id": account,
            "counterparty_id": f"CP_NEW_{i}",
            "step": n_normal + i,
            "amount": float(rng.normal(25_000_000, 1_000_000)),
            "label": True,
        })

    return pd.DataFrame(rows)


def test_heuristic_score_always_tags_method() -> None:
    df = _build_synthetic_frame()
    scored = compute_heuristic_scores(df)
    assert (scored["method"] == "heuristic_fallback").all()


def test_heuristic_score_discriminates_anomalies() -> None:
    """Test quan trọng nhất: risk_score của giao dịch bất thường (label=True)
    phải cao hơn RÕ RỆT so với giao dịch bình thường — nếu test này fail,
    heuristic không có giá trị sử dụng thực tế.
    """
    df = _build_synthetic_frame()
    scored = compute_heuristic_scores(df)

    mean_normal = scored.loc[~scored["label"], "risk_score"].mean()
    mean_anomaly = scored.loc[scored["label"], "risk_score"].mean()

    assert mean_anomaly > mean_normal, (
        f"Heuristic không phân biệt được bất thường: "
        f"mean_anomaly={mean_anomaly:.2f} <= mean_normal={mean_normal:.2f}"
    )
    # Yêu cầu khoảng cách đủ lớn, không chỉ nhỉnh hơn một chút do nhiễu
    assert mean_anomaly - mean_normal > 20


def test_heuristic_score_no_leakage_from_future() -> None:
    """z_score tại thời điểm t chỉ được dùng dữ liệu TRƯỚC t — kiểm tra
    bằng cách đảo ngược thứ tự nạp dữ liệu, kết quả phải giống hệt (vì
    hàm tự sort lại theo time_col trước khi tính).
    """
    df = _build_synthetic_frame(n_normal=50, n_anomalies=5)
    scored_forward = compute_heuristic_scores(df)
    scored_shuffled = compute_heuristic_scores(df.sample(frac=1, random_state=1))

    merged = scored_forward.merge(
        scored_shuffled, on=["account_id", "counterparty_id", "step"], suffixes=("_a", "_b")
    )
    assert np.allclose(merged["risk_score_a"], merged["risk_score_b"], atol=1e-6)


def test_first_transactions_get_neutral_zscore() -> None:
    """Giao dịch đầu tiên của một tài khoản chưa có lịch sử -> z_score = 0
    (trung tính), không phải NaN hay lỗi."""
    df = _build_synthetic_frame(n_normal=10, n_anomalies=0)
    scored = compute_heuristic_scores(df)
    assert scored.iloc[0]["z_score"] == 0.0
    assert not scored["risk_score"].isna().any()


def test_classify_tier_thresholds() -> None:
    assert classify_tier(10) == "low"
    assert classify_tier(30) == "low"
    assert classify_tier(31) == "medium"
    assert classify_tier(70) == "medium"
    assert classify_tier(71) == "critical"
    assert classify_tier(100) == "critical"


def test_try_load_drre_checkpoint_returns_none_when_missing() -> None:
    """Không có checkpoint -> PHẢI trả về None, không raise, không tự
    dùng trọng số ngẫu nhiên (xem cảnh báo trung thực trong module)."""
    result = try_load_drre_checkpoint("models/nonexistent_checkpoint.pt")
    assert result is None
