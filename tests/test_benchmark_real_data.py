"""Test cho src/qape/benchmark.py — trọng tâm là build_alerts_from_real_data()
và run_benchmark_on_alerts(), đảm bảo đường dữ liệu thật hoạt động đúng và
KHÔNG lệch pha với pipeline Dashboard thật (data_service/scoring_service/
qape_service).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.qape.benchmark import (
    build_alerts_from_real_data,
    generate_synthetic_batch,
    run_benchmark_on_alerts,
)


def _make_synthetic_processed_df(n_accounts: int = 20, n_transactions: int = 200, seed: int = 0) -> pd.DataFrame:
    """Dựng DataFrame đúng schema đầu ra của src/data/preprocessing.py."""
    rng = np.random.default_rng(seed)
    accounts = [f"ACC_{i}" for i in range(n_accounts)]
    rows = []
    for step in range(n_transactions):
        sender, receiver = rng.choice(accounts, size=2, replace=False)
        amount = float(rng.exponential(1_000_000))
        rows.append({
            "account_id": sender,
            "counterparty_id": receiver,
            "step": step,
            "amount": amount,
            "amount_norm": float(np.log1p(amount)),
            "channel_encoded": int(rng.integers(0, 3)),
            "label": bool(rng.random() < 0.05),
        })
    return pd.DataFrame(rows)


@pytest.fixture
def processed_dir(tmp_path: Path) -> Path:
    out_dir = tmp_path / "processed"
    out_dir.mkdir()
    _make_synthetic_processed_df(seed=1).to_parquet(out_dir / "synthetic_vn_test.parquet")
    return out_dir


def test_build_alerts_from_real_data_uses_heuristic_when_no_checkpoint(processed_dir: Path) -> None:
    """Không có checkpoint -> phải tự rơi về Heuristic Scorer (đúng hành
    vi của scoring_service.py), KHÔNG raise lỗi.
    """
    alerts = build_alerts_from_real_data(
        source="synthetic_vn",
        split="test",
        processed_dir=str(processed_dir),
        checkpoint_path=str(processed_dir / "nonexistent_checkpoint.pt"),
        top_n=10,
    )
    assert len(alerts) == 10
    assert all(0.0 <= a.risk_score <= 100.0 for a in alerts)


def test_build_alerts_from_real_data_missing_file_raises_clear_error(processed_dir: Path) -> None:
    with pytest.raises(FileNotFoundError):
        build_alerts_from_real_data(
            source="nonexistent_source", split="test", processed_dir=str(processed_dir),
        )


def test_build_alerts_from_real_data_respects_top_n(processed_dir: Path) -> None:
    alerts = build_alerts_from_real_data(
        source="synthetic_vn", split="test", processed_dir=str(processed_dir), top_n=5,
    )
    assert len(alerts) == 5


def test_build_alerts_from_real_data_sorted_by_risk_descending(processed_dir: Path) -> None:
    alerts = build_alerts_from_real_data(
        source="synthetic_vn", split="test", processed_dir=str(processed_dir), top_n=10,
    )
    scores = [a.risk_score for a in alerts]
    assert scores == sorted(scores, reverse=True)


def test_run_benchmark_on_alerts_handles_empty_list_gracefully() -> None:
    """Không được crash khi danh sách alerts rỗng (vd. dữ liệu quá ít)."""
    run_benchmark_on_alerts([], budget=50.0, lambda_1_values=[1.0], run_qaoa=False)


def test_run_benchmark_on_alerts_runs_on_synthetic_data() -> None:
    """Test hồi quy: đảm bảo tách run_benchmark_on_alerts ra khỏi
    run_benchmark KHÔNG làm hỏng đường dữ liệu tổng hợp cũ.
    """
    alerts = generate_synthetic_batch(n_alerts=6, seed=1)
    run_benchmark_on_alerts(alerts, budget=15.0, lambda_1_values=[1.0, 5.0], run_qaoa=False)
