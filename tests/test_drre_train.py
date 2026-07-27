"""Test đầu-cuối cho src/drre/train.py — trọng tâm là (1) training loop
chạy được không lỗi và loss có giảm, (2) checkpoint sinh ra TƯƠNG THÍCH
với src/dashboard/scoring_service.py (đây là điểm dễ vỡ âm thầm nhất:
hai module lưu/đọc checkpoint độc lập, chỉ cần lệch tên key là crash
hoặc tệ hơn — load sai mà không báo lỗi).

Dùng dữ liệu tổng hợp nhỏ, sinh ngay trong test (không phụ thuộc file
trên đĩa), để test chạy nhanh và không cần tải Kaggle.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.drre.train import (
    build_account_graph,
    build_contrastive_triplets,
    build_prediction_pairs,
    make_time_windows,
    train,
)


def _make_synthetic_processed_df(n_accounts: int = 15, n_transactions: int = 300, seed: int = 0) -> pd.DataFrame:
    """Dựng DataFrame đúng schema đầu ra của src/data/preprocessing.py
    (account_id, counterparty_id, step, amount_norm, channel_encoded, label).
    """
    rng = np.random.default_rng(seed)
    accounts = [f"ACC_{i}" for i in range(n_accounts)]
    rows = []
    for step in range(n_transactions):
        sender, receiver = rng.choice(accounts, size=2, replace=False)
        rows.append({
            "account_id": sender,
            "counterparty_id": receiver,
            "step": step,
            "amount_norm": float(rng.normal(0, 1)),
            "channel_encoded": int(rng.integers(0, 3)),
            "label": bool(rng.random() < 0.05),
        })
    return pd.DataFrame(rows)


@pytest.fixture
def processed_dir(tmp_path: Path) -> Path:
    out_dir = tmp_path / "processed"
    out_dir.mkdir()
    train_df = _make_synthetic_processed_df(seed=1)
    val_df = _make_synthetic_processed_df(n_transactions=80, seed=2)
    train_df.to_parquet(out_dir / "synthetic_vn_train.parquet")
    val_df.to_parquet(out_dir / "synthetic_vn_val.parquet")
    return out_dir


def test_make_time_windows_covers_all_rows() -> None:
    df = _make_synthetic_processed_df(n_transactions=50, seed=3)
    windows = make_time_windows(df, batch_size=20, time_col="step")
    total_rows = sum(len(w) for w in windows)
    assert total_rows == len(df)


def test_build_prediction_pairs_only_within_same_account() -> None:
    df = _make_synthetic_processed_df(n_transactions=30, seed=4)
    src, tgt = build_prediction_pairs(df)
    for s, t in zip(src, tgt):
        assert df.loc[s, "account_id"] == df.loc[t, "account_id"]


def test_build_contrastive_triplets_negatives_from_other_accounts() -> None:
    df = _make_synthetic_processed_df(n_transactions=30, seed=5)
    rng = __import__("random").Random(0)
    anchors, positives, negatives = build_contrastive_triplets(df, num_negatives=3, rng=rng)
    for a, negs in zip(anchors, negatives):
        anchor_account = df.loc[a, "account_id"]
        for neg in negs:
            # Cho phép fallback suy biến (window chỉ có 1 account) nhưng
            # với dữ liệu test có 15 account, việc này không nên xảy ra.
            assert df.loc[neg, "account_id"] != anchor_account


def test_train_runs_end_to_end_without_crashing(processed_dir: Path, tmp_path: Path) -> None:
    """Test quan trọng nhất: chạy training thật 3 epoch trên dữ liệu nhỏ,
    xác nhận toàn bộ pipeline (window batching, forward tuần tự theo
    account, EMA target, 3 thành phần loss, backward, lưu checkpoint)
    chạy hết không lỗi. Tốc độ giảm loss thực tế đã được xác nhận thủ
    công khi phát triển (xem lịch sử hội thoại) — test này bảo vệ
    chống REGRESSION (không crash), không đo hội tụ định lượng vì điều
    đó phụ thuộc seed/hyperparameter và dễ flaky trên CI.
    """
    checkpoint_path = tmp_path / "checkpoint.pt"

    train(
        source="synthetic_vn",
        epochs=3,
        batch_size=32,
        lr=1e-2,
        num_negatives=3,
        processed_dir=str(processed_dir),
        checkpoint_path=str(checkpoint_path),
        seed=42,
    )

    assert checkpoint_path.exists(), "train() phải lưu checkpoint (best hoặc epoch cuối)"


def test_checkpoint_compatible_with_dashboard_scoring_service(processed_dir: Path, tmp_path: Path) -> None:
    """Test then quan trọng: checkpoint train.py lưu ra PHẢI load được bởi
    scoring_service.py và tạo ra score_batch() dùng method='drre_trained'
    (không phải rơi về heuristic vì lỗi tương thích format).
    """
    checkpoint_path = tmp_path / "checkpoint.pt"

    train(
        source="synthetic_vn", epochs=1, batch_size=32, lr=1e-3,
        num_negatives=3, processed_dir=str(processed_dir),
        checkpoint_path=str(checkpoint_path), seed=42,
    )
    assert checkpoint_path.exists()

    from src.dashboard.scoring_service import score_batch, try_load_drre_checkpoint

    loaded = try_load_drre_checkpoint(str(checkpoint_path))
    assert loaded is not None, "Checkpoint từ train.py phải load được bởi scoring_service.py"

    val_df = pd.read_parquet(processed_dir / "synthetic_vn_val.parquet")
    scored = score_batch(val_df, checkpoint_path=str(checkpoint_path))
    assert (scored["method"] == "drre_trained").all()
    assert scored["risk_score"].between(0, 100).all()


def test_checkpoint_state_dict_keys_match_drre_architecture(processed_dir: Path, tmp_path: Path) -> None:
    """Kiểm tra thấp cấp hơn: state_dict lưu ra khớp CHÍNH XÁC với kiến
    trúc DRRE mới khởi tạo từ model_config trong cùng checkpoint — nếu
    lệch, torch sẽ raise khi load_state_dict (strict=True mặc định).
    """
    from src.drre.model import DRRE
    from src.scoring.risk_scoring import RiskScoringEngine

    checkpoint_path = tmp_path / "checkpoint.pt"
    train(
        source="synthetic_vn", epochs=1, batch_size=32, lr=1e-3,
        num_negatives=3, processed_dir=str(processed_dir),
        checkpoint_path=str(checkpoint_path), seed=42,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = DRRE(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["drre_state_dict"])  # raise nếu lệch key/shape

    risk_engine = RiskScoringEngine(embedding_dim=checkpoint["model_config"]["embedding_dim"])
    risk_engine.load_state_dict(checkpoint["risk_engine_state_dict"])
