"""Unit tests cho src/data/loaders.py::load_ieee_cis.

Trọng tâm: (1) ánh xạ đúng schema thật của IEEE-CIS (đã xác minh qua
tài liệu Kaggle chính thức, không suy đoán), (2) các giới hạn đã biết
(counterparty_id=None, ip_country=None, account_id là pseudo-ID) được
tuân thủ NHẤT QUÁN — không bị "quên" trong một số trường hợp.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.loaders import load_ieee_cis


def _fake_transaction_csv(tmp_path, n: int = 200, seed: int = 0) -> str:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "TransactionID": range(1000, 1000 + n),
        "isFraud": rng.choice([0, 1], n, p=[0.95, 0.05]),
        "TransactionDT": np.sort(rng.integers(86400, 86400 * 30, n)),
        "TransactionAmt": np.round(rng.exponential(120, n), 2),
        "ProductCD": rng.choice(["W", "C", "H", "S", "R"], n),
        "card1": rng.integers(1000, 9999, n),
        "card2": rng.choice([100, 200, 300, np.nan], n),
        "card3": rng.choice([150, 185], n),
        "card5": rng.choice([100, 102, 226], n),
        "addr1": rng.choice(range(100, 500), n),
        "P_emaildomain": rng.choice(["gmail.com", "yahoo.com", np.nan], n),
    })
    path = tmp_path / "train_transaction.csv"
    df.to_csv(path, index=False)
    return str(path)


def _fake_identity_csv(tmp_path, transaction_path: str, coverage: float = 0.3, seed: int = 1) -> str:
    rng = np.random.default_rng(seed)
    tx_ids = pd.read_csv(transaction_path)["TransactionID"].to_numpy()
    covered = rng.choice(tx_ids, size=int(len(tx_ids) * coverage), replace=False)
    identity = pd.DataFrame({
        "TransactionID": covered,
        "DeviceInfo": rng.choice(["Windows", "iOS Device", "SM-G900"], len(covered)),
    })
    path = tmp_path / "train_identity.csv"
    identity.to_csv(path, index=False)
    return str(path)


def test_missing_file_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_ieee_cis(str(tmp_path / "does_not_exist.csv"))


def test_missing_required_columns_raises(tmp_path) -> None:
    bad_path = tmp_path / "bad.csv"
    pd.DataFrame({"TransactionID": [1, 2]}).to_csv(bad_path, index=False)
    with pytest.raises(ValueError):
        load_ieee_cis(str(bad_path))


def test_counterparty_id_always_none(tmp_path) -> None:
    """Giới hạn quan trọng nhất: IEEE-CIS không có người nhận -> phải
    LUÔN None, không được suy diễn hay để trống rỗng ("") gây hiểu nhầm.
    """
    path = _fake_transaction_csv(tmp_path)
    df = load_ieee_cis(path)
    assert df["counterparty_id"].isna().all()


def test_ip_country_always_none(tmp_path) -> None:
    """Không được tự gán ý nghĩa địa lý cho các cột id_* đã ẩn danh hóa."""
    path = _fake_transaction_csv(tmp_path)
    df = load_ieee_cis(path)
    assert df["ip_country"].isna().all()


def test_basic_field_mapping(tmp_path) -> None:
    path = _fake_transaction_csv(tmp_path, n=50)
    df = load_ieee_cis(path)
    raw = pd.read_csv(path)

    assert len(df) == len(raw)
    assert (df["amount"].to_numpy() == raw["TransactionAmt"].to_numpy()).all()
    assert (df["step"].to_numpy() == raw["TransactionDT"].to_numpy()).all()
    assert (df["label"].to_numpy() == raw["isFraud"].astype(bool).to_numpy()).all()
    assert (df["channel"].to_numpy() == raw["ProductCD"].to_numpy()).all()
    assert (df["source"] == "ieee_cis").all()


def test_account_id_is_deterministic_pseudo_id(tmp_path) -> None:
    """Cùng tổ hợp card/addr/email phải luôn ra cùng account_id (tính
    chất tối thiểu để pseudo-ID này có ích, dù không chính xác tuyệt đối).
    """
    path = _fake_transaction_csv(tmp_path, n=20)
    df1 = load_ieee_cis(path)
    df2 = load_ieee_cis(path)
    assert (df1["account_id"] == df2["account_id"]).all()
    assert df1["account_id"].str.startswith("IEEECIS_").all()


def test_without_identity_path_device_id_all_none(tmp_path) -> None:
    path = _fake_transaction_csv(tmp_path)
    df = load_ieee_cis(path, identity_path=None)
    assert df["device_id"].isna().all()


def test_with_identity_path_partial_device_id_coverage(tmp_path) -> None:
    """identity table không phủ hết mọi giao dịch -> device_id có thật
    (không tổng hợp) cho MỘT PHẦN, phần còn lại vẫn None."""
    tx_path = _fake_transaction_csv(tmp_path, n=200)
    id_path = _fake_identity_csv(tmp_path, tx_path, coverage=0.3)

    df = load_ieee_cis(tx_path, identity_path=id_path)
    matched = df["device_id"].notna().sum()
    assert 0 < matched < len(df)  # phải là PHỦ MỘT PHẦN, không phải 0% hay 100%


def test_missing_identity_file_warns_and_falls_back(tmp_path) -> None:
    tx_path = _fake_transaction_csv(tmp_path)
    df = load_ieee_cis(tx_path, identity_path=str(tmp_path / "no_such_identity.csv"))
    assert df["device_id"].isna().all()


def test_is_synthetic_field_always_empty(tmp_path) -> None:
    """device_id từ identity table là dữ liệu THẬT (không tổng hợp) —
    is_synthetic_field phải luôn rỗng cho nguồn ieee_cis, khác với PaySim
    khi synthesize_device_ip=True.
    """
    path = _fake_transaction_csv(tmp_path)
    df = load_ieee_cis(path)
    assert (df["is_synthetic_field"].apply(len) == 0).all()


def test_ieee_cis_output_rejected_by_graph_encoder(tmp_path) -> None:
    """Test tích hợp quan trọng: output của load_ieee_cis (counterparty_id
    = None cho toàn bộ) PHẢI bị build_account_graph từ chối tường minh —
    nếu không, mọi dòng sẽ map về CÙNG một 'node None', tạo ra một hub giả
    kết nối với mọi tài khoản mà KHÔNG có bất kỳ lỗi/cảnh báo runtime nào
    (lỗi âm thầm đã phát hiện qua kiểm thử thực tế và vá trong
    src/dashboard/graph_utils.py — test này bảo vệ chống regression).
    """
    from src.dashboard.graph_utils import build_account_graph

    path = _fake_transaction_csv(tmp_path, n=30)
    df = load_ieee_cis(path)
    df["amount_norm"] = 0.0  # cột giả lập, chỉ cần tồn tại cho lời gọi này
    df["channel_encoded"] = 0

    with pytest.raises(ValueError, match="counterparty_id"):
        build_account_graph(df, ["amount_norm", "channel_encoded"])
