"""Kiểm thử đặc trưng dòng tiền hai chiều.

Phép kiểm quan trọng nhất là `test_khong_nhin_thay_tuong_lai`: thêm giao dịch
ở tương lai rồi đòi mọi giá trị cũ KHÔNG đổi một số nào. Đó là định nghĩa
thao tác được của tính nhân quả, và là cùng một phép kiểm đã dùng cho
tests/test_features.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.graph_features import (
    ALL_GRAPH_FEATURES,
    add_graph_features,
    graph_features_for_source,
)


def _frame(rows: list[tuple]) -> pd.DataFrame:
    """rows = (account_id, counterparty_id, amount), theo đúng thứ tự thời gian."""
    df = pd.DataFrame(rows, columns=["account_id", "counterparty_id", "amount"])
    df["timestamp"] = pd.date_range("2022-09-01", periods=len(df), freq="h")
    return df


# A gửi B 100, rồi B gửi C 90 (B là trung chuyển), rồi A gửi C 50.
CHAIN = [("A", "B", 100.0), ("B", "C", 90.0), ("A", "C", 50.0)]


def test_nguon_khong_co_doi_tac_thi_bo_qua():
    assert graph_features_for_source("ieee_cis") == []
    df = _frame(CHAIN)
    assert add_graph_features(df, "ieee_cis") is df or \
        not set(ALL_GRAPH_FEATURES) & set(add_graph_features(df, "ieee_cis").columns)


def test_sinh_du_cac_cot_cho_amlworld():
    out = add_graph_features(_frame(CHAIN), "amlworld")
    assert set(ALL_GRAPH_FEATURES) <= set(out.columns)
    assert np.isfinite(out[ALL_GRAPH_FEATURES].to_numpy(dtype=float)).all()


def test_dong_dau_tien_khong_co_lich_su():
    out = add_graph_features(_frame(CHAIN), "amlworld")
    first = out.iloc[0]
    assert first["log_amt_out_prior"] == pytest.approx(0.0)
    assert first["log_amt_in_prior"] == pytest.approx(0.0)
    assert first["log_n_in_prior"] == pytest.approx(0.0)
    assert first["passthrough_ratio"] == pytest.approx(0.0)
    assert first["flow_imbalance"] == pytest.approx(0.0)


def test_chieu_NHAN_duoc_ghi_nhan_cho_ben_gui():
    """Điểm mới của module: bên gửi của giao dịch 2 là B, và B ĐÃ NHẬN 100
    ở giao dịch 1. Trước đây không đặc trưng nào mô tả được điều đó."""
    out = add_graph_features(_frame(CHAIN), "amlworld")
    b_sends = out.iloc[1]
    assert b_sends["account_id"] == "B"
    assert b_sends["log_amt_in_prior"] == pytest.approx(np.log1p(100.0))
    assert b_sends["log_n_in_prior"] == pytest.approx(np.log1p(1.0))
    # B chưa từng chuyển đi trước đó
    assert b_sends["log_amt_out_prior"] == pytest.approx(0.0)


def test_khong_nhin_thay_khoan_thu_cua_chinh_giao_dich_do():
    """Bên gửi KHÔNG được thấy khoản bên nhận thu trong cùng giao dịch."""
    out = add_graph_features(_frame([("A", "B", 100.0)]), "amlworld")
    assert out.iloc[0]["log_amt_in_prior"] == pytest.approx(0.0)
    assert out.iloc[0]["log_cp_amt_in_prior"] == pytest.approx(0.0)


def test_tai_khoan_trung_chuyen_co_passthrough_cao():
    """A gửi B 100; B gửi C 100; rồi B gửi D 1 -> lúc gửi D, B đã thu 100 chi 100."""
    out = add_graph_features(
        _frame([("A", "B", 100.0), ("B", "C", 100.0), ("B", "D", 1.0)]), "amlworld")
    b_third = out.iloc[2]
    assert b_third["account_id"] == "B"
    assert b_third["passthrough_ratio"] == pytest.approx(1.0)
    assert b_third["flow_imbalance"] == pytest.approx(0.0)


def test_tai_khoan_chi_nhan_thi_flow_imbalance_bang_1():
    """A gửi B hai lần; lần B tự gửi đi thì B mới có chi."""
    out = add_graph_features(
        _frame([("A", "B", 50.0), ("A", "B", 50.0), ("B", "Z", 1.0)]), "amlworld")
    b = out.iloc[2]
    assert b["flow_imbalance"] == pytest.approx(1.0)   # chỉ thu, chưa chi
    assert b["passthrough_ratio"] == pytest.approx(0.0)


def test_doi_tac_duoc_ghi_nhan_lich_su_nhan():
    """Giao dịch 3 gửi cho C, mà C đã nhận 90 ở giao dịch 2."""
    out = add_graph_features(_frame(CHAIN), "amlworld")
    third = out.iloc[2]
    assert third["counterparty_id"] == "C"
    assert third["log_cp_amt_in_prior"] == pytest.approx(np.log1p(90.0))


def test_khong_nhin_thay_tuong_lai():
    """Thêm giao dịch ở tương lai KHÔNG được làm đổi bất kỳ giá trị cũ nào."""
    base = add_graph_features(_frame(CHAIN), "amlworld")
    longer = add_graph_features(
        _frame(CHAIN + [("B", "A", 999.0), ("C", "A", 777.0)]), "amlworld")
    for col in ALL_GRAPH_FEATURES:
        np.testing.assert_allclose(
            base[col].to_numpy(dtype=float),
            longer[col].to_numpy(dtype=float)[: len(base)],
            err_msg=f"'{col}' đổi giá trị khi thêm dòng tương lai — RÒ RỈ",
        )


def test_counterparty_rong_thi_bao_loi_ro_rang():
    df = _frame(CHAIN)
    df.loc[1, "counterparty_id"] = None
    with pytest.raises(ValueError, match="counterparty_id rỗng"):
        add_graph_features(df, "amlworld")


def test_features_for_source_kem_nhom_do_thi():
    from src.data.features import features_for_source

    with_graph = features_for_source("amlworld")
    without = features_for_source("amlworld", include_graph=False)
    assert len(with_graph) == len(without) + len(ALL_GRAPH_FEATURES)
    assert set(ALL_GRAPH_FEATURES) <= set(with_graph)
    assert not set(ALL_GRAPH_FEATURES) & set(without)
    # ieee_cis không có đối tác -> hai danh sách phải bằng nhau
    assert features_for_source("ieee_cis") == features_for_source(
        "ieee_cis", include_graph=False)
