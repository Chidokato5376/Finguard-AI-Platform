"""Đặc trưng nhân quả (src/data/features.py).

Test quan trọng nhất ở đây là test_them_du_lieu_tuong_lai_khong_doi_dac_trung_cu:
nếu một đặc trưng dùng thống kê trên cả cột (mean/std/count toàn tập) thì khi
thêm dữ liệu TƯƠNG LAI vào, đặc trưng của các dòng CŨ sẽ thay đổi — đó chính
là dấu hiệu rò rỉ. Test này bắt được mọi lỗi loại đó mà không cần đọc code.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.features import (
    BASE_FEATURES, CLOCK_FEATURES, COUNTERPARTY_FEATURES, SECONDS_PER_STEP,
    add_causal_features, features_for_source,
)


def _frame(rows: list[tuple]) -> pd.DataFrame:
    """rows = (step, account_id, counterparty_id, amount)."""
    df = pd.DataFrame(rows, columns=["step", "account_id", "counterparty_id", "amount"])
    df["amount_norm"] = np.log1p(df["amount"])
    df["channel"] = "transfer"
    df["channel_encoded"] = 0
    df["label"] = False
    return df


BASIC = _frame([
    (10, "A", "X", 1_000_000.0),
    (11, "A", "Y", 1_200_000.0),
    (12, "B", "X", 50_000.0),
    (13, "A", "X", 30_000_000.0),   # A chuyển gấp nhiều lần mức thường
    (14, "B", "Z", 60_000.0),
])

FUTURE = _frame([
    (20, "A", "W", 900_000.0),
    (21, "B", "X", 55_000.0),
    (22, "C", "A", 10_000.0),
])


def test_them_du_lieu_tuong_lai_khong_doi_dac_trung_cu():
    """Đặc trưng của 5 dòng đầu phải GIỮ NGUYÊN khi nối thêm dòng tương lai."""
    cols = features_for_source("paysim", include_graph=False)
    truoc = add_causal_features(BASIC, "paysim")[cols]
    sau = add_causal_features(
        pd.concat([BASIC, FUTURE], ignore_index=True), "paysim"
    ).iloc[: len(BASIC)][cols]
    pd.testing.assert_frame_equal(truoc, sau, check_exact=False, rtol=1e-12)


def test_z_amount_va_log_amount_ratio_khop_voi_tinh_tay():
    out = add_causal_features(BASIC, "paysim")
    log_amt = np.log1p(BASIC["amount"].to_numpy())
    # Dòng 3 (index 3) là giao dịch thứ 3 của A; lịch sử là dòng 0 và 1
    prior = log_amt[[0, 1]]
    z_tay = (log_amt[3] - prior.mean()) / prior.std(ddof=1)
    assert out.loc[3, "z_amount"] == pytest.approx(min(z_tay, 5.0), rel=1e-9)
    assert out.loc[3, "log_amount_ratio"] == pytest.approx(
        min(log_amt[3] - prior.mean(), 5.0), rel=1e-9)
    # Giao dịch ĐẦU TIÊN của một tài khoản không có lịch sử -> 0, không phải NaN
    assert out.loc[0, "z_amount"] == 0.0
    assert out.loc[0, "log_amount_ratio"] == 0.0
    assert out.loc[0, "log_n_prior"] == 0.0


def test_log_gap_dung_don_vi_cua_tung_nguon():
    """step của PaySim là GIỜ, của IEEE-CIS là GIÂY — nhầm là sai 3600 lần."""
    assert SECONDS_PER_STEP["paysim"] == 3600.0
    assert SECONDS_PER_STEP["ieee_cis"] == 1.0
    out = add_causal_features(BASIC, "paysim")
    # Dòng 1 cách dòng 0 đúng 1 step = 1 giờ = 3600 giây
    assert out.loc[1, "log_gap"] == pytest.approx(np.log1p(3600.0))
    out_ieee = add_causal_features(BASIC, "ieee_cis")
    assert out_ieee.loc[1, "log_gap"] == pytest.approx(np.log1p(1.0))


def test_fan_in_bat_duoc_tai_khoan_trung_gian():
    """5 người gửi khác nhau dồn tiền vào một tài khoản -> log_n_distinct_senders tăng."""
    mule = _frame([(100 + i, f"S{i}", "MULE", 20_000_000.0) for i in range(5)])
    out = add_causal_features(mule, "paysim")
    # Dòng cuối: đã có 4 người gửi khác nhau chuyển cho MULE trước đó
    assert out.loc[4, "log_n_distinct_senders"] == pytest.approx(np.log1p(4))
    assert out.loc[0, "log_n_distinct_senders"] == 0.0
    # Mỗi người gửi chỉ có 1 giao dịch -> không có lịch sử riêng để so
    assert (out["log_n_prior"] == 0).all()


def test_new_counterparty_va_so_doi_tac_khac_nhau():
    out = add_causal_features(BASIC, "paysim")
    assert out.loc[0, "new_counterparty"] == 1.0   # A -> X lần đầu
    assert out.loc[1, "new_counterparty"] == 1.0   # A -> Y lần đầu
    assert out.loc[3, "new_counterparty"] == 0.0   # A -> X lặp lại
    # Tại dòng 3, A đã từng giao dịch với 2 đối tác khác nhau (X, Y)
    assert out.loc[3, "log_n_distinct_cp"] == pytest.approx(np.log1p(2))


def test_is_round_amount():
    df = _frame([(1, "A", "X", 5_000_000.0), (2, "A", "Y", 1_234_567.0)])
    out = add_causal_features(df, "paysim")
    assert out.loc[0, "is_round_amount"] == 1.0
    assert out.loc[1, "is_round_amount"] == 0.0


def test_paysim_khong_dung_dac_trung_dong_ho():
    """Giờ trong ngày là đặc trưng RÒ RỈ của PaySim (giờ 4-8 có 100% gian lận)."""
    cols = features_for_source("paysim", include_graph=False)
    assert not (set(cols) & set(CLOCK_FEATURES)), cols
    assert len(cols) == 11


def test_ieee_khong_dung_dac_trung_doi_tac():
    """IEEE-CIS có counterparty_id = None cho toàn bộ bản ghi."""
    cols = features_for_source("ieee_cis", include_graph=False)
    assert not (set(cols) & set(COUNTERPARTY_FEATURES)), cols
    assert set(CLOCK_FEATURES) <= set(cols)
    assert len(cols) == 10


def test_amlworld_dung_14_dac_trung_va_LOAI_is_fx():
    """`is_fx` phải NẰM NGOÀI bộ đặc trưng — nó là đặc trưng rò rỉ NGƯỢC.

    Đo trên toàn bộ 5.078.345 dòng HI-Small ngày 2026-10-19: 72.170 giao dịch
    đổi tiền có ĐÚNG 0 nhãn dương, trong khi giao dịch cùng loại tiền có tỷ lệ
    0,1034%. Tức `is_fx = 1` là dấu hiệu loại trừ hoàn hảo do máy sinh dữ liệu
    tạo ra.

    Nguy hiểm hơn việc AUPRC phình lên: ngoài thực tế, đổi tiền LÀ dấu hiệu
    của bước layering — chiều quan hệ thật ngược hẳn. Mô hình học trên đặc
    trưng này sẽ giảm điểm rủi ro cho đúng những giao dịch xuyên tiền tệ đáng
    nghi nhất khi triển khai.

    Test này tồn tại để không ai thêm `is_fx` trở lại vì thấy "bộ amlworld
    đầy đủ nhất mà lại thiếu một cột".
    """
    cols = features_for_source("amlworld", include_graph=False)
    assert "is_fx" not in cols, (
        "is_fx là đặc trưng rò rỉ ngược — xem cảnh báo đầu src/data/features.py")
    assert "is_fx" not in features_for_source("transxion", include_graph=False)
    assert len(cols) == 14


def test_van_chuan_hoa_is_fx_ve_float_de_con_bao_cao_duoc():
    """Cột `is_fx` vẫn phải nằm trong dữ liệu (để báo cáo lỗi), chỉ không vào mô hình."""
    frame = BASIC.assign(timestamp=pd.Timestamp("2022-09-01"), is_fx=True)
    out = add_causal_features(frame, "amlworld")
    assert out["is_fx"].dtype == float


def test_thieu_cot_is_fx_khong_con_lam_hong_pipeline():
    """Trước đây thiếu `is_fx` là lỗi cứng; giờ nó không còn là đặc trưng nữa."""
    out = add_causal_features(BASIC.assign(timestamp=pd.Timestamp("2022-09-01")), "amlworld")
    assert set(features_for_source("amlworld", include_graph=False)) - set(BASE_FEATURES) <= set(out.columns)


def test_moi_dac_trung_deu_huu_han():
    """Không được để NaN/inf lọt vào mô hình — torch sẽ ra loss=nan âm thầm."""
    out = add_causal_features(BASIC, "paysim")
    vals = out[features_for_source("paysim", include_graph=False)].to_numpy(dtype=float)
    assert np.isfinite(vals).all()
