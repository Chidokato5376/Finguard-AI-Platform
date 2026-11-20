"""Đặc trưng theo kịch bản: đối chiếu tính tay, nhân quả, không phụ thuộc thứ tự dòng.

Phép kiểm quan trọng nhất là `test_nhan_qua_tung_mo_dun`: thêm giao dịch ở tương lai rồi
đòi mọi giá trị cũ không đổi một bit nào.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import scenarios
from src.data.scenarios import REGISTRY, features_of, resolve
from src.data.scenarios.stream import DAY, HOUR, DistinctIndex, Stream

ALL_MODULES = sorted(REGISTRY)


def _random(n: int, seed: int = 0, start: str = "2022-09-01") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    minutes = np.sort(rng.integers(0, 60 * 24 * 10, n))
    accounts = [f"B{b}_{a}" for b in range(3) for a in range(8)]
    return pd.DataFrame({
        "account_id": rng.choice(accounts, n),
        "counterparty_id": rng.choice(accounts, n),
        "timestamp": pd.Timestamp(start) + pd.to_timedelta(minutes, unit="m"),
        "amount_usd": np.round(np.exp(rng.normal(7, 2, n)), 2),
        "channel": rng.choice(["ach", "cash", "cheque", "wire"], n),
    })


def _rows(rows: list[tuple]) -> pd.DataFrame:
    """rows = (gửi, nhận, phút từ mốc, USD, kênh)."""
    frame = pd.DataFrame(rows, columns=["account_id", "counterparty_id", "minute", "amount_usd", "channel"])
    frame["timestamp"] = pd.Timestamp("2022-09-01") + pd.to_timedelta(frame.pop("minute"), unit="m")
    return frame


def _compute(name: str, frame: pd.DataFrame) -> dict[str, np.ndarray]:
    return REGISTRY[name].compute(Stream.from_frame(frame))


# --------------------------------------------------------------------------
# Bộ công cụ, đối chiếu với vòng lặp ngây thơ
# --------------------------------------------------------------------------

def _naive_window(entity, time, query_entity, query_time, window):
    return [np.flatnonzero((entity == e) & (time >= t - window) & (time < t))
            for e, t in zip(query_entity, query_time)]


@pytest.mark.parametrize("window", [HOUR, DAY, 3 * DAY])
def test_dem_va_tong_khop_tinh_tay(window):
    stream = Stream.from_frame(_random(600))
    lo, hi = stream.out.bounds(stream.q_s, window)
    expected = _naive_window(stream.s, stream.t, stream.s, stream.t, window)
    np.testing.assert_array_equal(hi - lo, [len(e) for e in expected])
    np.testing.assert_array_equal(
        stream.out.sum(stream.cents, lo, hi), [stream.cents[e].sum() for e in expected])


@pytest.mark.parametrize("window", [HOUR, DAY, 3 * DAY])
def test_dem_khac_biet_khop_tinh_tay(window):
    stream = Stream.from_frame(_random(600, seed=1))
    expected = _naive_window(stream.s, stream.t, stream.s, stream.t, window)
    np.testing.assert_array_equal(
        stream.out_distinct.count(stream.q_s, window),
        [len(set(stream.r[e].tolist())) for e in expected])


def test_dem_khac_biet_voi_doi_tac_lap_lai_cung_phut():
    stream = Stream.from_frame(_rows([
        ("B1_a", "B1_x", 0, 1.0, "ach"), ("B1_a", "B1_x", 0, 2.0, "ach"),
        ("B1_a", "B1_x", 5, 3.0, "ach"), ("B1_a", "B1_y", 5, 4.0, "ach"),
        ("B1_a", "B1_z", 10, 5.0, "ach"),
    ]))
    index = DistinctIndex(stream.s, stream.r, stream.t, stream.base)
    np.testing.assert_array_equal(index.count(stream.q_s, DAY), [0, 0, 1, 1, 2])


def test_cua_so_gom_dau_mut_cu_va_bo_cung_thoi_diem():
    frame = _rows([("B1_a", "B1_x", 0, 1.0, "ach"), ("B1_a", "B1_y", 60, 1.0, "ach"),
                   ("B1_a", "B1_z", 60, 1.0, "ach")])
    stream = Stream.from_frame(frame)
    lo, hi = stream.out.bounds(stream.q_s, HOUR)
    # t − W đúng bằng lần đầu nên vẫn tính; hai dòng cùng phút không thấy nhau.
    np.testing.assert_array_equal(hi - lo, [0, 1, 1])


def test_cap_lan_dau_khong_phu_thuoc_dong_cung_phut():
    stream = Stream.from_frame(_rows([
        ("B1_a", "B1_x", 0, 1.0, "ach"), ("B1_a", "B1_x", 0, 2.0, "ach"),
        ("B1_a", "B1_x", 9, 3.0, "ach")]))
    np.testing.assert_array_equal(stream.first_pair, [1, 1, 0])


# --------------------------------------------------------------------------
# Tính chất chung của mọi mô-đun
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", ALL_MODULES)
def test_nhan_qua_tung_mo_dun(name):
    past = _random(500, seed=2)
    future = _random(300, seed=3, start="2022-09-11")
    future.loc[:, "timestamp"] = future["timestamp"] + pd.Timedelta(days=1)
    joined = pd.concat([past, future], ignore_index=True)
    before, after = _compute(name, past), _compute(name, joined)
    for feature, values in before.items():
        np.testing.assert_array_equal(after[feature][: len(past)], values, err_msg=feature)


@pytest.mark.parametrize("name", ALL_MODULES)
def test_khong_phu_thuoc_thu_tu_dong(name):
    frame = _random(500, seed=4)
    shuffled = frame.sample(frac=1.0, random_state=9)
    original, permuted = _compute(name, frame), _compute(name, shuffled)
    for feature, values in original.items():
        np.testing.assert_array_equal(permuted[feature], values[shuffled.index.to_numpy()],
                                      err_msg=feature)


@pytest.mark.parametrize("name", ALL_MODULES)
def test_dung_danh_sach_cot_va_huu_han(name):
    out = _compute(name, _random(400, seed=5))
    assert set(out) == set(REGISTRY[name].FEATURES)
    for feature, values in out.items():
        assert values.dtype == np.float32 and len(values) == 400, feature
        assert np.isfinite(values).all(), feature


@pytest.mark.parametrize("name", ALL_MODULES)
def test_dong_dau_tien_khong_co_lich_su(name):
    for feature, values in _compute(name, _random(50, seed=6).iloc[:1]).items():
        if feature not in ("cb_cross_bank", "cb_log_amount_usd", "cs_cash_ratio_ctr"):
            assert values[0] in (0.0, -1.0), feature


def test_stream_khong_chua_nhan():
    frame = _random(30)
    frame["label"] = True
    frame["laundering_pattern"] = "STACK"
    stream = Stream.from_frame(frame)
    assert not any("label" in key or "pattern" in key for key in vars(stream))


# --------------------------------------------------------------------------
# Ý nghĩa từng kịch bản, tính tay
# --------------------------------------------------------------------------

def test_fan_out():
    out = _compute("fan_out", _rows([
        ("B1_a", "B1_x", 0, 100.0, "ach"), ("B1_a", "B1_y", 1, 100.0, "ach"),
        ("B1_a", "B1_x", 2, 100.0, "ach"), ("B1_a", "B1_z", 3, 100.0, "ach"),
    ]))
    np.testing.assert_allclose(out["fo_log_n_recv_24h"], np.log1p([0, 1, 2, 2]))
    np.testing.assert_allclose(out["fo_log_n_new_recv_24h"], np.log1p([0, 1, 2, 2]))
    np.testing.assert_allclose(out["fo_amt_std_24h"], [-1, -1, 0, 0])
    np.testing.assert_allclose(out["fo_amt_dev_24h"], [-1, 0, 0, 0])


def test_fan_in():
    out = _compute("fan_in", _rows([
        ("B1_a", "B1_r", 0, 100.0, "ach"), ("B1_b", "B1_r", 1, 200.0, "ach"),
        ("B1_a", "B1_r", 2, 300.0, "ach"), ("B1_c", "B1_r", 3, 1.0, "ach"),
    ]))
    np.testing.assert_allclose(out["fi_log_n_send_24h"], np.log1p([0, 1, 2, 2]))
    np.testing.assert_allclose(out["fi_log_amt_in_24h"], np.log1p([0, 100, 300, 600]), rtol=1e-6)
    np.testing.assert_allclose(out["fi_new_sender_ratio_24h"], [0, 1, 1, 2 / 3], rtol=1e-6)


def test_gather_scatter():
    out = _compute("gather_scatter", _rows([
        ("B1_x", "B1_m", 0, 100.0, "ach"), ("B1_y", "B1_m", 30, 100.0, "ach"),
        ("B1_m", "B1_p", 60, 150.0, "ach"), ("B1_m", "B1_q", 90, 50.0, "ach"),
    ]))
    np.testing.assert_allclose(out["gs_passthrough_24h"][2:], [0.0, 150 / 200], rtol=1e-6)
    np.testing.assert_allclose(out["gs_log_min_since_first_in_24h"][2:], np.log1p([60, 90]), rtol=1e-6)
    np.testing.assert_allclose(out["gs_log_deg_product_24h"][2:], np.log1p([0, 2]), rtol=1e-6)


def test_cycle():
    out = _compute("cycle", _rows([
        ("B1_a", "B1_b", 0, 100.0, "ach"), ("B1_b", "B1_a", 10, 90.0, "ach"),
        ("B1_a", "B1_c", 20, 80.0, "ach"),
    ]))
    np.testing.assert_array_equal(out["cy_reciprocity_7d"], [0, 1, 0])
    np.testing.assert_allclose(out["cy_log_min_since_reverse"], [-1, np.log1p(10), -1], rtol=1e-6)


def test_mule_chain():
    out = _compute("mule_chain", _rows([
        ("B1_x", "B1_m", 0, 1000.0, "ach"), ("B1_m", "B1_y", 10, 950.0, "ach"),
    ]))
    assert out["mc_forward_ratio_24h"][0] == -1
    assert out["mc_forward_ratio_24h"][1] == pytest.approx(0.95, rel=1e-6)
    assert out["mc_log_min_since_in"][1] == pytest.approx(np.log1p(10), rel=1e-6)
    assert 0.9 < out["mc_amt_sim_last_in"][1] < 1.0
    assert out["mc_log_age_days_cap3"][1] == pytest.approx(np.log1p(10 / 1440), rel=1e-5)


def test_mule_chain_tuoi_tai_khoan_bi_chan_3_ngay():
    """Tuổi không được tăng mãi theo đồng hồ bộ dữ liệu: quá 3 ngày thì như nhau."""
    out = _compute("mule_chain", _rows([
        ("B1_a", "B1_b", 0, 10.0, "ach"),
        ("B1_a", "B1_c", 60 * 24 * 2, 10.0, "ach"),
        ("B1_a", "B1_d", 60 * 24 * 5, 10.0, "ach"),
        ("B1_a", "B1_e", 60 * 24 * 9, 10.0, "ach"),
    ]))
    age = out["mc_log_age_days_cap3"]
    np.testing.assert_allclose(age[:2], [0.0, np.log1p(2)], rtol=1e-6)
    np.testing.assert_allclose(age[2:], [np.log1p(3), np.log1p(3)], rtol=1e-6)


def test_synchronized_dem_so_tien_giong_nhau():
    out = _compute("synchronized", _rows([
        ("B1_a", "B1_r", 0, 1000.0, "ach"), ("B1_b", "B1_r", 1, 1000.0, "ach"),
        ("B1_c", "B1_r", 2, 9000.0, "ach"),
    ]))
    np.testing.assert_allclose(out["sy_log_same_amt_in_6h"], np.log1p([0, 1, 0]), rtol=1e-6)
    np.testing.assert_allclose(out["sy_log_n_send_6h"], np.log1p([0, 1, 2]), rtol=1e-6)


def test_hub_tinh_ca_hai_chieu():
    out = _compute("hub", _rows([
        ("B1_a", "B1_h", 0, 1.0, "ach"), ("B1_h", "B1_b", 1, 1.0, "ach"),
        ("B1_c", "B1_h", 2, 1.0, "ach"),
    ]))
    np.testing.assert_allclose(out["hub_log_deg_receiver_7d"], np.log1p([0, 0, 2]), rtol=1e-6)
    np.testing.assert_allclose(out["hub_log_deg_sender_7d"], np.log1p([0, 1, 0]), rtol=1e-6)


def test_cross_bank():
    out = _compute("cross_bank_fx", _rows([
        ("B1_a", "B1_b", 0, 1.0, "ach"), ("B1_a", "B2_c", 1, 1.0, "ach"),
        ("B1_a", "B3_d", 2, 1.0, "ach"),
    ]))
    np.testing.assert_array_equal(out["cb_cross_bank"], [0, 1, 1])
    np.testing.assert_allclose(out["cb_log_n_banks_7d"], np.log1p([0, 1, 2]), rtol=1e-6)


def test_cash_structuring():
    out = _compute("cash_structuring", _rows([
        ("B1_a", "B1_x", 0, 9000.0, "cash"), ("B1_a", "B1_y", 1, 9500.0, "cash"),
        ("B1_a", "B1_z", 2, 100.0, "ach"),
    ]))
    np.testing.assert_allclose(out["cs_cash_ratio_ctr"], [0.9, 1.85, 1.85], rtol=1e-6)
    np.testing.assert_allclose(out["cs_log_n_near_ctr_7d"], np.log1p([0, 1, 2]), rtol=1e-6)
    np.testing.assert_allclose(out["cs_log_n_channels_7d"], np.log1p([0, 1, 1]), rtol=1e-6)


def test_integration_khoan_lon_lan_dau():
    rows = [("B1_a", "B1_x", i, 100.0, "ach") for i in range(6)]
    rows.append(("B1_a", "B1_new", 6, 100000.0, "ach"))
    out = _compute("integration", _rows(rows))
    assert out["in_large_first_time"][-1] == 1
    assert out["in_large_first_time"][:-1].sum() == 0
    assert out["in_log_ratio_prior_max"][-1] > 5


def test_temporal_windows():
    out = _compute("temporal_windows", _rows([
        ("B1_a", "B1_x", 0, 100.0, "ach"), ("B1_a", "B1_x", 30, 100.0, "ach"),
        ("B1_a", "B1_x", 120, 100.0, "ach"),
    ]))
    np.testing.assert_allclose(out["tw_log_n_out_1h"], np.log1p([0, 1, 0]), rtol=1e-6)
    np.testing.assert_allclose(out["tw_log_n_out_24h"], np.log1p([0, 1, 2]), rtol=1e-6)
    np.testing.assert_allclose(out["tw_log_n_recv_24h"], np.log1p([0, 1, 2]), rtol=1e-6)


# --------------------------------------------------------------------------
# Cờ bật/tắt
# --------------------------------------------------------------------------

def test_co_bat_tat():
    assert resolve("none") == []
    assert resolve("all") == list(REGISTRY)
    assert resolve("fan_in,cycle") == ["fan_in", "cycle"]
    assert "hub" not in resolve("all,-hub")
    with pytest.raises(ValueError, match="khong_co"):
        resolve("khong_co")


def test_khong_trung_ten_giua_cac_mo_dun():
    features = features_of(list(REGISTRY))
    assert len(features) == len(set(features))


def test_du_11_kich_ban_va_cua_so():
    assert len(scenarios.MODULES) == 12


def test_khong_dung_is_fx():
    assert not any("is_fx" in feature for feature in features_of(list(REGISTRY)))
