"""Loader IBM AMLWorld / TransXion.

Hai bộ này dùng CHUNG schema, nên cùng một loader phải đọc được cả hai —
đó là điều kiện để train trên AMLWorld rồi kiểm định chéo trên TransXion.

Fixture dưới đây tái tạo đúng hai cái bẫy của file gốc:
  1. header có HAI cột cùng tên "Account" (bên gửi và bên nhận);
  2. số tài khoản TRÙNG nhau giữa các ngân hàng khác nhau.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.loaders import healthy_day_mask, load_amlworld
from src.data.preprocessing import run_pipeline

HEADER = ("Timestamp,From Bank,Account,To Bank,Account,Amount Received,Receiving Currency,"
          "Amount Paid,Payment Currency,Payment Format,Is Laundering")

# 8 giao dịch ngày 01/09, 2 giao dịch ngày 02/09 -> dùng để kiểm tra lọc ngày.
# Tài khoản "100" xuất hiện ở CẢ bank 1 và bank 2 — phải thành hai định danh khác nhau.
ROWS = [
    "2022/09/01 00:10,1,100,2,200,1000.00,US Dollar,1000.00,US Dollar,Cheque,0",
    "2022/09/01 01:20,1,100,2,201,2000.00,US Dollar,2000.00,US Dollar,Cheque,0",
    "2022/09/01 02:30,2,100,1,100,500.00,Euro,550.00,US Dollar,Wire,1",
    "2022/09/01 03:40,2,100,3,300,700.00,US Dollar,700.00,US Dollar,ACH,1",
    "2022/09/01 04:50,3,300,1,100,900.00,US Dollar,900.00,US Dollar,Credit Card,0",
    "2022/09/01 05:00,3,300,2,200,1100.00,US Dollar,1100.00,US Dollar,ACH,0",
    "2022/09/01 06:10,1,101,3,301,1300.00,US Dollar,1300.00,US Dollar,Cheque,0",
    "2022/09/01 07:20,1,101,3,302,1500.00,US Dollar,1500.00,US Dollar,Wire,0",
    "2022/09/02 00:30,2,201,1,101,1700.00,US Dollar,1700.00,US Dollar,ACH,1",
    "2022/09/02 01:40,2,201,3,300,1900.00,US Dollar,1900.00,US Dollar,Cheque,0",
]


@pytest.fixture()
def amlworld_csv(tmp_path):
    path = tmp_path / "HI-Tiny_Trans.csv"
    path.write_text("\n".join([HEADER, *ROWS]) + "\n", encoding="utf-8")
    return path


def test_doc_duoc_header_co_hai_cot_account_trung_ten(amlworld_csv):
    df = load_amlworld(amlworld_csv)
    assert len(df) == 10
    for col in ("account_id", "counterparty_id", "timestamp", "amount", "channel",
                "label", "source", "currency", "is_fx"):
        assert col in df.columns, col


def test_ghep_ma_ngan_hang_nen_tai_khoan_trung_so_khong_bi_gop(amlworld_csv):
    df = load_amlworld(amlworld_csv)
    # "100" ở bank 1 và "100" ở bank 2 là HAI tài khoản khác nhau
    assert {"B1_100", "B2_100"} <= set(df["account_id"])
    assert df.loc[0, "account_id"] == "B1_100"
    assert df.loc[0, "counterparty_id"] == "B2_200"


def test_nhan_va_dau_hieu_doi_tien(amlworld_csv):
    df = load_amlworld(amlworld_csv)
    assert df["label"].dtype == bool
    assert int(df["label"].sum()) == 3
    # Chỉ dòng thứ 3 có Payment Currency khác Receiving Currency
    assert df["is_fx"].tolist() == [False, False, True, False, False,
                                    False, False, False, False, False]
    assert df.loc[2, "amount"] == pytest.approx(550.0)  # dùng Amount Paid, không phải Received


def test_header_sai_thi_bao_loi_ro_rang(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b,c\n1,2,3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="schema AMLWorld"):
        load_amlworld(bad)


def test_loc_ngay_theo_luu_luong_dung_cho_nguon_dung_timestamp(amlworld_csv):
    df = load_amlworld(amlworld_csv)
    keep = healthy_day_mask(df, min_daily_volume=5)
    assert int(keep.sum()) == 8  # giữ ngày 01/09 (8 giao dịch), loại ngày 02/09 (2 giao dịch)
    assert healthy_day_mask(df, min_daily_volume=0).all()  # 0 = không lọc


def test_run_pipeline_amlworld_ghi_du_ba_tap_va_co_eval_universe(amlworld_csv, tmp_path):
    out = tmp_path / "processed"
    run_pipeline(source="amlworld", raw_path=str(amlworld_csv), output_dir=str(out))
    total = 0
    for split in ("train", "val", "test"):
        part = pd.read_parquet(out / f"amlworld_{split}.parquet")
        total += len(part)
        assert {"amount_norm", "channel_encoded", "eval_universe"} <= set(part.columns)
        # Vũ trụ chấm điểm của AMLWorld LOẠI `wire` và `reinvestment`: trên
        # toàn bộ 5.078.345 dòng HI-Small, hai hình thức này có ĐÚNG 0 nhãn
        # dương trên 652.911 dòng (12,9% dữ liệu) — cùng lỗi với PaySim, nơi
        # cash_in/payment/debit cũng có đúng 0 nhãn dương. Chấm điểm trên cả
        # 7 hình thức là cho mô hình một phần đáp án miễn phí.
        ngoai_vu_tru = part.loc[~part["eval_universe"], "channel"]
        assert set(ngoai_vu_tru) <= {"wire", "reinvestment"}
        trong_vu_tru = part.loc[part["eval_universe"], "channel"]
        assert not set(trong_vu_tru) & {"wire", "reinvestment"}
    assert total == 10


def test_run_pipeline_amlworld_ghi_tap_drift_khi_loc_ngay(amlworld_csv, tmp_path):
    out = tmp_path / "processed"
    run_pipeline(source="amlworld", raw_path=str(amlworld_csv), output_dir=str(out),
                 min_daily_volume=5)
    drift = pd.read_parquet(out / "amlworld_drift.parquet")
    assert len(drift) == 2
    kept = sum(len(pd.read_parquet(out / f"amlworld_{s}.parquet"))
               for s in ("train", "val", "test"))
    assert kept == 8


# --------------------------------------------------------------------------
# Nhãn kiểu hình rửa tiền — {bộ}_Patterns.txt
# --------------------------------------------------------------------------

# Ba cái bẫy của file Patterns thật, tái tạo đủ cả ba:
#   1. có kiểu hình CÓ mô tả sau dấu hai chấm, có kiểu hình KHÔNG;
#   2. tên kiểu hình chứa dấu gạch nối (FAN-OUT, GATHER-SCATTER);
#   3. các dòng giao dịch dùng ĐÚNG 11 cột như Trans.csv.
PATTERNS_TEXT = "\n".join([
    "BEGIN LAUNDERING ATTEMPT - FAN-OUT:  Max 2-degree Fan-Out",
    "2022/09/01 02:30,2,100,1,100,500.00,Euro,550.00,US Dollar,Wire,1",
    "2022/09/01 03:40,2,100,3,300,700.00,US Dollar,700.00,US Dollar,ACH,1",
    "END LAUNDERING ATTEMPT - FAN-OUT",
    "BEGIN LAUNDERING ATTEMPT - BIPARTITE",
    "2022/09/02 00:30,2,201,1,101,1700.00,US Dollar,1700.00,US Dollar,ACH,1",
    "END LAUNDERING ATTEMPT - BIPARTITE",
])


@pytest.fixture()
def patterns_txt(tmp_path):
    path = tmp_path / "HI-Tiny_Patterns.txt"
    path.write_text(PATTERNS_TEXT + "\n", encoding="utf-8")
    return path


def test_doc_duoc_ca_kieu_hinh_co_va_khong_co_mo_ta(patterns_txt):
    from src.data.loaders import parse_amlworld_patterns

    pat = parse_amlworld_patterns(patterns_txt)
    assert len(pat) == 3
    assert set(pat["laundering_pattern"]) == {"FAN-OUT", "BIPARTITE"}
    assert pat.loc[pat.laundering_pattern == "FAN-OUT", "pattern_detail"].iloc[0] == \
        "Max 2-degree Fan-Out"
    assert pat.loc[pat.laundering_pattern == "BIPARTITE", "pattern_detail"].iloc[0] == ""
    assert pat["pattern_id"].nunique() == 2


def test_dinh_danh_ghep_dung_quy_tac_bank_account(patterns_txt):
    """Phải ra "B<bank>_<account>" y như loader, nếu không thì không ghép được."""
    from src.data.loaders import parse_amlworld_patterns

    pat = parse_amlworld_patterns(patterns_txt)
    assert pat["account_id"].iloc[0] == "B2_100"
    assert pat["counterparty_id"].iloc[0] == "B1_100"


def test_dong_khong_phai_rua_tien_trong_file_patterns_thi_bao_loi(tmp_path):
    """Giả định 'mọi dòng ở đây đều là rửa tiền' là nền tảng của phép ghép —
    nếu file vi phạm thì phải nổ, không được ghép thầm lặng rồi bỏ sót."""
    from src.data.loaders import parse_amlworld_patterns

    path = tmp_path / "xau.txt"
    path.write_text("\n".join([
        "BEGIN LAUNDERING ATTEMPT - CYCLE:  Max 2 hops",
        "2022/09/01 02:30,2,100,1,100,500.00,Euro,550.00,US Dollar,Wire,0",
        "END LAUNDERING ATTEMPT - CYCLE",
    ]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="is_laundering != 1"):
        parse_amlworld_patterns(path)


def test_dong_sai_so_cot_thi_bao_loi_kem_so_dong(tmp_path):
    from src.data.loaders import parse_amlworld_patterns

    path = tmp_path / "thieu_cot.txt"
    path.write_text("\n".join([
        "BEGIN LAUNDERING ATTEMPT - STACK",
        "2022/09/01 02:30,2,100,1,100",
        "END LAUNDERING ATTEMPT - STACK",
    ]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Dòng 2"):
        parse_amlworld_patterns(path)


def test_gan_kieu_hinh_chi_cham_vao_dong_co_nhan_duong(amlworld_csv, patterns_txt):
    from src.data.loaders import attach_laundering_pattern, load_amlworld, parse_amlworld_patterns

    df = load_amlworld(amlworld_csv)
    out = attach_laundering_pattern(df, parse_amlworld_patterns(patterns_txt))

    am = ~out["label"].fillna(False).astype(bool)
    assert out.loc[am, "laundering_pattern"].isna().all(), \
        "dòng không phải rửa tiền phải để trống, không được gán kiểu hình"

    duong = out.loc[~am, "laundering_pattern"].astype(str)
    assert sorted(duong) == ["BIPARTITE", "FAN-OUT", "FAN-OUT"]


def test_nhan_duong_khong_nam_trong_file_patterns_thi_la_unspecified(amlworld_csv, tmp_path):
    """File Patterns thật chỉ liệt kê 3.209/5.177 nhãn dương — phần dư phải
    được gọi tên chứ không lặng lẽ gán vào một kiểu hình nào đó."""
    from src.data.loaders import (
        PATTERN_UNSPECIFIED, attach_laundering_pattern, load_amlworld, parse_amlworld_patterns,
    )

    path = tmp_path / "chi_mot_mau.txt"
    path.write_text("\n".join([
        "BEGIN LAUNDERING ATTEMPT - FAN-OUT:  Max 2-degree Fan-Out",
        "2022/09/01 02:30,2,100,1,100,500.00,Euro,550.00,US Dollar,Wire,1",
        "END LAUNDERING ATTEMPT - FAN-OUT",
    ]) + "\n", encoding="utf-8")

    out = attach_laundering_pattern(load_amlworld(amlworld_csv), parse_amlworld_patterns(path))
    duong = out.loc[out["label"].fillna(False).astype(bool), "laundering_pattern"].astype(str)
    assert sorted(duong) == ["FAN-OUT", PATTERN_UNSPECIFIED, PATTERN_UNSPECIFIED]


def test_run_pipeline_gan_duoc_kieu_hinh_qua_patterns_path(amlworld_csv, patterns_txt, tmp_path):
    out = tmp_path / "processed"
    run_pipeline(source="amlworld", raw_path=str(amlworld_csv), output_dir=str(out),
                 patterns_path=str(patterns_txt))
    phan = pd.concat([pd.read_parquet(out / f"amlworld_{s}.parquet")
                      for s in ("train", "val", "test")], ignore_index=True)
    assert "laundering_pattern" in phan.columns
    assert set(phan["laundering_pattern"].dropna().astype(str)) <= {"FAN-OUT", "BIPARTITE"}


# --------------------------------------------------------------------------
# Định danh Categorical và bộ dựng đồ thị
# --------------------------------------------------------------------------

def test_dung_do_thi_duoc_voi_dinh_danh_categorical(amlworld_csv):
    """Loader để định danh dưới dạng Categorical để tiết kiệm bộ nhớ, nhưng
    `Series.map()` trên Categorical trả về một Categorical MỚI có categories
    là số nguyên — `.to_numpy()` khi đó cho mảng dtype=object và numpy từ
    chối dùng nó làm chỉ số:

        IndexError: arrays used as indices must be of integer (or boolean) type

    Lỗi nằm im tới lần đầu chạy DRRE trên AMLWorld (2026-10-20) vì PaySim và
    dữ liệu tổng hợp đều để định danh là chuỗi thường. Test này giữ đường đó
    luôn mở.
    """
    pytest.importorskip("torch")
    import numpy as np

    from src.data.features import add_causal_features, features_for_source
    from src.data.preprocessing import normalize_amount
    from src.dashboard.graph_utils import build_account_graph

    from src.data.graph_features import add_graph_features

    df = normalize_amount(load_amlworld(amlworld_csv))
    df = add_causal_features(df, "amlworld")
    df = add_graph_features(df, "amlworld")
    df["channel_encoded"] = 0.0
    feats = features_for_source("amlworld")

    assert isinstance(df["account_id"].dtype, pd.CategoricalDtype), \
        "loader phải giữ định danh ở dạng Categorical — đó là lý do test này tồn tại"

    node_features, edge_index, account_to_idx = build_account_graph(df, feats)
    assert edge_index.dtype.is_floating_point is False
    assert int(edge_index.max()) < len(account_to_idx)
    assert node_features.shape == (len(account_to_idx), len(feats))
    assert np.isfinite(node_features.numpy()).all()


def test_dinh_danh_la_khong_co_trong_do_thi_thi_bao_loi_ro_rang():
    """Thông báo phải chỉ ra NGUYÊN NHÂN (hai tập dòng khác nhau), không chỉ KeyError."""
    from src.dashboard.graph_utils import _to_node_index

    series = pd.Series(pd.Categorical(["B1_100", "B2_200", "B3_300"]))
    with pytest.raises(KeyError, match="CÙNG một tập dòng"):
        _to_node_index(series, {"B1_100": 0, "B2_200": 1}, "account_id")
