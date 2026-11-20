"""Kiểm thử kho tri thức kiểu hình + truy xuất vector + sinh lời giải thích.

Ngoài các phép kiểm thông thường, file này còn ĐO hai giới hạn đã khai báo
trong `typology_kb` thay vì chỉ ghi chúng trong tài liệu:
  - GATHER-SCATTER và SCATTER-GATHER có chữ ký gần trùng nhau;
  - RANDOM không có chữ ký nên không khớp được bằng đường chữ ký.
Nếu ai đó sửa chữ ký làm hai điều này sai đi, test sẽ báo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.loaders import AMLWORLD_TYPOLOGIES
from src.knowledge import (
    SIGNATURE_FEATURES,
    TYPOLOGIES,
    TypologyRetriever,
    build_narrative,
    knowledge_documents,
    typology_by_name,
)
from src.knowledge.narrative import MATCH_THRESHOLD
from src.knowledge.retriever import signals_to_query


# --------------------------------------------------------------------------
# Kho tri thức
# --------------------------------------------------------------------------

def test_kho_tri_thuc_phu_dung_8_kieu_hinh_cua_amlworld():
    """Tên kiểu hình phải khớp đúng nhãn AMLWorld, nếu không thì không ghép được."""
    assert {t.name for t in TYPOLOGIES} == set(AMLWORLD_TYPOLOGIES)


def test_moi_kieu_hinh_co_du_mo_ta_va_dau_hieu():
    for item in TYPOLOGIES:
        assert item.vi_name and item.summary and item.mechanism
        assert len(item.red_flags) >= 2, f"{item.name} quá ít dấu hiệu cần kiểm"


def test_tra_kieu_hinh_theo_ten_va_bao_loi_ro_rang():
    assert typology_by_name("CYCLE").vi_name == "Dòng tiền vòng tròn"
    with pytest.raises(KeyError, match="FAN-OUTT"):
        typology_by_name("FAN-OUTT")


def test_chu_ky_chi_dung_dac_trung_co_that_trong_pipeline():
    """Chữ ký tham chiếu đặc trưng nào thì đặc trưng đó phải tồn tại thật."""
    from src.data.features import ALL_FEATURES

    for item in TYPOLOGIES:
        for feature in item.signature:
            assert feature in SIGNATURE_FEATURES
            assert feature in ALL_FEATURES, f"{feature} không có trong pipeline"


def test_gioi_han_da_khai_bao_gather_scatter_gan_trung_nhau():
    """Hai kiểu hình này được khai báo là không phân biệt được — phải đo đúng vậy."""
    def vector(name: str) -> np.ndarray:
        item = typology_by_name(name)
        return np.array([item.signature.get(f, 0.0) for f in SIGNATURE_FEATURES])

    a, b = vector("GATHER-SCATTER"), vector("SCATTER-GATHER")
    cosine = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    assert cosine > 0.85, (
        "Chữ ký của hai kiểu hình này đã tách ra xa nhau — nếu đó là chủ ý thì "
        "hãy bỏ distinguishable=False và cập nhật tài liệu."
    )
    assert not typology_by_name("GATHER-SCATTER").distinguishable
    assert not typology_by_name("SCATTER-GATHER").distinguishable


# --------------------------------------------------------------------------
# Đường CHỮ — truy xuất theo câu hỏi
# --------------------------------------------------------------------------

@pytest.mark.parametrize("cau_hoi, mong_doi", [
    ("nhiều người gửi tiền về một tài khoản rồi chuyển đi ngay", "FAN-IN"),
    ("một tài khoản chia tiền ra hàng chục tài khoản mới", "FAN-OUT"),
    ("tiền đi lòng vòng rồi quay về chỗ cũ", "CYCLE"),
    ("tiền đi qua nhiều lớp tài khoản nối tiếp ở nhiều ngân hàng", "STACK"),
])
def test_truy_xuat_theo_cau_hoi_tieng_viet(cau_hoi, mong_doi):
    hits = TypologyRetriever().search(cau_hoi, k=3)
    assert hits[0].name == mong_doi, f"{cau_hoi!r} -> {[h.name for h in hits]}"


def test_cau_hoi_rong_tra_ve_rong_thay_vi_ket_qua_bua():
    assert TypologyRetriever().search("   ") == []


def test_so_tai_lieu_bang_so_kieu_hinh():
    assert len(knowledge_documents()) == len(TYPOLOGIES)


# --------------------------------------------------------------------------
# Đường SỐ — chữ ký và chuẩn hóa
# --------------------------------------------------------------------------

@pytest.fixture
def khung_tham_chieu() -> pd.DataFrame:
    """Khung giả lập với phân phối đã biết, để z-score tính ra số tròn."""
    rng = np.random.default_rng(0)
    n = 500
    return pd.DataFrame({
        f: rng.normal(loc=0.0, scale=1.0, size=n) for f in SIGNATURE_FEATURES
    })


def test_phai_fit_tham_chieu_truoc_khi_chuan_hoa():
    with pytest.raises(RuntimeError, match="fit_reference"):
        TypologyRetriever().z_scores({"log_n_prior": 1.0})


def test_fit_tham_chieu_bao_loi_khi_khung_khong_co_dac_trung_nao():
    with pytest.raises(ValueError, match="đặc trưng chữ ký"):
        TypologyRetriever().fit_reference(pd.DataFrame({"khong_lien_quan": [1.0, 2.0]}))


def test_giao_dich_binh_thuong_khong_sinh_cau_truy_van(khung_tham_chieu):
    r = TypologyRetriever().fit_reference(khung_tham_chieu)
    binh_thuong = {f: 0.0 for f in SIGNATURE_FEATURES}
    assert signals_to_query(r.z_scores(binh_thuong)) == ""


def test_tin_hieu_manh_sinh_cau_truy_van_va_khop_fan_in(khung_tham_chieu):
    r = TypologyRetriever().fit_reference(khung_tham_chieu)
    fan_in = {f: 0.0 for f in SIGNATURE_FEATURES}
    fan_in["log_n_distinct_senders"] = 4.0
    fan_in["log_n_prior_as_cp"] = 4.0
    z = r.z_scores(fan_in)
    assert "nhận tiền từ rất nhiều người gửi khác nhau" in signals_to_query(z)
    assert r.match_by_signals(fan_in, k=1)[0].name == "FAN-IN"


def test_cong_hien_thi_dong_lai_khi_khong_co_tin_hieu(khung_tham_chieu):
    """`is_unusual` là cổng HIỂN THỊ, tách khỏi phép xếp hạng.

    Vì sao tách: nhét cổng vào thẳng `match_by_signals` làm top-3 của đường
    chữ ký rơi từ 46,4% xuống 10,8% trên val AMLWorld — 4/5 giao dịch rửa
    tiền không có đặc trưng nào ở đuôi cực trị nên bị tính là trượt, dù thứ
    hạng của chúng vẫn đúng. Xếp hạng luôn trả lời được; chỉ việc hiển thị
    mới cần dè dặt.
    """
    r = TypologyRetriever().fit_reference(khung_tham_chieu)
    binh_thuong = {f: 0.0 for f in SIGNATURE_FEATURES}
    assert not r.is_unusual(binh_thuong)
    # Xếp hạng vẫn trả lời, vì đó không phải việc của nó để từ chối
    assert r.match_by_signals(binh_thuong) != []


def test_cong_hien_thi_mo_khi_co_dac_trung_o_duoi_cuc_tri(khung_tham_chieu):
    r = TypologyRetriever().fit_reference(khung_tham_chieu)
    bat_thuong = {f: 0.0 for f in SIGNATURE_FEATURES}
    bat_thuong["log_n_distinct_senders"] = 4.0
    assert r.is_unusual(bat_thuong)


def test_phai_fit_tham_chieu_truoc_khi_hoi_cong(khung_tham_chieu):
    with pytest.raises(RuntimeError, match="fit_reference"):
        TypologyRetriever().is_unusual({f: 0.0 for f in SIGNATURE_FEATURES})


def test_random_khong_bao_gio_duoc_khop_bang_chu_ky(khung_tham_chieu):
    """RANDOM không có chữ ký, nên phải nhận điểm 0 chứ không phải một số giả."""
    r = TypologyRetriever().fit_reference(khung_tham_chieu)
    manh = {f: 3.0 for f in SIGNATURE_FEATURES}
    diem = {h.name: h.score for h in r.match_by_signals(manh, k=len(TYPOLOGIES))}
    assert diem.get("RANDOM", 0.0) == pytest.approx(0.0)


# --------------------------------------------------------------------------
# Bộ phân loại học từ train
# --------------------------------------------------------------------------

def _khung_co_nhan(n_moi_lop: int = 40) -> pd.DataFrame:
    """Khung train giả lập: hai kiểu hình tách bạch trên đúng một đặc trưng."""
    from src.data.features import ALL_FEATURES

    rng = np.random.default_rng(7)
    rows = []
    for kieu, dich in (("FAN-IN", +3.0), ("FAN-OUT", -3.0)):
        for _ in range(n_moi_lop):
            row = {f: float(rng.normal()) for f in ALL_FEATURES}
            row["log_n_distinct_senders"] = float(rng.normal(loc=dich))
            row["laundering_pattern"] = kieu
            rows.append(row)
    return pd.DataFrame(rows)


def test_hoc_bo_phan_loai_bao_loi_khi_qua_it_nhan():
    with pytest.raises(ValueError, match="quá ít để học"):
        TypologyRetriever().fit_supervised(_khung_co_nhan(n_moi_lop=5))


def test_hoc_bo_phan_loai_bao_loi_khi_thieu_cot_nhan():
    with pytest.raises(KeyError, match="laundering_pattern"):
        TypologyRetriever().fit_supervised(pd.DataFrame({"amount_norm": [1.0] * 60}))


def test_sau_khi_hoc_thi_match_dung_duong_hoc_may(khung_tham_chieu):
    khung = _khung_co_nhan()
    r = TypologyRetriever().fit_reference(khung_tham_chieu).fit_supervised(khung)
    row = {f: 0.0 for f in SIGNATURE_FEATURES}
    row["log_n_distinct_senders"] = 3.0
    hits = r.match_by_signals(row, k=2)
    assert hits[0].name == "FAN-IN"
    # Đường học máy trả XÁC SUẤT nên tổng phải nằm trong [0, 1] từng cái một
    assert all(0.0 <= h.score <= 1.0 for h in hits)


# --------------------------------------------------------------------------
# Sinh lời giải thích
# --------------------------------------------------------------------------

def test_tin_hieu_yeu_thi_noi_thang_la_khong_khop(khung_tham_chieu):
    r = TypologyRetriever().fit_reference(khung_tham_chieu)
    ket_qua = build_narrative({f: 0.0 for f in SIGNATURE_FEATURES}, r)
    assert ket_qua.typology is None
    assert "lệch đủ mạnh" in ket_qua.headline.lower()
    assert ket_qua.next_steps == []


def test_loi_giai_thich_chi_lay_noi_dung_tu_kho_tri_thuc(khung_tham_chieu):
    """Cam kết 'không bịa': mọi câu phải truy được về kho tri thức."""
    r = TypologyRetriever().fit_reference(khung_tham_chieu)
    row = {f: 0.0 for f in SIGNATURE_FEATURES}
    row["log_n_distinct_senders"] = 4.0
    row["log_n_prior_as_cp"] = 4.0
    ket_qua = build_narrative(row, r, risk_score=91.0)

    goc = typology_by_name(ket_qua.typology)
    assert ket_qua.mechanism == goc.mechanism
    assert ket_qua.next_steps == list(goc.red_flags)
    assert ket_qua.confidence >= MATCH_THRESHOLD
    assert "91/100" in ket_qua.headline
    assert ket_qua.generator == "template"


def test_kieu_hinh_khong_phan_biet_duoc_thi_phai_noi_ra(khung_tham_chieu):
    r = TypologyRetriever().fit_reference(khung_tham_chieu)
    row = {f: 0.0 for f in SIGNATURE_FEATURES}
    row["log_n_distinct_senders"] = 3.0
    row["log_n_distinct_cp"] = 3.0
    ket_qua = build_narrative(row, r)
    if ket_qua.typology in ("GATHER-SCATTER", "SCATTER-GATHER"):
        assert "không phân biệt được chắc chắn" in ket_qua.headline


def test_duong_llm_hong_thi_rot_ve_mau_cau_chu_khong_vo(khung_tham_chieu):
    def llm_hong(_prompt: str) -> str:
        raise RuntimeError("mạng hỏng")

    r = TypologyRetriever().fit_reference(khung_tham_chieu)
    row = {f: 0.0 for f in SIGNATURE_FEATURES}
    row["log_n_distinct_senders"] = 4.0
    ket_qua = build_narrative(row, r, llm=llm_hong)
    assert ket_qua.generator == "template"
    assert ket_qua.typology is not None


def test_prompt_llm_khong_chua_dinh_danh_hay_so_tien(khung_tham_chieu):
    """Cam kết riêng tư: prompt chỉ được chứa cụm từ tín hiệu và văn bản kho."""
    da_nhan: list[str] = []

    def llm_ghi_lai(prompt: str) -> str:
        da_nhan.append(prompt)
        return "Giải thích do mô hình ngôn ngữ viết."

    r = TypologyRetriever().fit_reference(khung_tham_chieu)
    row = {f: 0.0 for f in SIGNATURE_FEATURES}
    row["log_n_distinct_senders"] = 4.0
    row["account_id"] = "B021174_800737690"
    row["amount"] = 5738987.96
    ket_qua = build_narrative(row, r, llm=llm_ghi_lai)

    assert ket_qua.generator == "llm"
    prompt = da_nhan[0]
    assert "B021174_800737690" not in prompt
    assert "5738987" not in prompt
