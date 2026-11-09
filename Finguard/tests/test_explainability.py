"""Kiểm thử hai đường giải thích: nội sinh cho DRRE, quy kết cho họ mô hình cây.

File này tồn tại vì một lý do cụ thể: trước 2026-10-19,
`src/explainability/explain.py` KHÔNG được file nào gọi tới và KHÔNG có test
nào chạy qua — tức là dự án có một module giải thích mà không ai biết nó
đúng hay sai, trong khi dashboard thì giải thích bằng một bộ luật viết tay.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.explainability import (
    SCORED_EXPLANATION_COLUMNS,
    FeatureAttribution,
    explain_scored_row,
    global_attribution,
)


# --------------------------------------------------------------------------
# Giải thích nội sinh cho DRRE
# --------------------------------------------------------------------------

def _dong_da_cham(**ghi_de) -> dict[str, float]:
    row = {
        "max_attention_weight": 0.65,
        "beta_transaction": 0.2,
        "beta_behavior": 0.5,
        "beta_graph": 0.3,
    }
    row.update(ghi_de)
    return row


def test_thieu_cot_thi_bao_loi_noi_ro_cot_nao():
    """Nguyên nhân gần như luôn là khung được chấm bằng nhánh quy tắc, nên
    thông báo lỗi phải dẫn người đọc tới đó chứ không chỉ nói 'KeyError'."""
    thieu = _dong_da_cham()
    thieu.pop("beta_graph")
    with pytest.raises(KeyError) as err:
        explain_scored_row(thieu)
    assert "beta_graph" in str(err.value)
    assert "drre_trained" in str(err.value)


def test_moi_cot_can_thiet_deu_do_scoring_service_sinh_ra():
    """Danh sách cột phải khớp đúng cái `_score_with_drre` ghi ra."""
    import inspect

    from src.dashboard import scoring_service

    ma_nguon = inspect.getsource(scoring_service._score_with_drre)
    for cot in SCORED_EXPLANATION_COLUMNS:
        assert f'"{cot}"' in ma_nguon, f"{cot} không còn được scoring_service sinh ra"


def test_nhan_dung_nguon_tin_hieu_chu_dao():
    ket_qua = explain_scored_row(_dong_da_cham(beta_graph=0.8, beta_behavior=0.1))
    assert "quan hệ mạng lưới tài khoản" in ket_qua.dominant_signal_summary
    assert ket_qua.gate_weights["beta_graph"] == pytest.approx(0.8)


def test_attention_thap_thi_noi_la_khong_khop_mau_hanh_vi_nao():
    thap = explain_scored_row(_dong_da_cham(max_attention_weight=0.05))
    cao = explain_scored_row(_dong_da_cham(max_attention_weight=0.90))
    assert "không khớp" in thap.behavior_deviation_summary
    assert "khớp với một mẫu hành vi đã biết" in cao.behavior_deviation_summary


def test_giu_duoc_tai_khoan_lien_quan_nhat():
    ket_qua = explain_scored_row(
        _dong_da_cham(), top_related_account="B01_123", top_related_account_weight=0.42)
    assert ket_qua.top_related_account == "B01_123"
    assert ket_qua.top_related_account_weight == pytest.approx(0.42)


# --------------------------------------------------------------------------
# Quy kết đặc trưng cho họ mô hình cây
# --------------------------------------------------------------------------

@pytest.fixture
def mo_hinh_va_du_lieu():
    """Bài toán giả: chỉ cột 0 mang thông tin, ba cột còn lại là nhiễu."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    rng = np.random.default_rng(0)
    n = 1200
    x = rng.normal(size=(n, 4))
    y = (x[:, 0] + 0.25 * rng.normal(size=n) > 1.0).astype(int)
    model = HistGradientBoostingClassifier(max_iter=60, random_state=0).fit(x, y)
    return model, x, y, ["co_tin_hieu", "nhieu_1", "nhieu_2", "nhieu_3"]


def test_hoan_vi_tim_dung_dac_trung_mang_thong_tin(mo_hinh_va_du_lieu):
    model, x, y, names = mo_hinh_va_du_lieu
    ket_qua = global_attribution(
        model, x, y, feature_names=names, method="permutation", n_repeats=3)
    assert ket_qua.method == "permutation"
    assert ket_qua.scoring == "average_precision"
    assert ket_qua.top(1)[0][0] == "co_tin_hieu"


def test_hoan_vi_can_nhan(mo_hinh_va_du_lieu):
    model, x, _, names = mo_hinh_va_du_lieu
    with pytest.raises(ValueError, match="cần nhãn"):
        global_attribution(model, x, None, feature_names=names, method="permutation")


def test_phuong_phap_la_thi_bao_loi(mo_hinh_va_du_lieu):
    model, x, y, names = mo_hinh_va_du_lieu
    with pytest.raises(ValueError, match="'auto' \\| 'permutation' \\| 'shap'"):
        global_attribution(model, x, y, feature_names=names, method="ngau_nhien")


def test_sai_so_luong_ten_dac_trung_thi_bao_loi(mo_hinh_va_du_lieu):
    model, x, y, _ = mo_hinh_va_du_lieu
    with pytest.raises(ValueError, match="3 tên đặc trưng"):
        global_attribution(model, x, y, feature_names=["a", "b", "c"])


def test_lay_mau_khi_qua_nhieu_dong(mo_hinh_va_du_lieu):
    model, x, y, names = mo_hinh_va_du_lieu
    ket_qua = global_attribution(
        model, x, y, feature_names=names, method="permutation", n_repeats=2, max_rows=300)
    assert ket_qua.n_rows == 300


def test_bang_markdown_doc_duoc():
    ket_qua = FeatureAttribution(
        method="permutation", feature_names=("a", "b"), importance=(0.4, 0.1),
        scoring="average_precision", n_rows=10)
    bang = ket_qua.as_markdown()
    assert "average_precision" in bang
    assert bang.index("`a`") < bang.index("`b`"), "phải sắp giảm dần theo đóng góp"


def test_auto_chon_duoc_phuong_phap_dung_voi_moi_truong(mo_hinh_va_du_lieu):
    from src.explainability import shap_available

    model, x, y, names = mo_hinh_va_du_lieu
    ket_qua = global_attribution(model, x, y, feature_names=names, method="auto", n_repeats=2)
    assert ket_qua.method == ("shap" if shap_available() else "permutation")
