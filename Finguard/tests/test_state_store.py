"""Mang trạng thái Behavior Memory qua các cửa sổ (BehaviorStateStore).

Test quan trọng nhất: test_hai_cua_so_mang_trang_thai_bang_mot_cua_so_lon.
Nếu trạng thái được mang đúng thì chia một dòng dữ liệu thành hai cửa sổ phải
cho ra CÙNG embedding như chạy nó trong một cửa sổ duy nhất — đó là định nghĩa
của "bộ nhớ liên tục". Test này chạy với use_graph=False để loại ảnh hưởng của
đồ thị (đồ thị được dựng lại theo từng cửa sổ nên vốn khác nhau).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from src.dashboard.graph_utils import build_account_graph
from src.drre.model import DRRE
from src.drre.train import BehaviorStateStore, run_encoder_over_window

FEATS = ["amount_norm", "channel_encoded"]


def _window(rows: list[tuple]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["step", "account_id", "counterparty_id", "amount"])
    df["amount_norm"] = np.log1p(df["amount"])
    df["channel_encoded"] = 0.0
    return df.reset_index(drop=True)


STREAM = _window([
    (1, "A", "X", 100.0),
    (2, "B", "Y", 200.0),
    (3, "A", "Y", 300.0),
    (4, "B", "X", 400.0),
    (5, "A", "Z", 500.0),
    (6, "B", "Z", 600.0),
])


def _model() -> DRRE:
    torch.manual_seed(0)
    model = DRRE(input_dim=len(FEATS), hidden_dim=8, num_slots=3,
                 graph_hidden_dim=8, graph_num_layers=2, embedding_dim=16)
    model.eval()
    return model


def _run(model: DRRE, window: pd.DataFrame, store: BehaviorStateStore | None):
    node_features, edge_index, idx = build_account_graph(window, FEATS, use_graph=False)
    with torch.no_grad():
        return run_encoder_over_window(
            model, window, node_features, edge_index, idx, torch.device("cpu"),
            feature_columns=FEATS, state_store=store)["e"]


def test_hai_cua_so_mang_trang_thai_bang_mot_cua_so_lon():
    model = _model()
    mot_lan = _run(model, STREAM, None)

    store = BehaviorStateStore(hidden_dim=8, num_slots=3, capacity=16)
    phan_dau = _run(model, STREAM.iloc[:3].reset_index(drop=True), store)
    phan_sau = _run(model, STREAM.iloc[3:].reset_index(drop=True), store)
    hai_lan = torch.cat([phan_dau, phan_sau])

    torch.testing.assert_close(hai_lan, mot_lan, rtol=1e-5, atol=1e-6)


def test_khong_mang_trang_thai_thi_ket_qua_khac():
    """Đối chứng: không có store thì cửa sổ sau khởi tạo lại từ đầu -> khác."""
    model = _model()
    mot_lan = _run(model, STREAM, None)
    phan_dau = _run(model, STREAM.iloc[:3].reset_index(drop=True), None)
    phan_sau = _run(model, STREAM.iloc[3:].reset_index(drop=True), None)
    hai_lan = torch.cat([phan_dau, phan_sau])
    assert not torch.allclose(hai_lan, mot_lan, rtol=1e-3, atol=1e-4)


def test_luu_roi_nap_lai_dung_trang_thai():
    store = BehaviorStateStore(hidden_dim=4, num_slots=2, capacity=8)
    h = torch.arange(8.0).reshape(2, 4)
    m = torch.arange(16.0).reshape(2, 2, 4)
    store.save(["A", "B"], h, m)

    h0 = torch.zeros(3, 4)
    m0 = torch.zeros(3, 2, 4)
    h_out, m_out = store.load(["Z", "B", "A"], h0, m0)
    assert torch.equal(h_out[0], torch.zeros(4))     # Z chưa từng gặp
    assert torch.equal(h_out[1], h[1])               # B
    assert torch.equal(h_out[2], h[0])               # A
    assert torch.equal(m_out[1], m[1])
    assert store.hits == 2 and store.lookups == 3
    assert store.hit_rate == pytest.approx(2 / 3)


def test_vuot_dung_luong_thi_day_tai_khoan_cu_ra():
    store = BehaviorStateStore(hidden_dim=2, num_slots=1, capacity=2)
    store.save(["A"], torch.ones(1, 2), torch.ones(1, 1, 2))
    store.save(["B"], 2 * torch.ones(1, 2), torch.ones(1, 1, 2))
    store.save(["C"], 3 * torch.ones(1, 2), torch.ones(1, 1, 2))  # đẩy A ra
    assert store.evictions == 1
    h_out, _ = store.load(["A", "B", "C"], torch.zeros(3, 2), torch.zeros(3, 1, 2))
    assert torch.equal(h_out[0], torch.zeros(2))          # A đã bị đẩy ra
    assert torch.equal(h_out[1], 2 * torch.ones(2))
    assert torch.equal(h_out[2], 3 * torch.ones(2))


def test_reset_xoa_sach_trang_thai():
    """Bắt buộc gọi ở đầu mỗi epoch: epoch mới chạy lại từ thời điểm đầu."""
    store = BehaviorStateStore(hidden_dim=2, num_slots=1, capacity=4)
    store.save(["A"], torch.ones(1, 2), torch.ones(1, 1, 2))
    store.reset()
    h_out, _ = store.load(["A"], torch.zeros(1, 2), torch.zeros(1, 1, 2))
    assert torch.equal(h_out[0], torch.zeros(2))
    assert store.lookups == 1 and store.hits == 0


def test_tai_khoan_lap_lai_duoc_ghi_de_chu_khong_cap_them_o():
    store = BehaviorStateStore(hidden_dim=2, num_slots=1, capacity=4)
    store.save(["A"], torch.ones(1, 2), torch.ones(1, 1, 2))
    store.save(["A"], 9 * torch.ones(1, 2), torch.ones(1, 1, 2))
    assert store.evictions == 0
    h_out, _ = store.load(["A"], torch.zeros(1, 2), torch.zeros(1, 1, 2))
    assert torch.equal(h_out[0], 9 * torch.ones(2))
