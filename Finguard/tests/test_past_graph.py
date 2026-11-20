"""Đồ thị chỉ gồm cạnh quá khứ (Giai đoạn 2B1, src/drre/past_graph.py).

Bất biến:
  * đồ thị con L bước cho embedding TRÙNG KHÍT đồ thị đầy đủ của khoảng quá khứ;
  * thêm giao dịch tương lai không đổi điểm của giao dịch cũ (đồ thị cửa sổ cũ thì đổi);
  * chạy hai cửa sổ liền nhau qua bộ đệm = chạy một cửa sổ gộp;
  * cạnh quá `lookback` bị bỏ; cạnh cùng thời điểm với đầu khối không được thấy;
  * train với --graph past rồi nạp lại và chấm ở mọi đường được.
"""

from __future__ import annotations

import random

import numpy as np
import pandas as pd
import pytest
import torch

from src.dashboard.graph_utils import build_account_graph
from src.drre.model_v2 import DRREv2
from src.drre.past_graph import PastGraphBuffer, encode_rows
from src.drre.train import run_encoder_over_window

DEVICE = torch.device("cpu")
FEATS = ["amount_norm", "channel_encoded", "log_n_prior", "log_gap", "new_counterparty",
         "log_n_distinct_cp", "log_n_distinct_senders", "cb_log_n_banks_7d", "cy_reciprocity_7d"]


def _stream(n: int = 120, n_accounts: int = 12, seed: int = 0) -> pd.DataFrame:
    """Giao dịch dày đặc giữa ít tài khoản, `step` tăng dần (có trùng thời điểm)."""
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        a, b = rng.sample(range(n_accounts), 2)
        row = {"account_id": f"A{a}", "counterparty_id": f"A{b}", "step": i // 2,
               "label": float(i % 11 == 0)}
        row.update({c: rng.uniform(-2, 30) for c in FEATS})
        row["channel_encoded"] = float(i % 3)
        rows.append(row)
    return pd.DataFrame(rows)


def _model(layers: int = 2, seed: int = 0) -> DRREv2:
    torch.manual_seed(seed)
    model = DRREv2(input_dim=len(FEATS), feature_columns=FEATS, hidden_dim=8, num_slots=2,
                   graph_hidden_dim=8, graph_num_layers=layers, embedding_dim=12, branch_dim=8)
    model.fit_normalizer(_stream()[FEATS].to_numpy())
    model.graph_mode, model.graph_lookback_hours, model.graph_block = "past", 1000.0, 16
    model.feature_columns = FEATS
    return model.eval()


def _buffer(model, lookback=1000.0, block=16) -> PastGraphBuffer:
    return PastGraphBuffer(FEATS, lookback, block, model.graph_encoder.num_layers)


@pytest.mark.parametrize("layers", [1, 2, 3])
def test_do_thi_con_trung_do_thi_day_du(layers):
    model = _model(layers)
    df = _stream()
    graphs = _buffer(model).window_graphs(df)
    got = encode_rows(model, graphs, len(df), DEVICE)
    for g in graphs:
        tau = df["step"].iloc[g.start]
        past = df[df["step"] < tau].reset_index(drop=True)
        rows = df.iloc[g.start:g.stop]
        if past.empty:
            continue
        # Đồ thị đầy đủ của khoảng quá khứ, thêm các tài khoản gửi của khối làm nút cô lập.
        filler = rows[["account_id"]].assign(counterparty_id=rows["account_id"])
        nf, ei, idx = build_account_graph(past, FEATS)
        extra = [a for a in filler["account_id"].unique() if a not in idx]
        for a in extra:
            idx[a] = len(idx)
        nf = torch.cat([nf, torch.zeros(len(extra), nf.shape[1])])
        full = model.encode_graph(nf, ei)
        want = full[torch.as_tensor([idx[a] for a in rows["account_id"]])]
        torch.testing.assert_close(got[g.start:g.stop], want, rtol=1e-5, atol=1e-6)


def test_them_giao_dich_tuong_lai_khong_doi_diem_cu():
    model = _model()
    df = _stream(n=160)
    n = 100

    def past_scores(frame):
        buf = _buffer(model)
        return run_encoder_over_window(model, frame, None, None, None, DEVICE,
                                       feature_columns=FEATS, graphs=buf.window_graphs(frame))["e"]

    torch.testing.assert_close(past_scores(df.iloc[:n].reset_index(drop=True)),
                               past_scores(df)[:n])

    # Đồ thị cửa sổ cũ thì NHÌN TRƯỚC: cùng phép thử phải thấy điểm cũ bị đổi.
    def window_scores(frame):
        nf, ei, idx = build_account_graph(frame, FEATS)
        return run_encoder_over_window(model, frame, nf, ei, idx, DEVICE, feature_columns=FEATS)["e"]

    assert not torch.allclose(window_scores(df.iloc[:n].reset_index(drop=True)),
                              window_scores(df)[:n])


def test_hai_cua_so_qua_bo_dem_bang_mot_cua_so_gop():
    model = _model()
    df = _stream(n=160)
    first, second = df.iloc[:64].reset_index(drop=True), df.iloc[64:].reset_index(drop=True)
    whole = encode_rows(model, _buffer(model).window_graphs(df), len(df), DEVICE)
    buf = _buffer(model)
    buf.append(first)
    split = encode_rows(model, buf.window_graphs(second), len(second), DEVICE)
    torch.testing.assert_close(split, whole[64:])


def test_lookback_va_canh_cung_thoi_diem():
    df = pd.DataFrame({
        "account_id": ["X", "Y", "Z", "W"], "counterparty_id": ["T", "T", "T", "T"],
        "step": [0, 5, 9, 9], **{c: [1.0, 2.0, 3.0, 4.0] for c in FEATS}})
    buf = PastGraphBuffer(FEATS, lookback_hours=6, block=1, num_hops=1)
    graphs = buf.window_graphs(df)
    # Khối của dòng thứ 4 (step 9): thấy Y (step 5), không thấy X (quá 6 giờ), không thấy Z
    # (cùng step 9 — không chặt chẽ trước).
    last = graphs[3]
    assert last.edge_index.shape[1] == 0           # W chưa nhận từ ai
    buf_t = PastGraphBuffer(FEATS, lookback_hours=6, block=1, num_hops=1)
    probe = pd.concat([df, pd.DataFrame({"account_id": ["T"], "counterparty_id": ["Q"], "step": [9],
                                         **{c: [0.0] for c in FEATS}})], ignore_index=True)
    g = buf_t.window_graphs(probe)[4]
    assert g.edge_index.shape[1] == 1              # chỉ cạnh Y -> T
    # Bộ đệm bỏ cạnh quá cũ sau khi nạp
    buf.append(df)
    assert len(buf) == 3


def test_cua_so_nguoc_thoi_gian_bi_tu_choi():
    buf = PastGraphBuffer(FEATS, 10, 4, 2)
    df = _stream(n=20)
    buf.append(df.iloc[10:].reset_index(drop=True))
    with pytest.raises(ValueError):
        buf.window_graphs(df.iloc[:10].reset_index(drop=True))


def test_luu_va_nap_bo_dem():
    model = _model()
    df = _stream(n=120)
    a = _buffer(model)
    a.append(df.iloc[:60].reset_index(drop=True))
    b = _buffer(model)
    b.load_state_dict(a.state_dict())
    rest = df.iloc[60:].reset_index(drop=True)
    torch.testing.assert_close(encode_rows(model, a.window_graphs(rest), len(rest), DEVICE),
                               encode_rows(model, b.window_graphs(rest), len(rest), DEVICE))


# --------------------------------------------------------------------------
# Huấn luyện đầu-cuối
# --------------------------------------------------------------------------

def _processed(tmp_path):
    from tests.test_drre_train import _make_synthetic_processed_df

    out = tmp_path / "processed"
    out.mkdir()
    train_df = _make_synthetic_processed_df(n_transactions=300, seed=1)
    val_df = _make_synthetic_processed_df(n_transactions=80, seed=2)
    # Val nối tiếp train theo thời gian, như dữ liệu thật.
    val_df["step"] += 300
    val_df["timestamp"] += pd.Timedelta(hours=300)
    train_df.to_parquet(out / "synthetic_vn_train.parquet")
    val_df.to_parquet(out / "synthetic_vn_val.parquet")
    return out


def test_train_do_thi_qua_khu_roi_cham_diem(tmp_path):
    from src.dashboard.scoring_service import score_batch, try_load_drre_checkpoint
    from src.drre.past_graph import make_buffer
    from src.drre.train import BehaviorStateStore, train
    from src.evaluation.stateful_scoring import score_split, time_sorted, warm_up_store

    processed = _processed(tmp_path)
    checkpoint = tmp_path / "ck.pt"
    common = dict(source="synthetic_vn", epochs=2, batch_size=64, lr=1e-3, num_negatives=3,
                  processed_dir=str(processed), seed=42, runs_dir=None, arch="v2")
    with pytest.raises(ValueError):
        train(checkpoint_path=str(checkpoint), graph="past", **common)   # thiếu carry_state
    train(checkpoint_path=str(checkpoint), graph="past", graph_lookback_hours=48,
          graph_block=16, carry_state=True, **common)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert saved["model_config"]["graph_mode"] == "past"
    model, engine = try_load_drre_checkpoint(str(checkpoint))
    assert model.graph_mode == "past" and model.graph_block == 16

    train_df = time_sorted(pd.read_parquet(processed / "synthetic_vn_train.parquet"))
    val_df = time_sorted(pd.read_parquet(processed / "synthetic_vn_val.parquet"))
    store = BehaviorStateStore(model.behavior_memory.hidden_dim, model.behavior_memory.num_slots,
                               capacity=100)
    buf = make_buffer(model)
    warm_up_store(model, engine, train_df, store, None, {}, window=64, graph_buffer=buf)
    scores = score_split(model, engine, val_df, store, window=64, graph_buffer=buf)
    assert scores.shape == (len(val_df),) and np.isfinite(scores).all()

    dashboard = score_batch(val_df, str(checkpoint))
    assert dashboard["method"].iloc[0] == "drre_trained"
    assert np.isfinite(dashboard["risk_score"]).all()


# --------------------------------------------------------------------------
# Chế độ "edge" (Giai đoạn 2B2–2B5, R5)
# --------------------------------------------------------------------------

def _edge_model(layers: int = 3, seed: int = 0, fanout: int = 3) -> DRREv2:
    torch.manual_seed(seed)
    model = DRREv2(input_dim=len(FEATS), feature_columns=FEATS, hidden_dim=8, num_slots=2,
                   graph_hidden_dim=8, graph_num_layers=layers, embedding_dim=12, branch_dim=8,
                   graph_arch="edge")
    model.fit_normalizer(_stream()[FEATS].to_numpy())
    model.graph_mode, model.graph_lookback_hours, model.graph_block = "past", 1000.0, 16
    model.graph_fanout = fanout
    model.feature_columns = FEATS
    return model.eval()


def _edge_buffer(model, lookback=1000.0, block=16) -> PastGraphBuffer:
    return PastGraphBuffer(FEATS, lookback, block, model.graph_encoder.num_layers, arch="edge",
                           fanout=model.graph_fanout)


def _full_edge_graph(df: pd.DataFrame, start: int, stop: int, fanout: int, lookback: float):
    """Đồ thị đầy đủ (mọi nút của khoảng quá khứ) dựng bằng vòng lặp thẳng, để đối chiếu."""
    from src.drre.past_graph import BlockGraph

    tau = df["step"].iloc[start]
    hist = df[df["step"] < tau]
    past = hist[hist["step"] >= tau - lookback]
    rows = df.iloc[start:stop]
    accounts = list(pd.unique(pd.concat([past["account_id"], past["counterparty_id"],
                                         rows["account_id"], rows["counterparty_id"]])))
    idx = {a: i for i, a in enumerate(accounts)}
    kept = []
    for i, r in past.iterrows():
        later_out = ((hist["account_id"] == r["account_id"]) & (hist.index > i)).sum()
        later_in = ((hist["counterparty_id"] == r["counterparty_id"]) & (hist.index > i)).sum()
        kept.append(later_out < fanout and later_in < fanout)
    kept = np.array(kept, dtype=bool)
    x = past[FEATS].to_numpy(np.float32)
    n = len(accounts)
    out_sum, in_sum = np.zeros((n, len(FEATS))), np.zeros((n, len(FEATS)))
    counts = np.zeros((n, 2))
    for k, (a, b) in enumerate(zip(past["account_id"], past["counterparty_id"])):
        out_sum[idx[a]] += x[k]
        counts[idx[a], 0] += 1
        in_sum[idx[b]] += x[k]
        counts[idx[b], 1] += 1
    sel = past[kept]
    return BlockGraph(
        start=start, stop=stop,
        node_features=torch.tensor(out_sum / np.maximum(counts[:, :1], 1), dtype=torch.float32),
        edge_index=torch.tensor([[idx[a] for a in sel["account_id"]],
                                 [idx[b] for b in sel["counterparty_id"]]], dtype=torch.long
                                ).reshape(2, -1),
        row_node=torch.tensor([idx[a] for a in rows["account_id"]]),
        row_cp_node=torch.tensor([idx[b] for b in rows["counterparty_id"]]),
        in_features=torch.tensor(in_sum / np.maximum(counts[:, 1:], 1), dtype=torch.float32),
        counts=torch.tensor(counts, dtype=torch.float32),
        edge_x=torch.tensor(x[kept]),
        edge_age=torch.tensor(((tau - sel["step"]) / lookback).to_numpy(), dtype=torch.float32),
    )


@pytest.mark.parametrize("layers,fanout", [(1, 2), (2, 3), (3, 3), (3, 100)])
def test_edge_do_thi_con_trung_do_thi_day_du(layers, fanout):
    model = _edge_model(layers, fanout=fanout)
    df = _stream(n=120)
    lookback = 20.0
    graphs = _edge_buffer(model, lookback=lookback).window_graphs(df)
    got = encode_rows(model, graphs, len(df), DEVICE)
    assert got.shape == (len(df), 16)                       # người gửi ‖ người nhận
    for g in graphs[1:]:
        want = model.encode_block(_full_edge_graph(df, g.start, g.stop, fanout, lookback), DEVICE)
        torch.testing.assert_close(got[g.start:g.stop], want, rtol=1e-5, atol=1e-5)


def test_edge_fanout_gioi_han_bac():
    # Một tài khoản trung chuyển nhận từ rất nhiều người: chỉ `fanout` cạnh vào gần nhất được giữ.
    n = 60
    df = pd.DataFrame({"account_id": [f"S{i}" for i in range(n)] + ["Q"],
                       "counterparty_id": ["HUB"] * (n + 1),
                       "step": list(range(n)) + [n], **{c: [1.0] * (n + 1) for c in FEATS}})
    buf = PastGraphBuffer(FEATS, 1000, block=n, num_hops=2, arch="edge", fanout=4)
    g = buf.window_graphs(df)[1]                             # khối của dòng cuối
    dst = g.edge_index[1]
    assert int((dst == g.row_cp_node[0]).sum()) == 4
    assert int(g.counts[g.row_cp_node[0], 1]) == n            # hồ sơ vẫn tính mọi cạnh


def test_edge_them_giao_dich_tuong_lai_khong_doi_diem_cu():
    model = _edge_model()
    df = _stream(n=160)
    n = 100

    def scores(frame):
        graphs = _edge_buffer(model).window_graphs(frame)
        return run_encoder_over_window(model, frame, None, None, None, DEVICE,
                                       feature_columns=FEATS, graphs=graphs)["e"]

    torch.testing.assert_close(scores(df.iloc[:n].reset_index(drop=True)), scores(df)[:n])


def test_edge_hai_cua_so_qua_bo_dem_bang_mot_cua_so_gop():
    model = _edge_model()
    df = _stream(n=160)
    first, second = df.iloc[:64].reset_index(drop=True), df.iloc[64:].reset_index(drop=True)
    whole = encode_rows(model, _edge_buffer(model).window_graphs(df), len(df), DEVICE)
    buf = _edge_buffer(model)
    buf.append(first)
    split = encode_rows(model, buf.window_graphs(second), len(second), DEVICE)
    torch.testing.assert_close(split, whole[64:])


def test_edge_gradient_den_bo_ma_hoa_do_thi():
    model = _edge_model().train()
    df = _stream(n=64)
    out = run_encoder_over_window(model, df, None, None, None, DEVICE, feature_columns=FEATS,
                                  graphs=_edge_buffer(model).window_graphs(df))["e"]
    out.sum().backward()
    grads = [p.grad for p in model.graph_encoder.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert any(float(g.abs().sum()) > 0 for g in grads)


def test_train_do_thi_canh_roi_cham_diem(tmp_path):
    from src.dashboard.scoring_service import score_batch, try_load_drre_checkpoint
    from src.drre.past_graph import make_buffer
    from src.drre.train import BehaviorStateStore, train
    from src.evaluation.stateful_scoring import score_split, time_sorted, warm_up_store

    processed = _processed(tmp_path)
    checkpoint = tmp_path / "ck.pt"
    common = dict(source="synthetic_vn", epochs=2, batch_size=64, lr=1e-3, num_negatives=3,
                  processed_dir=str(processed), seed=42, runs_dir=None, carry_state=True)
    with pytest.raises(ValueError):
        train(checkpoint_path=str(checkpoint), arch="v2", graph="window", graph_arch="edge",
              **common)
    train(checkpoint_path=str(checkpoint), arch="v2", graph="past", graph_arch="edge",
          graph_fanout=5, graph_layers=3, graph_lookback_hours=48, graph_block=16, **common)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert saved["model_config"]["graph_arch"] == "edge"
    assert saved["model_config"]["graph_num_layers"] == 3
    model, engine = try_load_drre_checkpoint(str(checkpoint))
    assert model.graph_arch == "edge" and model.graph_fanout == 5
    buf = make_buffer(model)
    assert buf.arch == "edge" and buf.fanout == 5 and buf.num_hops == 3

    train_df = time_sorted(pd.read_parquet(processed / "synthetic_vn_train.parquet"))
    val_df = time_sorted(pd.read_parquet(processed / "synthetic_vn_val.parquet"))
    store = BehaviorStateStore(model.behavior_memory.hidden_dim, model.behavior_memory.num_slots,
                               capacity=100)
    warm_up_store(model, engine, train_df, store, None, {}, window=64, graph_buffer=buf)
    scores = score_split(model, engine, val_df, store, window=64, graph_buffer=buf)
    assert scores.shape == (len(val_df),) and np.isfinite(scores).all()
    dashboard = score_batch(val_df, str(checkpoint))
    assert dashboard["method"].iloc[0] == "drre_trained"
