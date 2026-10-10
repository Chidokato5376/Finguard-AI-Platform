"""Kiểm thử bản vector hóa của encoder và của việc lấy mẫu negative
(src/drre/train.py).

Bất biến quan trọng nhất: bản vector hóa phải cho CÙNG KẾT QUẢ với bản
tuần tự tham chiếu. Nếu test này đỏ thì việc tăng tốc đã làm đổi phép
tính, và mọi số liệu đo trước đó không còn so sánh được.

Lưu ý phạm vi: chỉ ENCODER là trùng khít. Việc lấy mẫu negative đã đổi từ
rng.sample sang bác bỏ có lặp (cùng phân phối, khác dãy mẫu) — xem
build_contrastive_triplets.
"""

from __future__ import annotations

import random
from collections import deque

import numpy as np
import pandas as pd
import pytest
import torch

from src.dashboard.graph_utils import build_account_graph
from src.drre.losses import DRRECompositeLoss, EMATargetEncoder
from src.drre.model import DRRE
from src.drre.train import (
    FEATURE_COLUMNS,
    SMOOTH_WINDOW,
    _cpu_state,
    _mark_saved,
    _online_candidate,
    build_contrastive_triplets,
    eval_chunk_bounds,
    evaluate_auprc,
    iter_time_windows,
    make_time_windows,
    run_encoder_over_window,
    run_encoder_over_window_sequential,
    select_epoch,
    train_on_window,
    window_bounds,
)
from src.scoring.risk_scoring import RiskScoringEngine

DEVICE = torch.device("cpu")


def _window(n_accounts: int = 6, per_account: int = 4, seed: int = 0) -> pd.DataFrame:
    """Cửa sổ có tài khoản LẶP LẠI, để encoder phải chạy nhiều bước thời gian."""
    rng = random.Random(seed)
    rows = []
    for i in range(n_accounts * per_account):
        acc = f"ACC{i % n_accounts}"
        rows.append({
            "account_id": acc,
            "counterparty_id": f"CP{(i * 7) % n_accounts}",
            "step": i,
            "amount": rng.uniform(1e5, 1e7),
            "amount_norm": rng.uniform(-2, 2),
            "channel_encoded": float(i % 3),
            "label": float(i % 9 == 0),
        })
    return pd.DataFrame(rows).reset_index(drop=True)


def _model(seed: int = 0) -> DRRE:
    torch.manual_seed(seed)
    model = DRRE(input_dim=len(FEATURE_COLUMNS), hidden_dim=16, num_slots=3,
                 graph_hidden_dim=16, graph_num_layers=2, embedding_dim=24)
    model.eval()
    return model


@pytest.mark.parametrize("use_graph", [True, False])
def test_vectorized_encoder_matches_sequential(use_graph: bool) -> None:
    window = _window()
    model = _model()
    if not use_graph:
        model.fusion.set_disabled_branches({"graph"})
    node_features, edge_index, account_to_idx = build_account_graph(
        window, FEATURE_COLUMNS, use_graph=use_graph)

    with torch.no_grad():
        reference = run_encoder_over_window_sequential(
            model, window, node_features, edge_index, account_to_idx, DEVICE)
        vectorized = run_encoder_over_window(
            model, window, node_features, edge_index, account_to_idx, DEVICE)

    for key in ("e", "pred"):
        assert vectorized[key].shape == reference[key].shape
        max_diff = float((vectorized[key] - reference[key]).abs().max())
        assert max_diff < 1e-5, f"{key} lệch {max_diff:.2e} — bản vector hóa đã đổi phép tính"


def test_vectorized_encoder_handles_single_transaction_accounts() -> None:
    """Trường hợp PaySim: gần như mọi tài khoản chỉ có 1 giao dịch trong cửa sổ."""
    window = _window(n_accounts=20, per_account=1)
    model = _model(seed=1)
    node_features, edge_index, account_to_idx = build_account_graph(
        window, FEATURE_COLUMNS, use_graph=True)
    with torch.no_grad():
        reference = run_encoder_over_window_sequential(
            model, window, node_features, edge_index, account_to_idx, DEVICE)
        vectorized = run_encoder_over_window(
            model, window, node_features, edge_index, account_to_idx, DEVICE)
    assert float((vectorized["e"] - reference["e"]).abs().max()) < 1e-5


def test_vectorized_encoder_preserves_row_order() -> None:
    """Kết quả phải align theo thứ tự hàng GỐC, không theo thứ tự xử lý."""
    window = _window()
    model = _model(seed=2)
    node_features, edge_index, account_to_idx = build_account_graph(
        window, FEATURE_COLUMNS, use_graph=False)
    model.fusion.set_disabled_branches({"graph"})
    with torch.no_grad():
        out = run_encoder_over_window(
            model, window, node_features, edge_index, account_to_idx, DEVICE)
    # Hai giao dịch đầu của cùng một tài khoản phải có embedding khác nhau
    # (trạng thái GRU đã cập nhật), còn hai giao dịch đầu tiên của hai tài
    # khoản khác nhau thì cùng trạng thái khởi tạo.
    assert not torch.allclose(out["e"][0], out["e"][6], atol=1e-6), \
        "giao dịch thứ 2 của ACC0 phải khác giao dịch thứ 1"


def test_vectorized_encoder_raises_on_account_missing_from_graph() -> None:
    window = _window(n_accounts=3, per_account=2)
    model = _model(seed=3)
    node_features, edge_index, account_to_idx = build_account_graph(
        window, FEATURE_COLUMNS, use_graph=True)
    account_to_idx.pop("ACC0")
    with pytest.raises(KeyError, match="account_to_idx"):
        run_encoder_over_window(
            model, window, node_features, edge_index, account_to_idx, DEVICE)


def test_vectorized_encoder_gradients_reach_behavior_memory() -> None:
    """index_copy phải là phép KHÔNG tại chỗ, nếu không đồ thị tính đạo hàm
    qua các bước thời gian bị cắt và Behavior Memory không học được gì."""
    window = _window()
    model = _model(seed=4)
    model.train()
    node_features, edge_index, account_to_idx = build_account_graph(
        window, FEATURE_COLUMNS, use_graph=True)

    out = run_encoder_over_window(
        model, window, node_features, edge_index, account_to_idx, DEVICE)
    out["e"].sum().backward()

    for name, param in (
        ("slot_prototypes", model.behavior_memory.slot_prototypes),
        ("gru_cell.weight_ih", model.behavior_memory.gru_cell.weight_ih),
    ):
        assert param.grad is not None, f"{name} không nhận được gradient"
        assert float(param.grad.abs().sum()) > 0, f"gradient của {name} bằng 0"


# --------------------------------------------------------------------------
# Lấy mẫu negative
# --------------------------------------------------------------------------

def test_negatives_are_always_from_other_accounts() -> None:
    window = _window(n_accounts=8, per_account=3)
    accounts = window["account_id"].to_numpy()
    anchors, positives, negatives = build_contrastive_triplets(window, 5, random.Random(0))

    assert anchors and len(anchors) == len(positives) == len(negatives)
    for anchor, negs in zip(anchors, negatives):
        assert len(negs) == 5
        assert all(accounts[j] != accounts[anchor] for j in negs), \
            "negative phải thuộc tài khoản KHÁC anchor"


def test_positives_are_same_account_but_different_row() -> None:
    window = _window(n_accounts=5, per_account=3)
    accounts = window["account_id"].to_numpy()
    anchors, positives, _ = build_contrastive_triplets(window, 3, random.Random(1))
    for anchor, positive in zip(anchors, positives):
        assert accounts[anchor] == accounts[positive]
        assert anchor != positive


def test_negatives_degenerate_when_window_has_one_account() -> None:
    """Cửa sổ chỉ có một tài khoản: không có negative hợp lệ nào. Phải suy
    biến êm (dùng chính anchor) thay vì treo hoặc lặp vô hạn."""
    window = _window(n_accounts=1, per_account=4)
    anchors, _, negatives = build_contrastive_triplets(window, 3, random.Random(2))
    assert anchors
    for anchor, negs in zip(anchors, negatives):
        assert negs == [anchor] * 3


def test_no_anchors_when_every_account_appears_once() -> None:
    window = _window(n_accounts=10, per_account=1)
    anchors, positives, negatives = build_contrastive_triplets(window, 3, random.Random(3))
    assert anchors == [] and positives == [] and negatives == []


# --------------------------------------------------------------------------
# Một bước huấn luyện đầy đủ vẫn chạy
# --------------------------------------------------------------------------

def test_train_on_window_runs_and_updates_parameters() -> None:
    window = _window(n_accounts=6, per_account=4)
    torch.manual_seed(5)
    model = DRRE(input_dim=len(FEATURE_COLUMNS), hidden_dim=16, num_slots=3,
                 graph_hidden_dim=16, graph_num_layers=2, embedding_dim=24)
    engine = RiskScoringEngine(24)
    target = EMATargetEncoder(model, 0.99)
    loss_fn = DRRECompositeLoss(pos_weight=5.0)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(engine.parameters()), lr=1e-3)

    before = model.behavior_memory.slot_prototypes.detach().clone()
    stats = train_on_window(
        model, engine, target, loss_fn, optimizer, window, DEVICE, 4, random.Random(0))

    assert np.isfinite(stats["loss"])
    assert stats["n_pred_pairs"] > 0, "cửa sổ có tài khoản lặp lại nên phải có cặp L_pred"
    assert stats["n_contrastive_anchors"] > 0
    assert not torch.allclose(before, model.behavior_memory.slot_prototypes), \
        "một bước huấn luyện phải cập nhật slot_prototypes"


def test_make_time_windows_sorted_and_reindexed() -> None:
    df = _window(n_accounts=4, per_account=5).sample(frac=1.0, random_state=0)
    windows = make_time_windows(df, 7, "step")
    assert windows
    for w in windows:
        assert list(w.index) == list(range(len(w))), "cửa sổ phải có RangeIndex 0..n-1"
        assert w["step"].is_monotonic_increasing


def test_iter_time_windows_matches_materialised_version() -> None:
    """Bản sinh theo yêu cầu phải cho ĐÚNG các cửa sổ như bản vật chất hóa,
    chỉ khác thứ tự khi có rng."""
    df = _window(n_accounts=4, per_account=5).sample(frac=1.0, random_state=1)
    expected = make_time_windows(df, 7, "step")

    n_windows, generator = iter_time_windows(df, 7, "step")
    produced = list(generator)
    assert n_windows == len(expected) == len(produced)
    for got, want in zip(produced, expected):
        pd.testing.assert_frame_equal(got, want)


def test_iter_time_windows_shuffles_without_losing_rows() -> None:
    df = _window(n_accounts=5, per_account=6)
    n_windows, generator = iter_time_windows(df, 5, "step", random.Random(0))
    produced = list(generator)
    assert n_windows == len(produced)
    total = sum(len(w) for w in produced)
    reference = sum(len(w) for w in make_time_windows(df, 5, "step"))
    assert total == reference, "trộn thứ tự không được làm mất cửa sổ nào"
    steps = sorted(int(s) for w in produced for s in w["step"])
    assert steps == sorted(int(s) for w in make_time_windows(df, 5, "step") for s in w["step"])


def test_eval_chunk_bounds_covers_every_row() -> None:
    """Chấm điểm phải phủ TOÀN BỘ tập — khác với huấn luyện, không được bỏ dòng."""
    for n, batch in ((10, 4), (9, 4), (101, 8192), (1, 4), (0, 4), (8193, 8192)):
        bounds = eval_chunk_bounds(n, batch)
        covered = sum(stop - start for start, stop in bounds)
        assert covered == n, f"n={n} batch={batch} chỉ phủ {covered} dòng"
        assert all(stop > start for start, stop in bounds)
        if len(bounds) > 1:
            assert all(stop - start >= 2 for start, stop in bounds), \
                "mọi lô phải có ≥ 2 dòng để dựng được đồ thị"
        # Các khoảng phải liền nhau, không chồng lấn
        for (_, prev_stop), (next_start, _) in zip(bounds, bounds[1:]):
            assert prev_stop == next_start


def test_evaluate_auprc_is_invariant_to_input_row_order() -> None:
    """Bất biến quan trọng nhất của chấm theo lô: ĐIỂM PHẢI KHỚP ĐÚNG NHÃN.

    Hàm tự sắp lại theo thời gian, nên xáo trộn thứ tự dòng đầu vào không
    được làm đổi kết quả. Nếu việc ghép điểm với nhãn bị lệch, test này đỏ.
    """
    df = _window(n_accounts=8, per_account=5)
    model = _model(seed=9)
    model.fusion.set_disabled_branches({"graph"})
    engine = RiskScoringEngine(24)

    ordered = evaluate_auprc(model, engine, df, DEVICE, use_graph=False, batch_size=16)
    shuffled = evaluate_auprc(
        model, engine, df.sample(frac=1.0, random_state=3), DEVICE,
        use_graph=False, batch_size=16)
    assert ordered is not None
    assert ordered == pytest.approx(shuffled), "ghép điểm với nhãn đang bị lệch"


def test_evaluate_auprc_chunk_size_changes_result_but_stays_valid() -> None:
    """Chấm theo lô nhỏ KHÔNG trùng chấm một lô, vì trạng thái hành vi reset
    ở biên mỗi lô — giống hệt lúc huấn luyện. Cả hai vẫn phải là AUPRC hợp lệ."""
    df = _window(n_accounts=8, per_account=5)
    model = _model(seed=11)
    model.fusion.set_disabled_branches({"graph"})
    engine = RiskScoringEngine(24)

    values = [
        evaluate_auprc(model, engine, df, DEVICE, use_graph=False, batch_size=batch)
        for batch in (8, len(df))
    ]
    for value in values:
        assert value is not None and 0.0 <= value <= 1.0


def test_evaluate_auprc_returns_none_without_labels() -> None:
    df = _window().drop(columns=["label"])
    model = _model(seed=10)
    engine = RiskScoringEngine(24)
    assert evaluate_auprc(model, engine, df, DEVICE, use_graph=False) is None


def test_window_bounds_drops_only_single_row_window() -> None:
    # Cửa sổ cuối còn 2 dòng thì GIỮ (vẫn tạo được cặp), còn 1 dòng thì bỏ —
    # cùng quy tắc với make_time_windows.
    assert window_bounds(10, 4) == [(0, 4), (4, 8), (8, 10)]
    assert window_bounds(9, 4) == [(0, 4), (4, 8)]
    assert window_bounds(10, 5) == [(0, 5), (5, 10)]


# ---------------------------------------------------------------------------
# Chọn checkpoint bằng trung bình trượt (select_epoch / _online_candidate)
# ---------------------------------------------------------------------------

def test_select_epoch_argmax_picks_highest() -> None:
    # Cách cũ: đỉnh đơn lẻ ở epoch 3 thắng dù hai epoch kề nó đều tệ.
    values = [0.10, 0.11, 0.90, 0.12, 0.40, 0.42, 0.41]
    assert select_epoch(values, "argmax") == (3, 0.90, 0.90)


def test_select_epoch_smoothed_rejects_isolated_spike() -> None:
    # Cùng dãy trên. Epoch 3 (đỉnh) bị trung bình trượt dìm xuống 0.3767.
    # Epoch 4 bị phép min dìm xuống 0.12 (val thô của chính nó) — nếu chỉ dùng
    # trung bình trượt thuần thì epoch 4 sẽ THẮNG với 0.4733 dù tự nó chỉ 0.12.
    # Epoch 6 thắng với min(0.41, 0.42) = 0.41.
    values = [0.10, 0.11, 0.90, 0.12, 0.40, 0.42, 0.41]
    epoch, raw, score = select_epoch(values, "smoothed")
    assert epoch == 6
    assert raw == pytest.approx(0.42)
    assert score == pytest.approx((0.40 + 0.42 + 0.41) / 3)


def test_select_epoch_smoothed_never_exceeds_own_value() -> None:
    """Bất biến cốt lõi: điểm chọn của một epoch không bao giờ cao hơn val thô
    của chính epoch đó, nên không epoch nào thắng nhờ láng giềng."""
    values = [0.10, 0.11, 0.90, 0.12, 0.40, 0.42, 0.41]
    epoch, raw, score = select_epoch(values, "smoothed")
    assert score <= raw + 1e-12
    # epoch 4 là ca gây lỗi: trung bình trượt 0.4733 nhưng tự nó chỉ 0.12
    assert epoch != 4


def test_select_epoch_smoothed_never_picks_first_or_last() -> None:
    # Giá trị cao nhất nằm ở hai đầu nhưng không đủ lân cận -> không được chọn.
    assert select_epoch([0.9, 0.1, 0.1, 0.1], "smoothed")[0] != 1
    assert select_epoch([0.1, 0.1, 0.1, 0.9], "smoothed")[0] != 4


def test_select_epoch_falls_back_to_argmax_when_too_few() -> None:
    assert select_epoch([0.2, 0.5], "smoothed") == (2, 0.5, 0.5)
    assert select_epoch([0.3], "smoothed") == (1, 0.3, 0.3)
    assert select_epoch([], "smoothed") is None


def test_select_epoch_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="selection"):
        select_epoch([0.1, 0.2, 0.3], "trung_binh")


@pytest.mark.parametrize("mode", ["smoothed", "argmax"])
@pytest.mark.parametrize("seed", range(12))
def test_online_selection_matches_select_epoch(mode: str, seed: int) -> None:
    """Vòng lặp train() chỉ giữ 3 snapshot gần nhất nên phải chọn TRỰC TUYẾN.
    Test này chứng minh cách làm trực tuyến đó cho đúng kết quả của
    select_epoch chạy trên toàn bộ dãy — nếu đỏ thì checkpoint được ghi ra
    không phải epoch mà manifest nói là tốt nhất."""
    rng = random.Random(seed)
    values = [round(rng.random(), 4) for _ in range(rng.randint(1, 15))]

    snapshots: deque = deque(maxlen=SMOOTH_WINDOW)
    best_score, best_epoch = float("-inf"), None
    for i, v in enumerate(values, start=1):
        snapshots.append((i, v, {"w": i}, {"w": i}))
        cand = _online_candidate(snapshots, mode)
        if cand is not None and cand[1] > best_score:
            best_score, best_epoch = cand[1], cand[0]
    if best_epoch is None:  # cùng nhánh fallback với train()
        best_epoch = max(snapshots, key=lambda s: s[1])[0]

    assert best_epoch == select_epoch(values, mode)[0]


def test_online_candidate_returns_middle_snapshot_weights() -> None:
    # Trọng số ghi ra phải là của epoch Ở GIỮA, không phải epoch vừa chấm.
    snapshots = deque(
        [(4, 0.2, {"w": 4}, {"r": 4}), (5, 0.3, {"w": 5}, {"r": 5}), (6, 0.1, {"w": 6}, {"r": 6})],
        maxlen=SMOOTH_WINDOW,
    )
    epoch, score, raw, model_sd, risk_sd = _online_candidate(snapshots, "smoothed")
    assert (epoch, raw, model_sd, risk_sd) == (5, 0.3, {"w": 5}, {"r": 5})
    assert score == pytest.approx(0.2)


def test_online_candidate_argmax_uses_newest_snapshot() -> None:
    snapshots = deque([(1, 0.9, {"w": 1}, {"r": 1}), (2, 0.1, {"w": 2}, {"r": 2})], maxlen=SMOOTH_WINDOW)
    epoch, score, raw, model_sd, _ = _online_candidate(snapshots, "argmax")
    assert (epoch, score, raw, model_sd) == (2, 0.1, 0.1, {"w": 2})


def test_online_candidate_smoothed_waits_for_full_window() -> None:
    snapshots = deque(maxlen=SMOOTH_WINDOW)
    snapshots.append((1, 0.5, {}, {}))
    assert _online_candidate(snapshots, "smoothed") is None
    snapshots.append((2, 0.6, {}, {}))
    assert _online_candidate(snapshots, "smoothed") is None
    snapshots.append((3, 0.7, {}, {}))
    assert _online_candidate(snapshots, "smoothed")[0] == 2


def test_cpu_state_is_detached_copy() -> None:
    # Phải là bản sao: optimizer cập nhật tại chỗ sau đó không được làm đổi
    # trọng số đã chụp, nếu không checkpoint "epoch 5" lại chứa trọng số epoch 7.
    model = _model(seed=3)
    snap = _cpu_state(model)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    after = model.state_dict()
    assert any(not torch.allclose(snap[k], after[k].cpu()) for k in snap)
    assert all(not v.requires_grad for v in snap.values())


def test_mark_saved_marks_current_or_past_record() -> None:
    recorded = [{"epoch": 1}, {"epoch": 2}]
    current = {"epoch": 3}
    _mark_saved(recorded, current, 2)
    assert recorded[1]["saved_checkpoint"] is True
    _mark_saved(recorded, current, 3)
    assert current["saved_checkpoint"] is True
