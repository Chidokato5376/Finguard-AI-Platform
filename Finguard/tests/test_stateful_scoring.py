"""Chấm DRRE có mang trạng thái (src/evaluation/stateful_scoring.py) và ma trận nhầm lẫn."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.drre.model import DRRE
from src.drre.train import BehaviorStateStore
from src.evaluation.metrics import (
    best_f1_threshold,
    confusion_at_budget,
    confusion_at_threshold,
)
from src.evaluation.stateful_scoring import fingerprint, score_split, warm_up_store
from src.scoring.risk_scoring import RiskScoringEngine

FEATS = ["amount_norm", "channel_encoded"]
WINDOW = 4


def _frame(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "step": np.arange(n),
        "account_id": rng.choice(list("ABCDE"), n),
        "counterparty_id": rng.choice(list("VWXYZ"), n),
        "amount_norm": rng.normal(5, 1, n).astype(np.float32),
        "channel_encoded": rng.integers(0, 3, n).astype(np.float32),
    })


def _model():
    torch.manual_seed(0)
    model = DRRE(input_dim=len(FEATS), hidden_dim=8, num_slots=3,
                 graph_hidden_dim=8, graph_num_layers=2, embedding_dim=16)
    model.feature_columns = list(FEATS)
    model.use_graph = True
    model.eval()
    return model, RiskScoringEngine(16).eval()


def _store(capacity: int = 32) -> BehaviorStateStore:
    return BehaviorStateStore(8, 3, capacity=capacity)


ID = {"format": 1, "window": WINDOW, "checkpoint": ["x", 1, 1], "data": []}


def _collecting():
    rows: dict[int, np.ndarray] = {}

    def on_window(start, stop, embedding, logit):
        for offset, row in enumerate(logit.numpy()):
            rows[start + offset] = row
    return rows, on_window


def test_chay_tiep_giong_het_quet_lien(tmp_path):
    stream, target = _frame(40), _frame(12, seed=1)
    model, engine = _model()

    straight = _store()
    warm_up_store(model, engine, stream, straight, None, ID, window=WINDOW)
    expected = score_split(model, engine, target, straight, window=WINDOW)

    # Lần chạy đầu "chết" sau cửa sổ thứ 5; tiến độ đã lưu ở cửa sổ thứ 4.
    cache = tmp_path / "warm.pt"

    class Dead(Exception):
        pass

    def die_at_fifth(start, stop, embedding, logit):
        if start == 4 * WINDOW:
            raise Dead

    with pytest.raises(Dead):
        warm_up_store(model, engine, stream, _store(), cache, ID, window=WINDOW,
                      checkpoint_every=4, on_window=die_at_fifth)

    seen, on_window = _collecting()
    resumed = _store()
    warm_up_store(model, engine, stream, resumed, cache, ID, window=WINDOW,
                  checkpoint_every=4, on_window=on_window)
    assert min(seen) == 4 * WINDOW  # không quét lại phần đã lưu
    np.testing.assert_array_equal(score_split(model, engine, target, resumed, window=WINDOW), expected)


def test_ket_qua_co_trang_thai_khac_bo_nho_trong():
    stream, target = _frame(40), _frame(12, seed=1)
    model, engine = _model()
    warm = _store()
    warm_up_store(model, engine, stream, warm, None, ID, window=WINDOW)
    assert not np.array_equal(
        score_split(model, engine, target, warm, window=WINDOW),
        score_split(model, engine, target, _store(), window=WINDOW))


def test_dau_van_tay_lech_thi_quet_lai(tmp_path):
    stream = _frame(16)
    model, engine = _model()
    cache = tmp_path / "warm.pt"
    warm_up_store(model, engine, stream, _store(), cache, ID, window=WINDOW)

    seen, on_window = _collecting()
    warm_up_store(model, engine, stream, _store(), cache, {**ID, "window": 99}, window=WINDOW,
                  on_window=on_window)
    assert min(seen) == 0


def test_quet_xong_thi_lan_sau_khong_quet_lai(tmp_path):
    stream, target = _frame(16), _frame(8, seed=2)
    model, engine = _model()
    cache = tmp_path / "warm.pt"
    first = _store()
    warm_up_store(model, engine, stream, first, cache, ID, window=WINDOW)
    expected = score_split(model, engine, target, first, window=WINDOW)

    seen, on_window = _collecting()
    second = _store()
    warm_up_store(model, engine, stream, second, cache, ID, window=WINDOW, on_window=on_window)
    assert not seen
    np.testing.assert_array_equal(score_split(model, engine, target, second, window=WINDOW), expected)


def test_khong_quet_lai_lam_lech_du_lieu_goc():
    stream = _frame(16)
    before = stream.copy()
    model, engine = _model()
    warm_up_store(model, engine, stream, _store(), None, ID, window=WINDOW)
    pd.testing.assert_frame_equal(stream, before)


def test_store_nap_sai_capacity_bi_tu_choi():
    with pytest.raises(ValueError, match="capacity"):
        _store(8).load_state_dict(_store(16).state_dict())


def test_dau_van_tay_doi_khi_file_doi(tmp_path):
    ckpt, data = tmp_path / "m.pt", tmp_path / "d.parquet"
    ckpt.write_bytes(b"a")
    data.write_bytes(b"b")
    before = fingerprint(ckpt, [data], 8)
    data.write_bytes(b"bb")
    assert fingerprint(ckpt, [data], 8) != before


# --------------------------------------------------------------------------
# Ma trận nhầm lẫn
# --------------------------------------------------------------------------

Y = np.array([1, 0, 1, 0, 0, 1])
SCORES = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])


def test_ma_tran_theo_ngan_sach():
    c = confusion_at_budget(Y, SCORES, 3)
    assert (c["tp"], c["fp"], c["fn"], c["tn"]) == (2, 1, 1, 2)
    assert c["precision"] == pytest.approx(2 / 3)
    assert c["recall"] == pytest.approx(2 / 3)


def test_ma_tran_theo_nguong():
    c = confusion_at_threshold(Y, SCORES, 0.75)
    assert (c["tp"], c["fp"], c["fn"], c["tn"]) == (1, 1, 2, 2)
    assert c["threshold"] == 0.75


def test_nguong_f1_tot_nhat():
    threshold = best_f1_threshold(Y, SCORES)
    best = confusion_at_threshold(Y, SCORES, threshold)["f1"]
    for t in SCORES:
        assert best >= confusion_at_threshold(Y, SCORES, float(t))["f1"]


def test_khong_chon_nguong_tren_test_va_dung_nguong_tu_val():
    path = Path(__file__).resolve().parents[1] / "scripts" / "evaluate.py"
    spec = importlib.util.spec_from_file_location("evaluate_for_confusion", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    scores = {"m": SCORES}

    assert module.BEST_F1_KEY not in module.confusion_report(Y, scores, (2,), "test")["m"]
    row = module.confusion_report(Y, scores, (2,), "test", {"m": 0.75})["m"]
    assert row[module.VAL_THRESHOLD_KEY]["threshold"] == 0.75
    assert module.BEST_F1_KEY in module.confusion_report(Y, scores, (2,), "val")["m"]


def test_theo_kenh_dem_dung():
    path = Path(__file__).resolve().parents[1] / "scripts" / "evaluate.py"
    spec = importlib.util.spec_from_file_location("evaluate_for_channel", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    channels = np.array(["ach", "ach", "ach", "cash", "cash", "cash"], dtype=object)
    scores = {"m": SCORES}
    confusion = module.confusion_report(Y, scores, (2,), "test", {"m": 0.75})
    records = {r["channel"]: r for r in module.metrics_by_channel(channels, Y, scores, confusion, 2)}
    assert records["ach"]["n_pos"] == 2 and records["cash"]["n_pos"] == 1
    assert records["ach"]["methods"]["m"]["in_top"] == 2
    assert records["ach"]["methods"]["m"]["tp_nguong"] == 1
    assert records["cash"]["methods"]["m"]["tp_nguong"] == 0
