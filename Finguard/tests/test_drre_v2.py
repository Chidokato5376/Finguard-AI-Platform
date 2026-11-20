"""DRRE v2 — tầng chấm điểm đã sửa (Giai đoạn 2A) và việc gắn cột kịch bản.

Các bất biến được giữ:
  * chuẩn hóa nằm TRONG mô hình: đổi thang đo một cột rồi học lại thống kê thì
    điểm không đổi;
  * mỗi nhánh chỉ thấy nhóm cột của nó;
  * bản vector hóa của encoder vẫn trùng bản tuần tự tham chiếu;
  * checkpoint v2 nạp được ở mọi đường chấm điểm, checkpoint v1 vẫn ra DRRE cũ;
  * cột kịch bản gắn đúng dòng (file kịch bản thẳng hàng theo dòng gốc).
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.dashboard.graph_utils import build_account_graph
from src.drre.model import DRRE, build_drre
from src.drre.model_v2 import DRREv2, branch_groups
from src.drre.train import run_encoder_over_window, run_encoder_over_window_sequential, train
from src.scoring.risk_scoring import RiskScoringEngine, build_risk_engine

DEVICE = torch.device("cpu")
FEATS = ["amount_norm", "channel_encoded", "log_n_prior", "log_gap", "new_counterparty",
         "log_n_distinct_cp", "log_n_distinct_senders", "mc_forward_ratio_24h",
         "mc_log_age_days_cap3", "cb_log_n_banks_7d", "cy_reciprocity_7d"]


def _window(n_accounts: int = 6, per_account: int = 4, seed: int = 0) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    for i in range(n_accounts * per_account):
        row = {"account_id": f"ACC{i % n_accounts}", "counterparty_id": f"CP{(i * 7) % n_accounts}",
               "step": i, "label": float(i % 9 == 0)}
        row.update({c: rng.uniform(-2, 50) for c in FEATS})
        row["channel_encoded"] = float(i % 3)
        rows.append(row)
    return pd.DataFrame(rows)


def _model(seed: int = 0) -> DRREv2:
    torch.manual_seed(seed)
    model = DRREv2(input_dim=len(FEATS), feature_columns=FEATS, hidden_dim=16, num_slots=3,
                   graph_hidden_dim=16, graph_num_layers=2, embedding_dim=24, branch_dim=12)
    model.fit_normalizer(_window()[FEATS].to_numpy())
    return model.eval()


def _encode(model, window: pd.DataFrame) -> torch.Tensor:
    nf, ei, idx = build_account_graph(window, FEATS)
    return run_encoder_over_window(model, window, nf, ei, idx, DEVICE, feature_columns=FEATS)["e"]


def test_nhom_cot_theo_nhanh():
    groups = branch_groups(FEATS)
    named = {k: [FEATS[i] for i in v] for k, v in groups.items()}
    assert "channel_encoded" not in sum(named.values(), [])
    assert "amount_norm" in named["transaction"] and "amount_norm" in named["behavior"]
    assert "mc_forward_ratio_24h" in named["transaction"]       # ngoại lệ trong mule_chain
    assert "mc_log_age_days_cap3" in named["behavior"]
    assert {"cb_log_n_banks_7d", "cy_reciprocity_7d", "log_n_distinct_cp"} <= set(named["network"])


def test_dau_ra_dung_khoa_va_kich_thuoc():
    window = _window()
    model = _model()
    nf, ei, idx = build_account_graph(window, FEATS)
    x = torch.as_tensor(window[FEATS].to_numpy(np.float32))
    h, m = model.behavior_memory.init_state(len(window), DEVICE)
    node = torch.as_tensor([idx[a] for a in window["account_id"]])
    out = model(x, h, m, nf, ei, node)
    assert out["e_t"].shape == (len(window), 24) and out["e_pred"].shape == (len(window), 24)
    assert out["h_t"].shape == (len(window), 16) and out["m_t"].shape == (len(window), 3, 16)
    torch.testing.assert_close(out["beta"].sum(-1), torch.ones(len(window)))


def test_chuan_hoa_nam_trong_mo_hinh():
    """Nhân một cột với 1000 rồi học lại thống kê: điểm không đổi."""
    window = _window()
    scaled = window.copy()
    scaled["log_n_prior"] *= 1000.0
    a, b = _model(), _model()
    b.fit_normalizer(scaled[FEATS].to_numpy())
    torch.testing.assert_close(_encode(a, window), _encode(b, scaled), atol=1e-5, rtol=1e-4)
    assert {"feature_mean", "feature_std"} <= set(a.state_dict())


def test_kenh_di_qua_embedding_khong_qua_chuan_hoa():
    model = _model()
    i = FEATS.index("channel_encoded")
    assert model.feature_mean[i] == 0 and model.feature_std[i] == 1
    x = torch.as_tensor(_window()[FEATS].to_numpy(np.float32))
    tx, _ = model._branch_inputs(x)
    expected = model.channel_embedding(x[:, i].long())
    torch.testing.assert_close(tx[:, -expected.shape[1]:], expected)


def test_nhanh_giao_dich_khong_thay_cot_cua_nhanh_khac():
    model = _model()
    x = torch.as_tensor(_window()[FEATS].to_numpy(np.float32))
    changed = x.clone()
    for column in ("log_n_prior", "log_n_distinct_cp", "cy_reciprocity_7d"):
        changed[:, FEATS.index(column)] += 5.0
    torch.testing.assert_close(model._branch_inputs(x)[0], model._branch_inputs(changed)[0])
    assert not torch.equal(model._branch_inputs(x)[1], model._branch_inputs(changed)[1])


def test_vector_hoa_trung_ban_tuan_tu():
    window = _window()
    model = _model()
    nf, ei, idx = build_account_graph(window, FEATS)
    fast = run_encoder_over_window(model, window, nf, ei, idx, DEVICE, feature_columns=FEATS)
    slow = run_encoder_over_window_sequential(model, window, nf, ei, idx, DEVICE, feature_columns=FEATS)
    torch.testing.assert_close(fast["e"], slow["e"], atol=1e-5, rtol=1e-4)


def test_tat_nhanh_do_thi_thi_do_thi_khong_anh_huong():
    window = _window()
    model = _model()
    model.fusion.set_disabled_branches({"graph"})
    nf, ei, idx = build_account_graph(window, FEATS)
    base = run_encoder_over_window(model, window, nf, ei, idx, DEVICE, feature_columns=FEATS)["e"]
    other = run_encoder_over_window(model, window, nf * 3 + 1, ei, idx, DEVICE, feature_columns=FEATS)["e"]
    torch.testing.assert_close(base, other)


def test_build_drre_va_dau_cham_diem_theo_kien_truc():
    v1 = dict(input_dim=3, hidden_dim=8, num_slots=2, graph_hidden_dim=8, graph_num_layers=2,
              embedding_dim=16)
    assert type(build_drre(v1)) is DRRE
    assert hasattr(build_risk_engine(v1), "linear")
    v2 = dict(v1, input_dim=len(FEATS), arch="v2", feature_columns=FEATS, risk_hidden=8)
    assert type(build_drre(v2)) is DRREv2
    engine = build_risk_engine(v2)
    assert hasattr(engine, "mlp") and engine(torch.zeros(4, 16)).shape == (4,)
    assert isinstance(engine, RiskScoringEngine)


# --------------------------------------------------------------------------
# Huấn luyện đầu-cuối và gắn cột kịch bản
# --------------------------------------------------------------------------

def _processed(tmp_path: Path) -> Path:
    from tests.test_drre_train import _make_synthetic_processed_df

    out = tmp_path / "processed"
    (out / "scenarios").mkdir(parents=True)
    for split, n, seed in (("train", 300, 1), ("val", 80, 2)):
        df = _make_synthetic_processed_df(n_transactions=n, seed=seed)
        df.to_parquet(out / f"synthetic_vn_{split}.parquet")
        # Cột kịch bản giả: giá trị = số thứ tự dòng gốc, để kiểm tra gắn đúng dòng.
        rows = np.arange(n, dtype=np.float32)
        pd.DataFrame({"mc_forward_ratio_24h": rows, "mc_amt_sim_last_in": rows / n,
                      "mc_log_min_since_in": rows % 7, "mc_log_age_days_cap3": rows % 3}
                     ).to_parquet(out / "scenarios" / f"mule_chain.{split}.parquet")
    return out


def test_read_split_gan_cot_kich_ban_dung_dong(tmp_path):
    from src.data.scenarios.store import read_split

    path = _processed(tmp_path) / "synthetic_vn_train.parquet"
    frame = read_split(path, ["account_id", "mc_forward_ratio_24h"])
    np.testing.assert_array_equal(frame["mc_forward_ratio_24h"], np.arange(len(frame)))
    with pytest.raises(KeyError):
        read_split(path, ["khong_co_cot_nay"])


@pytest.mark.parametrize("arch", ["v1", "v2"])
def test_train_voi_kich_ban_roi_cham_diem_duoc(tmp_path, arch):
    from src.dashboard.scoring_service import try_load_drre_checkpoint

    processed = _processed(tmp_path)
    checkpoint = tmp_path / "ck.pt"
    train(source="synthetic_vn", epochs=2, batch_size=64, lr=1e-3, num_negatives=3,
          processed_dir=str(processed), checkpoint_path=str(checkpoint), seed=42, runs_dir=None,
          arch=arch, scenarios=["mule_chain"])
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert "mc_log_age_days_cap3" in saved["feature_columns"]
    assert "hour_sin" not in saved["feature_columns"]          # DROPPED_BASE_FEATURES
    assert saved["scenarios"] == ["mule_chain"]
    loaded = try_load_drre_checkpoint(str(checkpoint))
    assert loaded is not None
    model, engine = loaded
    assert type(model) is (DRREv2 if arch == "v2" else DRRE)
    if arch == "v2":
        assert float(model.feature_std[saved["feature_columns"].index("mc_forward_ratio_24h")]) > 1


def test_train_v2_chay_tiep_trung_chay_lien(tmp_path):
    processed = _processed(tmp_path)
    common = dict(source="synthetic_vn", epochs=2, batch_size=64, lr=1e-3, num_negatives=3,
                  processed_dir=str(processed), seed=42, runs_dir=None, arch="v2", select="argmax")
    common["selection"] = common.pop("select")
    train(checkpoint_path=str(tmp_path / "a.pt"), **common)
    train(checkpoint_path=str(tmp_path / "b.pt"), stop_after=1, **common)
    train(checkpoint_path=str(tmp_path / "b.pt"), resume=True, **common)
    a = torch.load(tmp_path / "a.pt", weights_only=True)["drre_state_dict"]
    b = torch.load(tmp_path / "b.pt", weights_only=True)["drre_state_dict"]
    for key in a:
        torch.testing.assert_close(a[key], b[key])
