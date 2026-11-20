"""Giai đoạn 2C: chống sụp embedding, cộng dồn gradient, học lại đầu chấm điểm (pha 2)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.drre.losses import DRRECompositeLoss, EMATargetEncoder
from src.drre.model_v2 import DRREv2
from src.drre.train import train, train_on_window, variance_covariance_loss
from src.scoring.risk_scoring import RiskScoringEngine

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import refit_head  # noqa: E402

FEATS = ["amount_norm", "channel_encoded", "log_n_prior", "log_gap", "new_counterparty",
         "log_n_distinct_cp", "log_n_distinct_senders"]


def test_so_hang_chong_sup_phat_embedding_sup():
    rng = torch.Generator().manual_seed(0)
    spread = torch.randn(4000, 32, generator=rng)
    collapsed = torch.randn(4000, 1, generator=rng) @ torch.randn(1, 32, generator=rng) * 0.1
    assert float(variance_covariance_loss(collapsed)) > 5 * float(variance_covariance_loss(spread))
    e = collapsed.clone().requires_grad_(True)
    variance_covariance_loss(e).backward()
    assert e.grad is not None and float(e.grad.abs().sum()) > 0


def _window(n=64, seed=0):
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        row = {"account_id": f"A{i % 8}", "counterparty_id": f"A{(i * 3 + 1) % 8}", "step": i,
               "label": float(i % 7 == 0), "eval_universe": True}
        row.update({c: rng.uniform(0, 5) for c in FEATS})
        row["channel_encoded"] = float(i % 3)
        rows.append(row)
    return pd.DataFrame(rows)


def test_cong_don_gradient_chi_cap_nhat_sau_k_cua_so():
    torch.manual_seed(0)
    model = DRREv2(len(FEATS), FEATS, hidden_dim=8, num_slots=2, graph_hidden_dim=8,
                   embedding_dim=12, branch_dim=8)
    engine = RiskScoringEngine(12, hidden=8)
    target = EMATargetEncoder(model)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(engine.parameters()), lr=1e-2)
    loss_fn = DRRECompositeLoss(pos_weight=5.0)
    before = [p.detach().clone() for p in model.parameters()]
    common = dict(feature_columns=FEATS, lambda_vc=0.1, accumulate=2, last_window=False)
    train_on_window(model, engine, target, loss_fn, optimizer, _window(seed=1), torch.device("cpu"),
                    3, random.Random(0), window_index=1, **common)
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters()))   # chưa cập nhật
    stats = train_on_window(model, engine, target, loss_fn, optimizer, _window(seed=2),
                            torch.device("cpu"), 3, random.Random(0), window_index=2, **common)
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))
    assert stats["l_vc"] > 0


def test_focal_loss_dung_cong_thuc():
    logits = torch.tensor([2.0, -1.0, 0.5])
    labels = torch.tensor([1.0, 0.0, 0.0])
    p = torch.sigmoid(logits)
    want = torch.stack([
        -0.25 * (1 - p[0]) ** 2 * torch.log(p[0]),
        -0.75 * p[1] ** 2 * torch.log(1 - p[1]),
        -0.75 * p[2] ** 2 * torch.log(1 - p[2]),
    ]).mean()
    torch.testing.assert_close(refit_head.head_loss("focal", logits, labels, 10.0), want)


def test_hoc_lai_dau_cham_diem_tach_duoc_du_lieu_tach_duoc():
    rng = np.random.default_rng(0)
    y = rng.random(6000) < 0.02
    e = rng.normal(size=(6000, 12)).astype(np.float32)
    e[y, 0] += 5.0
    e_t = torch.from_numpy(e)
    head, info = refit_head.fit_head(
        lambda: RiskScoringEngine(12, hidden=8), e_t[:5000], torch.from_numpy(y[:5000].astype(
            np.float32)), e_t[5000:], y[5000:], "bce", epochs=8, batch=512, lr=1e-2, patience=3,
        seed=0)
    assert info["holdout_auprc"] > 0.8
    assert info["pos_weight"] == pytest.approx((5000 - y[:5000].sum()) / y[:5000].sum())


def test_train_2c_roi_hoc_lai_dau_cham_diem(tmp_path, monkeypatch):
    from src.dashboard.scoring_service import try_load_drre_checkpoint
    from tests.test_drre_train import _make_synthetic_processed_df

    processed = tmp_path / "processed"
    processed.mkdir()
    _make_synthetic_processed_df(n_transactions=400, seed=1).to_parquet(
        processed / "synthetic_vn_train.parquet")
    _make_synthetic_processed_df(n_transactions=80, seed=2).to_parquet(
        processed / "synthetic_vn_val.parquet")
    checkpoint = tmp_path / "ck.pt"
    train(source="synthetic_vn", epochs=2, batch_size=64, lr=1e-3, num_negatives=3,
          processed_dir=str(processed), checkpoint_path=str(checkpoint), seed=42, runs_dir=None,
          arch="v2", carry_state=True, lambda_vc=0.05, accumulate=2)
    out = tmp_path / "ck_head.pt"
    report = tmp_path / "head.json"
    monkeypatch.setattr(sys, "argv", [
        "refit_head.py", "--checkpoint", str(checkpoint), "--processed-dir", str(processed),
        "--out-checkpoint", str(out), "--source", "synthetic_vn", "--epochs", "2",
        "--batch", "64", "--window", "64", "--report", str(report), "--holdout", "0.3"])
    assert refit_head.main() == 0
    saved = torch.load(out, map_location="cpu", weights_only=True)
    assert saved["head_refit"]["chosen"] in refit_head.LOSSES
    old = torch.load(checkpoint, map_location="cpu", weights_only=True)
    for key in old["drre_state_dict"]:                     # bộ mã hóa giữ nguyên
        torch.testing.assert_close(saved["drre_state_dict"][key], old["drre_state_dict"][key])
    assert try_load_drre_checkpoint(str(out)) is not None
    assert report.exists()
