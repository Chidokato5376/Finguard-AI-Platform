"""Unit tests cho src/drre/losses.py.

Trọng tâm: đảm bảo cơ chế chống representation collapse (target network /
stop-gradient) được thực thi đúng — đây là điểm kỹ thuật quan trọng nhất
được nêu trong đánh giá phản biện đặc tả gốc.
"""

import torch

from src.drre.losses import (
    DRRECompositeLoss,
    EMATargetEncoder,
    contrastive_loss,
    prediction_loss,
)


def test_prediction_loss_rejects_grad_requiring_target() -> None:
    """e_target phải đến từ target network (no_grad) — nếu không, raise."""
    e_pred = torch.randn(4, 8, requires_grad=True)
    e_target_bad = torch.randn(4, 8, requires_grad=True)
    try:
        prediction_loss(e_pred, e_target_bad)
        assert False, "Kỳ vọng ValueError khi e_target.requires_grad=True"
    except ValueError:
        pass


def test_prediction_loss_accepts_detached_target() -> None:
    e_pred = torch.randn(4, 8, requires_grad=True)
    e_target_ok = torch.randn(4, 8).detach()
    loss = prediction_loss(e_pred, e_target_ok)
    assert loss.item() >= 0.0


def test_ema_target_encoder_no_grad() -> None:
    online = torch.nn.Linear(8, 8)
    target = EMATargetEncoder(online, decay=0.99)
    for p in target.target_encoder.parameters():
        assert not p.requires_grad

    x = torch.randn(2, 8)
    out = target(x)
    assert not out.requires_grad


def test_contrastive_loss_shape() -> None:
    batch, dim, n_neg = 4, 16, 5
    e_i = torch.randn(batch, dim)
    e_pos = torch.randn(batch, dim)
    e_neg = torch.randn(batch, n_neg, dim)
    loss = contrastive_loss(e_i, e_pos, e_neg, temperature=0.1)
    assert loss.dim() == 0  # scalar
    assert loss.item() >= 0.0


def test_composite_loss_without_labels() -> None:
    """L_BCE là nhánh phụ — phải hoạt động khi không có nhãn (Mục 3.3)."""
    batch, dim, n_neg = 4, 16, 5
    loss_fn = DRRECompositeLoss(lambda_contrast=0.5, lambda_bce=0.3)
    out = loss_fn(
        e_pred=torch.randn(batch, dim, requires_grad=True),
        e_target=torch.randn(batch, dim).detach(),
        e_anchor=torch.randn(batch, dim),
        e_pos=torch.randn(batch, dim),
        e_neg=torch.randn(batch, n_neg, dim),
    )
    assert "loss" in out
    assert out["l_bce"].item() == 0.0
