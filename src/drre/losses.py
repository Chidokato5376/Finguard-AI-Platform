"""Mục tiêu huấn luyện DRRE (Mục 4.6).

L_pred     = || e_hat_{t+1} - e_{t+1} ||^2_2
L_contrast = -log[ exp(sim(e_i,e_i+)/tau) / sum_j exp(sim(e_i,e_j-)/tau) ]
L          = L_pred + lambda_1*L_contrast + lambda_2*L_BCE

=== GHI CHÚ KỸ THUẬT QUAN TRỌNG (đã nêu trong đánh giá phản biện) ===
L_pred nguyên bản so sánh output encoder với chính nó tại t+1. Nếu
target e_{t+1} được sinh bởi CHÍNH encoder đang huấn luyện (cùng
gradient path), nghiệm tầm thường là encoder ánh xạ mọi input về một
điểm cố định -> loss về 0 nhưng embedding vô nghĩa (representation
collapse — hiện tượng kinh điển trong self-supervised learning, xem
BYOL/SimSiam).

Cách xử lý bắt buộc trong implementation này: dùng một TARGET NETWORK
riêng (EMA — exponential moving average của online encoder), không lan
truyền gradient qua nhánh target. Đây là lý do class DRRELoss dưới đây
nhận `target_encoder` như một tham số tách biệt với `online_encoder`,
và target luôn được gọi trong `torch.no_grad()`.
"""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from torch import nn


class EMATargetEncoder:
    """Target network cập nhật bằng exponential moving average của online encoder.

    Đây chính là cơ chế chống collapse cho L_pred — KHÔNG được bỏ qua.
    """

    def __init__(self, online_encoder: nn.Module, decay: float = 0.996) -> None:
        self.decay = decay
        self.target_encoder = copy.deepcopy(online_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, online_encoder: nn.Module) -> None:
        for target_p, online_p in zip(
            self.target_encoder.parameters(), online_encoder.parameters()
        ):
            target_p.data = self.decay * target_p.data + (1 - self.decay) * online_p.data

    @torch.no_grad()
    def __call__(self, *args, **kwargs) -> torch.Tensor:
        return self.target_encoder(*args, **kwargs)


def prediction_loss(e_pred: torch.Tensor, e_target: torch.Tensor) -> torch.Tensor:
    """L_pred — e_target PHẢI đến từ target network (không có gradient), không phải
    trực tiếp từ online encoder đang huấn luyện.

    Args:
        e_pred: (batch, dim) — dự đoán của online encoder cho embedding t+1.
        e_target: (batch, dim) — embedding thật tại t+1, sinh bởi EMATargetEncoder
            (đã .detach() / no_grad).
    """
    if e_target.requires_grad:
        raise ValueError(
            "e_target đang có requires_grad=True — nguy cơ representation "
            "collapse. e_target phải đến từ target network trong torch.no_grad()."
        )
    return F.mse_loss(e_pred, e_target)


def contrastive_loss(
    e_i: torch.Tensor, e_pos: torch.Tensor, e_neg: torch.Tensor, temperature: float = 0.1
) -> torch.Tensor:
    """L_contrast — InfoNCE-style contrastive loss.

    Args:
        e_i: (batch, dim) — anchor embeddings.
        e_pos: (batch, dim) — positive pairs (cùng khách hàng, cùng mẫu hành vi bình thường).
        e_neg: (batch, num_negatives, dim) — negative samples (bất thường / synthetic negative).
        temperature: tau.
    """
    e_i = F.normalize(e_i, dim=-1)
    e_pos = F.normalize(e_pos, dim=-1)
    e_neg = F.normalize(e_neg, dim=-1)

    pos_sim = (e_i * e_pos).sum(dim=-1) / temperature  # (batch,)
    neg_sim = torch.einsum("bd,bnd->bn", e_i, e_neg) / temperature  # (batch, num_neg)

    logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # (batch, 1+num_neg)
    labels = torch.zeros(e_i.size(0), dtype=torch.long, device=e_i.device)  # positive luôn ở index 0
    return F.cross_entropy(logits, labels)


class DRRECompositeLoss(nn.Module):
    """L = L_pred + lambda_1 * L_contrast + lambda_2 * L_BCE"""

    def __init__(self, lambda_contrast: float = 0.5, lambda_bce: float = 0.3) -> None:
        super().__init__()
        self.lambda_contrast = lambda_contrast
        self.lambda_bce = lambda_bce
        self.bce = nn.BCEWithLogitsLoss()

    def forward(
        self,
        e_pred: torch.Tensor,
        e_target: torch.Tensor,
        e_anchor: torch.Tensor,
        e_pos: torch.Tensor,
        e_neg: torch.Tensor,
        bce_logits: torch.Tensor | None = None,
        bce_labels: torch.Tensor | None = None,
        temperature: float = 0.1,
    ) -> dict[str, torch.Tensor]:
        l_pred = prediction_loss(e_pred, e_target)
        l_contrast = contrastive_loss(e_anchor, e_pos, e_neg, temperature)

        l_bce = torch.tensor(0.0, device=e_pred.device)
        if bce_logits is not None and bce_labels is not None:
            # Chỉ áp dụng khi có nhãn — nhánh phụ, không bắt buộc (Mục 3.3).
            l_bce = self.bce(bce_logits, bce_labels.float())

        total = l_pred + self.lambda_contrast * l_contrast + self.lambda_bce * l_bce
        return {
            "loss": total,
            "l_pred": l_pred.detach(),
            "l_contrast": l_contrast.detach(),
            "l_bce": l_bce.detach(),
        }
