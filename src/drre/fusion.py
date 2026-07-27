"""Adaptive Risk Fusion (Mục 4.4).

beta = softmax(W_beta . [x_t || h~_t^c || g_t^c]) in R^3
e_t^c = beta_1*phi_1(x_t) + beta_2*phi_2(h~_t^c) + beta_3*phi_3(g_t^c)

Co che nay giai quyet truc tiep bai toan cold-start: khach hang moi
chua co lich su (Behavior Memory rong) se tu dong duoc gate don trong
so sang tin hieu do thi va giao dich tho.
"""

from __future__ import annotations

import torch
from torch import nn


class AdaptiveRiskFusion(nn.Module):
    """Hợp nhất ba nguồn thông tin bằng trọng số học được (gating network)."""

    def __init__(self, x_dim: int, h_dim: int, g_dim: int, embedding_dim: int) -> None:
        super().__init__()
        self.phi_1 = nn.Linear(x_dim, embedding_dim)
        self.phi_2 = nn.Linear(h_dim, embedding_dim)
        self.phi_3 = nn.Linear(g_dim, embedding_dim)
        self.gate = nn.Linear(x_dim + h_dim + g_dim, 3)

    def forward(
        self, x_t: torch.Tensor, h_tilde: torch.Tensor, g_t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x_t: (batch, x_dim) — giao dịch thô.
            h_tilde: (batch, h_dim) — ngữ cảnh hành vi (từ ContextAwareAttention).
            g_t: (batch, g_dim) — quan hệ mạng lưới (từ GraphRelationshipEncoder).

        Returns:
            e_t: (batch, embedding_dim) — Risk Embedding cuối cùng.
            beta: (batch, 3) — trọng số gate [beta_x, beta_h, beta_g], dùng cho Module 6.
        """
        concat = torch.cat([x_t, h_tilde, g_t], dim=-1)
        beta = torch.softmax(self.gate(concat), dim=-1)  # (batch, 3)

        e_t = (
            beta[:, 0:1] * self.phi_1(x_t)
            + beta[:, 1:2] * self.phi_2(h_tilde)
            + beta[:, 2:3] * self.phi_3(g_t)
        )
        return e_t, beta
