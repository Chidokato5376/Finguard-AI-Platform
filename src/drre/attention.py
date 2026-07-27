"""Context-aware Attention (Mục 4.2).

alpha_k = softmax_k(q_t . m_k / sqrt(d)),  q_t = W_q x_t
h~_t^c  = sum_k alpha_k * m_k
delta_t = ||x_t - h~_t^c||_2   hoac   delta_t = 1 - max_k(alpha_k)
"""

from __future__ import annotations

import torch
from torch import nn


class ContextAwareAttention(nn.Module):
    """So khớp giao dịch hiện tại với các mẫu hành vi đã lưu (không dùng ngưỡng chung).

    Độ lệch ngữ cảnh delta_t sinh ra tự nhiên từ chính trọng số attention,
    và bản thân nó đã mang ý nghĩa diễn giải được (dùng lại ở Module 6).
    """

    def __init__(self, input_dim: int, key_dim: int) -> None:
        super().__init__()
        self.key_dim = key_dim
        self.w_q = nn.Linear(input_dim, key_dim)

    def forward(
        self, x_t: torch.Tensor, m_t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x_t: (batch, input_dim)
            m_t: (batch, num_slots, hidden_dim) — slot bộ nhớ hiện tại.

        Returns:
            h_tilde: (batch, hidden_dim) — biểu diễn ngữ cảnh hóa.
            alpha_k: (batch, num_slots) — trọng số attention, dùng cho Module 6.
            delta_t: (batch,) — độ lệch ngữ cảnh = 1 - max_k(alpha_k).
        """
        q_t = self.w_q(x_t)  # (batch, key_dim)
        # Giả định hidden_dim của m_t == key_dim của q_t (đồng bộ trong config.yaml)
        scores = torch.einsum("bd,bkd->bk", q_t, m_t) / (self.key_dim ** 0.5)
        alpha_k = torch.softmax(scores, dim=-1)  # (batch, num_slots)

        h_tilde = torch.einsum("bk,bkd->bd", alpha_k, m_t)  # (batch, hidden_dim)
        delta_t = 1.0 - alpha_k.max(dim=-1).values  # (batch,)
        return h_tilde, alpha_k, delta_t
