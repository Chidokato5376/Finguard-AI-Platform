"""Behavior Memory Module (Mục 4.1).

h_t^c = GRUCell(x_t, h_{t-1}^c)
M_t^c = [m_1, ..., m_K]
m_k <- (1 - alpha_k)*m_k + alpha_k*f_enc(x_t),  alpha_k ~ similarity(x_t, m_k)
"""

from __future__ import annotations

import torch
from torch import nn


class BehaviorMemory(nn.Module):
    """Bộ nhớ hành vi động cho một khách hàng, cập nhật liên tục sau mỗi giao dịch.

    Gồm hai phần:
      - GRU state h_t^c: trạng thái hành vi cơ bản, cập nhật tuần tự.
      - K slot bộ nhớ M_t^c: giữ nhiều "mẫu hành vi điển hình" song song
        (vd. chi tiêu hàng ngày nhỏ + trả học phí theo quý), mỗi slot có
        decay riêng theo mức độ khớp với giao dịch mới.

    Đối với MVP 6 tuần: có thể khởi tạo num_slots=1 để tương đương một
    GRU đơn thuần, sau đó tăng dần nếu còn thời gian (xem README §7).
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_slots: int = 4) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_slots = num_slots

        self.gru_cell = nn.GRUCell(input_dim, hidden_dim)
        self.f_enc = nn.Linear(input_dim, hidden_dim)

        # Slot tương đồng dùng dot-product; decay rate alpha_k phụ thuộc
        # similarity nên không có tham số học riêng — được tính runtime.

    def init_state(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Khởi tạo (h_0, M_0) cho một batch khách hàng mới (cold-start)."""
        h0 = torch.zeros(batch_size, self.hidden_dim, device=device)
        m0 = torch.zeros(batch_size, self.num_slots, self.hidden_dim, device=device)
        return h0, m0

    def forward(
        self, x_t: torch.Tensor, h_prev: torch.Tensor, m_prev: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Cập nhật một bước thời gian.

        Args:
            x_t: (batch, input_dim) — giao dịch hiện tại đã encode.
            h_prev: (batch, hidden_dim) — trạng thái GRU trước đó.
            m_prev: (batch, num_slots, hidden_dim) — slot bộ nhớ trước đó.

        Returns:
            h_t: (batch, hidden_dim) trạng thái GRU mới.
            m_t: (batch, num_slots, hidden_dim) slot bộ nhớ đã cập nhật.
        """
        h_t = self.gru_cell(x_t, h_prev)

        encoded = self.f_enc(x_t)  # (batch, hidden_dim)
        # similarity(x_t, m_k) — cosine similarity giữa encoded và từng slot
        encoded_norm = torch.nn.functional.normalize(encoded, dim=-1).unsqueeze(1)
        slots_norm = torch.nn.functional.normalize(m_prev, dim=-1)
        similarity = (encoded_norm * slots_norm).sum(dim=-1)  # (batch, num_slots)
        alpha_k = torch.softmax(similarity, dim=-1).unsqueeze(-1)  # (batch, num_slots, 1)

        m_t = (1 - alpha_k) * m_prev + alpha_k * encoded.unsqueeze(1)
        return h_t, m_t
