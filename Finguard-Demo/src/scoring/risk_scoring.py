"""Risk Scoring Engine (Module 5).

r_t = 100 * sigmoid(w . e_t^c + b)

Anh xa Risk Embedding sang mot diem rui ro lien tuc tren thang 0-100,
thay vi nhan nhi phan Fraud/Normal.
"""

from __future__ import annotations

from enum import Enum

import torch
from torch import nn


class RiskTier(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    CRITICAL = "critical"


class RiskScoringEngine(nn.Module):
    """Ánh xạ tuyến tính + sigmoid từ Risk Embedding sang điểm 0–100."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(embedding_dim, 1)

    def forward(self, e_t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            e_t: (batch, embedding_dim) — Risk Embedding từ DRRE.

        Returns:
            r_t: (batch,) — điểm rủi ro liên tục trong [0, 100].
        """
        return 100.0 * torch.sigmoid(self.logits(e_t))

    def logits(self, e_t: torch.Tensor) -> torch.Tensor:
        """Trả về logit thô (trước sigmoid) — dùng cho BCEWithLogitsLoss
        khi huấn luyện (src/drre/train.py), tránh phải tính ngược
        sigmoid^{-1}(forward(e_t)/100) kém ổn định số học.
        """
        return self.linear(e_t).squeeze(-1)


def classify_tier(
    score: float, low_max: int = 30, medium_max: int = 70
) -> RiskTier:
    """Phân loại điểm rủi ro thành Low/Medium/Critical theo ngưỡng config.yaml."""
    if score <= low_max:
        return RiskTier.LOW
    if score <= medium_max:
        return RiskTier.MEDIUM
    return RiskTier.CRITICAL


TIER_ACTIONS: dict[RiskTier, str] = {
    RiskTier.LOW: "Xử lý bình thường, không cần can thiệp thêm",
    RiskTier.MEDIUM: "Yêu cầu thêm bước xác thực hoặc theo dõi",
    RiskTier.CRITICAL: "Can thiệp ngay: xác minh bổ sung, tạm hoãn, hoặc chuyển đội xử lý gian lận",
}
