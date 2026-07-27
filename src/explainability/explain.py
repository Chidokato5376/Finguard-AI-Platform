"""Explainable AI (Module 6).

Giải thích nội sinh từ ba nguồn — KHÔNG phải lớp diễn giải gắn thêm:
  1. Trọng số attention alpha_k (Mục 4.2) -> lệch hành vi lịch sử
  2. Trọng số gate beta (Mục 4.4)         -> nguồn tín hiệu chủ đạo
  3. Attention trên đồ thị (GAT edge weight) -> tài khoản liên quan mạnh nhất

SHAP chỉ dùng cho đầu dò bán giám sát phụ, KHÔNG phải nguồn giải thích
chính (xem docs/architecture.md, Module 6).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RiskExplanation:
    """Kết quả giải thích cho một giao dịch — hiển thị trực tiếp lên Dashboard."""

    max_attention_slot_weight: float          # max_k(alpha_k)
    behavior_deviation_summary: str            # diễn giải mức khớp với lịch sử
    gate_weights: dict[str, float]              # {beta_x, beta_h, beta_g}
    dominant_signal_summary: str                # nguồn tín hiệu chủ đạo
    top_related_account: str | None = None      # tài khoản có edge weight cao nhất (GAT)
    top_related_account_weight: float | None = None


def explain_behavior_signal(alpha_k: torch.Tensor) -> tuple[float, str]:
    """Diễn giải trọng số attention -> câu giải thích cho chuyên viên rủi ro."""
    max_alpha = alpha_k.max(dim=-1).values.item()
    if max_alpha < 0.2:
        summary = (
            f"Giao dịch không khớp với bất kỳ mẫu hành vi lịch sử nào "
            f"(max alpha_k = {max_alpha:.2f})"
        )
    else:
        summary = f"Giao dịch khớp với một mẫu hành vi đã biết (max alpha_k = {max_alpha:.2f})"
    return max_alpha, summary


def explain_fusion_signal(beta: torch.Tensor) -> tuple[dict[str, float], str]:
    """Diễn giải trọng số gate -> nguồn tín hiệu chủ đạo."""
    beta_x, beta_h, beta_g = beta[0].item(), beta[1].item(), beta[2].item()
    weights = {"beta_transaction": beta_x, "beta_behavior": beta_h, "beta_graph": beta_g}
    dominant = max(weights, key=weights.get)  # type: ignore[arg-type]

    labels = {
        "beta_transaction": "đặc trưng giao dịch thô",
        "beta_behavior": "lịch sử hành vi cá nhân",
        "beta_graph": "quan hệ mạng lưới tài khoản",
    }
    summary = f"Hệ thống dựa chủ yếu vào {labels[dominant]} ({dominant} = {weights[dominant]:.2f})"
    return weights, summary


def build_explanation(
    alpha_k: torch.Tensor,
    beta: torch.Tensor,
    top_related_account: str | None = None,
    top_related_account_weight: float | None = None,
) -> RiskExplanation:
    """Tổng hợp giải thích đầy đủ cho một giao dịch — dùng cho Module 7 + Dashboard."""
    max_alpha, behavior_summary = explain_behavior_signal(alpha_k)
    gate_weights, dominant_summary = explain_fusion_signal(beta)

    return RiskExplanation(
        max_attention_slot_weight=max_alpha,
        behavior_deviation_summary=behavior_summary,
        gate_weights=gate_weights,
        dominant_signal_summary=dominant_summary,
        top_related_account=top_related_account,
        top_related_account_weight=top_related_account_weight,
    )
