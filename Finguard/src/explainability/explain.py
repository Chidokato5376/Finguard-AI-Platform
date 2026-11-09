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
from typing import Mapping

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


#: Tên các cột mà `src/dashboard/scoring_service.py::_score_with_drre` ghi ra
#: sau khi chấm điểm. Đây là cầu nối giữa mô hình và lời giải thích: ba trọng
#: số gate và trọng số attention lớn nhất đều đã được tính sẵn lúc chấm, chỉ
#: chưa ai đọc chúng.
SCORED_EXPLANATION_COLUMNS = (
    "max_attention_weight", "beta_transaction", "beta_behavior", "beta_graph",
)


def explain_scored_row(
    row: Mapping[str, float],
    top_related_account: str | None = None,
    top_related_account_weight: float | None = None,
) -> RiskExplanation:
    """Giải thích một dòng ĐÃ CHẤM ĐIỂM, không cần tensor và không cần torch.

    VÌ SAO CÓ HÀM NÀY: `build_explanation` nhận tensor nên chỉ gọi được từ
    bên trong vòng lặp mô hình. Dashboard thì chỉ có khung dữ liệu đã chấm.
    Trước đây vì thiếu đúng một cầu nối như thế này mà cả module giải thích
    nội sinh KHÔNG được file nào gọi tới, còn dashboard thì giải thích bằng
    một bộ luật viết tay (`src/dashboard/story.py::explain`) — tức là thứ
    hiển thị cho người dùng không phải thứ mô hình thật sự tính.

    Args:
        row: một dòng của khung đã chấm, cần đủ `SCORED_EXPLANATION_COLUMNS`.
        top_related_account: tài khoản liên quan mạnh nhất, nếu tra được.
        top_related_account_weight: trọng số cạnh tương ứng.

    Raises:
        KeyError: thiếu cột — nói rõ thiếu cột nào, vì nguyên nhân gần như
            luôn là khung được chấm bằng nhánh quy tắc thống kê chứ không
            phải bằng DRRE.
    """
    missing = [c for c in SCORED_EXPLANATION_COLUMNS if c not in row]
    if missing:
        raise KeyError(
            f"Dòng thiếu cột giải thích: {missing}. Các cột này chỉ có khi khung "
            "được chấm bằng DRRE thật (method='drre_trained'); nhánh quy tắc "
            "thống kê không sinh ra chúng."
        )

    max_alpha = float(row["max_attention_weight"])
    if max_alpha < 0.2:
        behavior_summary = (
            f"Giao dịch không khớp với bất kỳ mẫu hành vi lịch sử nào "
            f"(max alpha_k = {max_alpha:.2f})"
        )
    else:
        behavior_summary = f"Giao dịch khớp với một mẫu hành vi đã biết (max alpha_k = {max_alpha:.2f})"

    weights = {
        "beta_transaction": float(row["beta_transaction"]),
        "beta_behavior": float(row["beta_behavior"]),
        "beta_graph": float(row["beta_graph"]),
    }
    labels = {
        "beta_transaction": "đặc trưng giao dịch thô",
        "beta_behavior": "lịch sử hành vi cá nhân",
        "beta_graph": "quan hệ mạng lưới tài khoản",
    }
    dominant = max(weights, key=weights.get)  # type: ignore[arg-type]
    dominant_summary = (
        f"Hệ thống dựa chủ yếu vào {labels[dominant]} ({dominant} = {weights[dominant]:.2f})"
    )

    return RiskExplanation(
        max_attention_slot_weight=max_alpha,
        behavior_deviation_summary=behavior_summary,
        gate_weights=weights,
        dominant_signal_summary=dominant_summary,
        top_related_account=top_related_account,
        top_related_account_weight=top_related_account_weight,
    )


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
