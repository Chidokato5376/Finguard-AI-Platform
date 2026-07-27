"""Chỉ số đánh giá hiệu quả — KPI (Mục 11).

- AUPRC (không dùng Accuracy/AUC-ROC do mất cân bằng nhãn nặng)
- So sánh DRRE vs ensemble baseline (XGBoost/LightGBM/Autoencoder)
- Hiệu năng trên cold-start subset
- Độ ổn định khi concept drift (suy giảm Recall)
- QAOA approximation ratio + tỷ lệ vi phạm ràng buộc
- Decision Agreement Rate
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, recall_score


def compute_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """AUPRC — chỉ số chính cho bài toán mất cân bằng nhãn nặng (Mục 11)."""
    return float(average_precision_score(y_true, y_score))


def compute_cold_start_auprc(
    y_true: np.ndarray, y_score: np.ndarray, is_cold_start: np.ndarray
) -> float:
    """AUPRC riêng trên tập con khách hàng mới — kiểm chứng luận điểm
    "không dùng ngưỡng chung" (Adaptive Fusion, Mục 4.4).
    """
    mask = is_cold_start.astype(bool)
    if mask.sum() == 0:
        raise ValueError("Không có mẫu cold-start nào trong tập dữ liệu.")
    return compute_auprc(y_true[mask], y_score[mask])


def compute_concept_drift_recall_drop(
    y_true_before: np.ndarray,
    y_score_before: np.ndarray,
    y_true_after: np.ndarray,
    y_score_after: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """Đo mức độ suy giảm Recall khi chèn một fraud pattern mới không
    xuất hiện trong tập huấn luyện (concept drift test, Mục 11).

    Returns:
        Chênh lệch recall_before - recall_after (dương = suy giảm).
    """
    recall_before = recall_score(y_true_before, (y_score_before >= threshold).astype(int))
    recall_after = recall_score(y_true_after, (y_score_after >= threshold).astype(int))
    return float(recall_before - recall_after)


def compute_qaoa_approximation_ratio(qaoa_objective: float, ilp_optimal_objective: float) -> float:
    """Approximation ratio = objective(QAOA) / objective(ILP optimal)."""
    if ilp_optimal_objective <= 0:
        raise ValueError("ilp_optimal_objective phải dương.")
    return qaoa_objective / ilp_optimal_objective


def compute_decision_agreement_rate(
    system_recommendations: list[str], expert_decisions: list[str]
) -> float:
    """Tỷ lệ chuyên viên đồng thuận với đề xuất hệ thống — phản ánh chất
    lượng giải thích của Module 6.
    """
    if len(system_recommendations) != len(expert_decisions):
        raise ValueError("Hai danh sách phải cùng độ dài (cùng batch quyết định).")
    if not system_recommendations:
        return 0.0
    agreements = sum(
        1 for s, e in zip(system_recommendations, expert_decisions) if s == e
    )
    return agreements / len(system_recommendations)
