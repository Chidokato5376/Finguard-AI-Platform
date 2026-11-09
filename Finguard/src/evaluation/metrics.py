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


def confusion_from_mask(y_true: np.ndarray, flagged: np.ndarray) -> dict[str, float]:
    """Ma trận nhầm lẫn và các tỷ lệ suy ra từ nó, cho một tập cảnh báo đã chọn."""
    y = np.asarray(y_true, dtype=bool)
    flagged = np.asarray(flagged, dtype=bool)
    tp, fp = int((flagged & y).sum()), int((flagged & ~y).sum())
    fn, tn = int((~flagged & y).sum()), int((~flagged & ~y).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "alerts": tp + fp,
            "precision": precision, "recall": recall, "f1": f1}


def confusion_at_budget(y_true: np.ndarray, y_score: np.ndarray, budget: int) -> dict[str, float]:
    """Ma trận nhầm lẫn khi đội điều tra xem đúng `budget` giao dịch nghi nhất."""
    order = np.argsort(-np.asarray(y_score, dtype=float), kind="stable")
    flagged = np.zeros(len(order), dtype=bool)
    flagged[order[:budget]] = True
    return confusion_from_mask(y_true, flagged)


def confusion_at_threshold(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, float]:
    out = confusion_from_mask(y_true, np.asarray(y_score, dtype=float) >= threshold)
    out["threshold"] = float(threshold)
    return out


def best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Ngưỡng cho F1 cao nhất. CHỈ chọn trên val, rồi giữ nguyên khi đo trên test."""
    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.argmax(f1))])


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
