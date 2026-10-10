"""Giao thức đánh giá ba cấu hình + cổng tuyên bố (Mục 6.5, 6.6).

=== BA CẤU HÌNH ===

    (i)   real_only      — chỉ dữ liệu thực
    (ii)  rule_augmented — thực + dữ liệu từ simulator theo LUẬT
    (iii) llm_augmented   — thực + dữ liệu từ đặc tả do LLM đề xuất

Cấu hình (ii) là ĐƯỜNG CƠ SỞ BẮT BUỘC. Nếu thiếu nó, phần cải thiện của
(iii) so với (i) có thể chỉ do bản thân việc tăng cường dữ liệu.

=== CỔNG TUYÊN BỐ ===

Xem CLAIM_GATE. Chỉ được tuyên bố LLM có đóng góp khi (iii) vượt (ii) có
ý nghĩa thống kê trên dữ liệu thực giữ lại, qua nhiều seed.

=== DỮ LIỆU TỔNG HỢP KHÔNG BAO GIỜ DÙNG ĐỂ ĐÁNH GIÁ ===

assert_eval_set_is_real() chặn việc này. Mọi con số báo cáo đều tính trên
lát dữ liệu thực giữ lại, chia theo thời gian.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

CLAIM_GATE = (
    "Chỉ tuyên bố 'đặc tả do LLM đề xuất có đóng góp' khi cấu hình (iii) vượt cấu hình "
    "(ii) trên dữ liệu THỰC giữ lại, với khoảng tin cậy 95% của hiệu số KHÔNG chứa 0, "
    "trên ít nhất 3 seed. Nếu chưa đạt, phải trình bày đây là GIẢ THUYẾT THỰC NGHIỆM "
    "chưa kết luận (Mục 6.6)."
)

CONFIG_NAMES: tuple[str, str, str] = ("real_only", "rule_augmented", "llm_augmented")


# --------------------------------------------------------------------------
# Rào an toàn
# --------------------------------------------------------------------------

def assert_eval_set_is_real(
    eval_df: pd.DataFrame, is_synthetic_col: str = "is_augmented"
) -> None:
    """Khẳng định tập đánh giá không chứa dòng tổng hợp nào.

    Raises:
        ValueError: nếu phát hiện dòng tổng hợp trong tập đánh giá.
    """
    if is_synthetic_col in eval_df.columns:
        n = int(eval_df[is_synthetic_col].fillna(False).astype(bool).sum())
        if n:
            raise ValueError(
                f"Tập đánh giá chứa {n} dòng tổng hợp. Dữ liệu tổng hợp là nguồn bổ sung "
                "cho HUẤN LUYỆN, không phải căn cứ đánh giá (Mục 6.1)."
            )
    if "spec_id" in eval_df.columns and eval_df["spec_id"].notna().any():
        raise ValueError(
            "Tập đánh giá có dòng mang spec_id — đó là dòng tổng hợp. Xem Mục 6.1."
        )


def build_config_datasets(
    real_train: pd.DataFrame,
    rule_rows: pd.DataFrame | None,
    llm_rows: pd.DataFrame | None,
    inject_fn: Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Dựng ba tập huấn luyện tương ứng ba cấu hình.

    Cùng một lát dữ liệu thực được dùng cho cả ba — khác biệt duy nhất là
    phần tổng hợp được thêm vào, nên so sánh là công bằng.
    """
    out: dict[str, pd.DataFrame] = {"real_only": inject_fn(real_train, pd.DataFrame())}
    if rule_rows is not None and not rule_rows.empty:
        out["rule_augmented"] = inject_fn(real_train, rule_rows)
    if llm_rows is not None and not llm_rows.empty:
        out["llm_augmented"] = inject_fn(real_train, llm_rows)
    return out


# --------------------------------------------------------------------------
# Chỉ số (Mục 6.5)
# --------------------------------------------------------------------------

def pr_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(y_true.astype(int), scores))


def recall_at_budget(y_true: np.ndarray, scores: np.ndarray, budget: int) -> float:
    """Recall khi chỉ được cảnh báo `budget` giao dịch — khớp với ràng buộc
    năng lực điều tra trong QAPE (Mục 6.5). Đây là chỉ số gần nhất với
    giá trị vận hành: một ca trực chỉ xử lý được số cảnh báo hữu hạn.
    """
    y_true = y_true.astype(bool)
    if y_true.sum() == 0:
        return float("nan")
    budget = int(min(budget, len(scores)))
    top = np.argsort(-np.asarray(scores, dtype=float), kind="stable")[:budget]
    return float(y_true[top].sum() / y_true.sum())


def false_positive_rate_at_budget(
    y_true: np.ndarray, scores: np.ndarray, budget: int
) -> float:
    """Tỷ lệ cảnh báo giả trong `budget` cảnh báo đầu — theo dõi RIÊNG vì
    dữ liệu tổng hợp có thể làm bộ phát hiện quá nhạy (Mục 6.5)."""
    y_true = y_true.astype(bool)
    budget = int(min(budget, len(scores)))
    if budget == 0:
        return float("nan")
    top = np.argsort(-np.asarray(scores, dtype=float), kind="stable")[:budget]
    return float((~y_true[top]).sum() / budget)


def brier_score(y_true: np.ndarray, probs: np.ndarray) -> float:
    y = y_true.astype(float)
    p = np.clip(np.asarray(probs, dtype=float), 0.0, 1.0)
    return float(np.mean((p - y) ** 2))


def expected_calibration_error(
    y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10
) -> float:
    """ECE theo bin đều — độ hiệu chỉnh (Mục 6.5).

    Đầu vào `probs` phải là XÁC SUẤT trong [0,1]. Nếu đang có điểm 0–100
    thì chia 100 trước khi gọi.
    """
    y = y_true.astype(float)
    p = np.clip(np.asarray(probs, dtype=float), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=True), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        ece += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return float(ece)


def per_archetype_recall(
    df: pd.DataFrame, scores: np.ndarray, budget: int,
    archetype_col: str = "fraud_scenario_type",
) -> dict[str, float]:
    """Recall theo từng kiểu hình, ở cùng một ngân sách cảnh báo.

    Chỉ có nghĩa khi tập đánh giá có cột kiểu hình (vd. synthetic_vn giữ
    lại, hoặc dữ liệu thực đã được gán kiểu hình bằng tay).
    """
    if archetype_col not in df.columns:
        return {}
    y = df["label"].fillna(False).astype(bool).to_numpy()
    budget = int(min(budget, len(scores)))
    top = set(np.argsort(-np.asarray(scores, dtype=float), kind="stable")[:budget].tolist())
    out: dict[str, float] = {}
    arche = df[archetype_col].fillna("khong_xac_dinh").to_numpy()
    for name in sorted(set(arche[y])):
        rows = np.flatnonzero(y & (arche == name))
        if rows.size == 0:
            continue
        out[str(name)] = float(sum(1 for r in rows if r in top) / rows.size)
    return out


# --------------------------------------------------------------------------
# So sánh cấu hình + bootstrap
# --------------------------------------------------------------------------

@dataclass
class ConfigResult:
    """Kết quả một cấu hình trên một seed."""

    config: str
    seed: int
    scores: np.ndarray
    pr_auc: float
    recall_at_budget: float
    fpr_at_budget: float
    ece: float
    per_archetype: dict[str, float] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "config": self.config,
            "seed": self.seed,
            "pr_auc": self.pr_auc,
            "recall_at_budget": self.recall_at_budget,
            "fpr_at_budget": self.fpr_at_budget,
            "ece": self.ece,
            "per_archetype_recall": self.per_archetype,
            **({"extra": self.extra} if self.extra else {}),
        }


def evaluate_scores(
    config: str,
    seed: int,
    eval_df: pd.DataFrame,
    scores: np.ndarray,
    alert_budget: int,
    score_is_probability: bool = False,
) -> ConfigResult:
    """Tính toàn bộ chỉ số của Mục 6.5 cho một lần chạy.

    `eval_df` PHẢI là dữ liệu thực giữ lại — hàm tự kiểm tra bằng
    assert_eval_set_is_real.
    """
    assert_eval_set_is_real(eval_df)
    y = eval_df["label"].fillna(False).astype(bool).to_numpy()
    scores = np.asarray(scores, dtype=float)
    probs = scores if score_is_probability else np.clip(scores / 100.0, 0.0, 1.0)
    return ConfigResult(
        config=config,
        seed=seed,
        scores=scores,
        pr_auc=pr_auc(y, scores),
        recall_at_budget=recall_at_budget(y, scores, alert_budget),
        fpr_at_budget=false_positive_rate_at_budget(y, scores, alert_budget),
        ece=expected_calibration_error(y, probs),
        per_archetype=per_archetype_recall(eval_df, scores, alert_budget),
    )


def bootstrap_metric_diff(
    y_true: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 300,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Khoảng tin cậy bootstrap cho hiệu số metric(a) − metric(b).

    Lấy mẫu CÙNG chỉ số cho cả hai bên (paired bootstrap) để loại bớt
    phương sai do chọn mẫu đánh giá.

    Returns:
        (hiệu số điểm, biên dưới, biên trên).
    """
    y_true = np.asarray(y_true)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    point = metric(y_true, scores_a) - metric(y_true, scores_b)
    diffs = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if y_true[idx].sum() == 0:
            diffs[i] = np.nan
            continue
        diffs[i] = metric(y_true[idx], scores_a[idx]) - metric(y_true[idx], scores_b[idx])
    valid = diffs[~np.isnan(diffs)]
    if valid.size == 0:
        return point, float("nan"), float("nan")
    lo, hi = np.quantile(valid, [alpha / 2, 1 - alpha / 2])
    return float(point), float(lo), float(hi)


@dataclass
class ClaimDecision:
    """Phán quyết của cổng tuyên bố — đưa nguyên văn vào báo cáo/slide."""

    can_claim: bool
    diff: float
    ci: tuple[float, float]
    n_seeds: int
    message: str

    def to_json(self) -> dict[str, Any]:
        return {
            "can_claim_llm_contribution": self.can_claim,
            "diff_llm_minus_rule": self.diff,
            "ci95": list(self.ci),
            "n_seeds": self.n_seeds,
            "gate": CLAIM_GATE,
            "message": self.message,
        }


def decide_llm_claim(
    y_true: np.ndarray,
    llm_scores_by_seed: Sequence[np.ndarray],
    rule_scores_by_seed: Sequence[np.ndarray],
    metric: Callable[[np.ndarray, np.ndarray], float] = pr_auc,
    min_seeds: int = 3,
    n_boot: int = 300,
    seed: int = 0,
) -> ClaimDecision:
    """Áp dụng CLAIM_GATE: (iii) có thật sự vượt (ii) hay không.

    Điểm của nhiều seed được lấy trung bình theo từng giao dịch trước khi
    bootstrap — cách này giữ được tính ghép cặp (paired) giữa hai cấu hình.
    """
    n_seeds = min(len(llm_scores_by_seed), len(rule_scores_by_seed))
    if n_seeds == 0:
        return ClaimDecision(
            False, float("nan"), (float("nan"), float("nan")), 0,
            "Chưa có lần chạy nào — không thể tuyên bố gì.")

    llm_mean = np.mean(np.vstack([np.asarray(s, dtype=float) for s in llm_scores_by_seed]), axis=0)
    rule_mean = np.mean(np.vstack([np.asarray(s, dtype=float) for s in rule_scores_by_seed]), axis=0)
    diff, lo, hi = bootstrap_metric_diff(
        y_true, llm_mean, rule_mean, metric, n_boot=n_boot, seed=seed)

    enough_seeds = n_seeds >= min_seeds
    excludes_zero = not (np.isnan(lo) or np.isnan(hi)) and lo > 0.0
    can_claim = bool(enough_seeds and excludes_zero)

    if can_claim:
        message = (
            f"(iii) vượt (ii): hiệu số {diff:+.4f}, KTC 95% [{lo:+.4f}; {hi:+.4f}], "
            f"{n_seeds} seed → ĐƯỢC tuyên bố đặc tả do LLM đề xuất có đóng góp."
        )
    elif not enough_seeds:
        message = (
            f"Chỉ có {n_seeds} seed (cần ≥ {min_seeds}). Hiệu số hiện tại {diff:+.4f}, "
            f"KTC 95% [{lo:+.4f}; {hi:+.4f}] — CHƯA đủ điều kiện tuyên bố."
        )
    else:
        message = (
            f"Hiệu số {diff:+.4f}, KTC 95% [{lo:+.4f}; {hi:+.4f}] có chứa 0 hoặc âm → "
            "KHÔNG tuyên bố LLM có đóng góp; trình bày như giả thuyết chưa kết luận."
        )
    return ClaimDecision(can_claim, diff, (lo, hi), n_seeds, message)
