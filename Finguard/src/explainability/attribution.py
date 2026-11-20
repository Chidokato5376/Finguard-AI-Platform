"""Quy kết đặc trưng cho các mô hình cổ điển (GBDT, Random Forest, cây).

=== VÌ SAO TÁCH KHỎI explain.py ===

`explain.py` giải thích DRRE bằng chính thứ DRRE tính ra (trọng số attention,
trọng số gate). Cách đó chỉ áp dụng được cho DRRE. Nhưng mốc đối chứng mạnh
nhất đo được cho tới nay là GBDT có cân bằng lớp, và phương án lai được đề
xuất lấy GBDT làm xương sống — nên phải có đường giải thích cho họ mô hình
cây nữa, nếu không thì phần "giải thích được kết quả" chỉ đúng với nhánh
không được chọn.

=== HAI PHƯƠNG PHÁP, VÀ VÌ SAO MẶC ĐỊNH LÀ HOÁN VỊ ===

- `permutation`: xáo trộn từng cột rồi đo ĐỘ TỤT của chỉ số. Luôn dùng được
  (chỉ cần scikit-learn), đúng với mọi mô hình, và quan trọng nhất: nó đo
  trên CHÍNH chỉ số ta quan tâm. Mặc định ở đây là `average_precision`
  (AUPRC) chứ không phải accuracy — với tỷ lệ nền 0,1%, accuracy gần 100%
  cho mọi thứ nên quy kết theo accuracy là vô nghĩa.
- `shap`: đóng góp theo từng dòng, cộng lại đúng bằng dự đoán. Chính xác hơn
  và giải thích được TỪNG cảnh báo. `shap>=0.45` CÓ khai báo trong
  requirements.txt, nhưng tính đến 2026-10-19 nó CHƯA được cài trong môi
  trường `.venv` của nhóm — nên code ở đây không được phép giả định có nó.

`method="auto"` dùng shap khi có, ngược lại rơi về hoán vị. Kết quả luôn ghi
rõ phương pháp nào đã chạy, để không ai đọc nhầm một bảng hoán vị thành SHAP.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)

#: Số dòng tối đa đưa vào tính quy kết. Hoán vị phải chấm lại toàn bộ tập
#: một lần cho MỖI cột và MỖI lần lặp, nên trên 760.000 dòng × 14 cột × 5
#: lần là 53 triệu lượt chấm — vài chục phút mà không thêm thông tin gì so
#: với một mẫu đủ lớn.
DEFAULT_MAX_ROWS = 50_000


def shap_available() -> bool:
    """`shap` có cài trong môi trường hay không."""
    try:  # pragma: no cover - phụ thuộc môi trường
        import shap  # noqa: F401

        return True
    except Exception:
        return False


@dataclass(frozen=True)
class FeatureAttribution:
    """Mức đóng góp của từng đặc trưng, đã sắp giảm dần."""

    method: str
    feature_names: tuple[str, ...]
    importance: tuple[float, ...]
    scoring: str | None = None
    n_rows: int = 0

    def top(self, k: int = 10) -> list[tuple[str, float]]:
        """k đặc trưng đóng góp nhiều nhất."""
        order = np.argsort(-np.asarray(self.importance))[:k]
        return [(self.feature_names[i], float(self.importance[i])) for i in order]

    def as_markdown(self, k: int = 10) -> str:
        """Bảng markdown để dán thẳng vào báo cáo."""
        unit = f"độ tụt {self.scoring}" if self.scoring else "đóng góp trung bình |SHAP|"
        lines = [
            f"| Đặc trưng | {unit} |",
            "|---|---|",
        ]
        lines += [f"| `{name}` | {value:.5f} |" for name, value in self.top(k)]
        return "\n".join(lines)


def global_attribution(
    estimator,
    x: np.ndarray,
    y: np.ndarray | None = None,
    feature_names: Sequence[str] | None = None,
    method: str = "auto",
    scoring: str = "average_precision",
    n_repeats: int = 5,
    max_rows: int = DEFAULT_MAX_ROWS,
    seed: int = 42,
) -> FeatureAttribution:
    """Mức đóng góp toàn cục của từng đặc trưng.

    Args:
        estimator: mô hình đã fit (bất kỳ mô hình sklearn nào có predict_proba).
        x: ma trận đặc trưng.
        y: nhãn — BẮT BUỘC với phương pháp hoán vị, không cần với shap.
        feature_names: tên cột; thiếu thì đánh số.
        method: "auto" | "permutation" | "shap".
        scoring: chỉ số dùng cho hoán vị. Mặc định `average_precision` =
            AUPRC, đúng chỉ số dự án báo cáo.
        n_repeats: số lần xáo mỗi cột.
        max_rows: lấy mẫu ngẫu nhiên tối đa bấy nhiêu dòng.
        seed: hạt giống cho cả việc lấy mẫu và việc xáo.

    Raises:
        ValueError: chọn hoán vị mà không có nhãn, hoặc chọn shap mà chưa cài.
    """
    x = np.asarray(x)
    names = tuple(feature_names) if feature_names is not None else tuple(
        f"f{i}" for i in range(x.shape[1]))
    if len(names) != x.shape[1]:
        raise ValueError(
            f"Có {len(names)} tên đặc trưng nhưng ma trận có {x.shape[1]} cột.")

    rng = np.random.default_rng(seed)
    if len(x) > max_rows:
        picked = rng.choice(len(x), size=max_rows, replace=False)
        x = x[picked]
        y = None if y is None else np.asarray(y)[picked]
        logger.info("Lấy mẫu %d/%d dòng để tính quy kết", max_rows, len(picked))

    if method == "auto":
        method = "shap" if shap_available() else "permutation"

    if method == "shap":
        if not shap_available():
            raise ValueError(
                "Chưa cài `shap`. Cài bằng `pip install shap`, hoặc dùng "
                "method='permutation' (không cần gói thêm)."
            )
        import shap  # pragma: no cover - phụ thuộc gói tùy chọn

        explainer = shap.Explainer(estimator, x)
        values = explainer(x).values
        if values.ndim == 3:  # (n, features, classes) -> lấy lớp dương
            values = values[:, :, -1]
        importance = np.abs(values).mean(axis=0)
        return FeatureAttribution(
            method="shap", feature_names=names,
            importance=tuple(float(v) for v in importance), n_rows=len(x),
        )

    if method != "permutation":
        raise ValueError(f"method phải là 'auto' | 'permutation' | 'shap', nhận {method!r}")
    if y is None:
        raise ValueError("Quy kết bằng hoán vị cần nhãn `y` để đo độ tụt của chỉ số.")

    from sklearn.inspection import permutation_importance

    result = permutation_importance(
        estimator, x, np.asarray(y), scoring=scoring,
        n_repeats=n_repeats, random_state=seed, n_jobs=1,
    )
    return FeatureAttribution(
        method="permutation", feature_names=names,
        importance=tuple(float(v) for v in result.importances_mean),
        scoring=scoring, n_rows=len(x),
    )
