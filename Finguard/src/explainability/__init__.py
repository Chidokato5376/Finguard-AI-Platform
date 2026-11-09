"""Giải thích kết quả — hai đường cho hai họ mô hình.

- `explain`     : giải thích NỘI SINH cho DRRE, bằng chính thứ mô hình tính
                  ra (trọng số attention alpha_k, trọng số gate beta). Dùng
                  `explain_scored_row` khi chỉ có khung đã chấm điểm.
- `attribution` : quy kết đặc trưng cho họ mô hình cây (GBDT, Random Forest),
                  bằng hoán vị trên AUPRC, hoặc SHAP nếu môi trường có.

Cần cả hai: DRRE giải thích được bằng cơ chế của chính nó, nhưng mốc đối
chứng mạnh nhất lại thuộc họ cây, nên nếu chỉ có đường thứ nhất thì phần
"giải thích được kết quả" chỉ đúng với nhánh không được chọn.
"""

from src.explainability.attribution import (
    FeatureAttribution,
    global_attribution,
    shap_available,
)
from src.explainability.explain import (
    SCORED_EXPLANATION_COLUMNS,
    RiskExplanation,
    build_explanation,
    explain_behavior_signal,
    explain_fusion_signal,
    explain_scored_row,
)

__all__ = [
    "FeatureAttribution",
    "RiskExplanation",
    "SCORED_EXPLANATION_COLUMNS",
    "build_explanation",
    "explain_behavior_signal",
    "explain_fusion_signal",
    "explain_scored_row",
    "global_attribution",
    "shap_available",
]
