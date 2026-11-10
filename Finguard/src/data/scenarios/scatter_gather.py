"""SCATTER-GATHER: chia ra nhiều trung gian rồi gom về một điểm.

Cố ý KHÔNG có đặc trưng thủ công (docs/lo-trinh-train-dot-2.md): nhận dạng đòi hỏi đường
hai bước qua trung gian, đúng phần giao cho nhánh đồ thị của DRRE. Giữ mô-đun rỗng để
kiểu hình này vẫn có mặt trong bảng đo R1 và để điều kiện thành công của DRRE (recall
SCATTER-GATHER cao hơn đối chứng) được đo trên đối chứng chưa nhận đường hai bước.
"""

from __future__ import annotations

import numpy as np

from src.data.scenarios.stream import Stream

NAME = "scatter_gather"
VERSION = 1
TARGETS = ("SCATTER-GATHER",)
FEATURES: list[str] = []


def compute(s: Stream) -> dict[str, np.ndarray]:
    return {}
