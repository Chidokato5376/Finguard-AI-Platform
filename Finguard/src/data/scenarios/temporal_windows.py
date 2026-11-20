"""Cửa sổ 1 giờ / 24 giờ / 7 ngày, thay cho đại lượng cộng dồn từ đầu chuỗi.

Đại lượng cộng dồn trôi theo thời gian (new_counterparty 18,8% ở train còn 2,2% ở test),
nên mô hình học trên train gặp một phân bố khác ở test. Cửa sổ cố định không trôi.
"""

from __future__ import annotations

import numpy as np

from src.data.scenarios.stream import DAY, HOUR, MONEY_SCALE, WEEK, Stream, log1p_f32

NAME = "temporal_windows"
VERSION = 1
TARGETS = ("ALL",)
WINDOWS = {"1h": HOUR, "24h": DAY, "7d": WEEK}
FEATURES = [
    f"tw_{name}_{tag}"
    for tag in WINDOWS
    for name in ("log_n_out", "log_amt_out", "log_n_in", "log_amt_in", "log_n_recv")
]


def compute(s: Stream) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for tag, window in WINDOWS.items():
        lo, hi = s.out.bounds(s.q_s, window)
        out[f"tw_log_n_out_{tag}"] = log1p_f32(hi - lo)
        out[f"tw_log_amt_out_{tag}"] = log1p_f32(s.out.sum(s.cents, lo, hi) / MONEY_SCALE)
        lo, hi = s.into.bounds(s.q_s, window)
        out[f"tw_log_n_in_{tag}"] = log1p_f32(hi - lo)
        out[f"tw_log_amt_in_{tag}"] = log1p_f32(s.into.sum(s.cents, lo, hi) / MONEY_SCALE)
        lo, hi = s.into.bounds(s.q_r, window)
        out[f"tw_log_n_recv_{tag}"] = log1p_f32(hi - lo)
    return out
