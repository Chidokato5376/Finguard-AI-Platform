"""Xuyên ngân hàng và quy đổi tiền tệ.

KHÔNG dùng cờ đổi tiền `is_fx`: trên AMLWorld nó là dấu hiệu loại trừ do máy sinh gây ra
(xem src/data/features.py). Tỷ giá chỉ dựng từ giai đoạn train (scenarios/fx.py).
"""

from __future__ import annotations

import numpy as np

from src.data.scenarios.stream import WEEK, DistinctIndex, Stream, log1p_f32

NAME = "cross_bank_fx"
VERSION = 1
TARGETS = ("FAN-OUT", "FAN-IN", "CYCLE", "RANDOM")
FEATURES = ["cb_cross_bank", "cb_log_n_banks_7d", "cb_log_amount_usd"]


def compute(s: Stream) -> dict[str, np.ndarray]:
    sender_bank, receiver_bank = s.bank[s.s], s.bank[s.r]
    index = DistinctIndex(s.s, receiver_bank, s.t, s.base)
    return {
        "cb_cross_bank": (sender_bank != receiver_bank).astype(np.float32),
        "cb_log_n_banks_7d": log1p_f32(index.count(s.q_s, WEEK)),
        "cb_log_amount_usd": log1p_f32(s.usd),
    }
