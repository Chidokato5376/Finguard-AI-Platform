"""BIPARTITE / STACK: nhiều tài khoản hành động đồng bộ, số tiền gần giống nhau."""

from __future__ import annotations

import numpy as np

from src.data.scenarios.stream import HOUR, LOG_SCALE, Stream, log1p_f32

NAME = "synchronized"
VERSION = 1
TARGETS = ("BIPARTITE", "STACK")
FEATURES = [
    "sy_log_n_send_6h", "sy_amt_std_in_6h", "sy_log_same_amt_in_6h", "sy_log_same_amt_out_6h",
    "sy_log_same_amt_system_1h",
]
WINDOW = 6 * HOUR
#: Bề rộng ô số tiền trên thang log: khoảng ±2,5% quanh tâm ô.
BUCKET = int(0.05 * LOG_SCALE)


def _count_same(s: Stream, group: np.ndarray, window: int) -> np.ndarray:
    index = s.event_index(group)
    lo, hi = index.bounds(s.query(group), window)
    return hi - lo


def compute(s: Stream) -> dict[str, np.ndarray]:
    bucket = s.log_usd // BUCKET
    width = int(bucket.max(initial=0)) + 1
    lo, hi = s.into.bounds(s.q_r, WINDOW)
    count, _, std = s.window_stats(s.into, lo, hi)
    return {
        "sy_log_n_send_6h": log1p_f32(s.in_distinct.count(s.q_r, WINDOW)),
        "sy_amt_std_in_6h": np.where(count > 1, std, -1.0).astype(np.float32),
        "sy_log_same_amt_in_6h": log1p_f32(_count_same(s, s.r * width + bucket, WINDOW)),
        "sy_log_same_amt_out_6h": log1p_f32(_count_same(s, s.s * width + bucket, WINDOW)),
        "sy_log_same_amt_system_1h": log1p_f32(_count_same(s, s.channel * width + bucket, HOUR)),
    }
