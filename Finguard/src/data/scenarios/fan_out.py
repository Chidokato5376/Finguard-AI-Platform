"""FAN-OUT: một tài khoản chia tiền cho nhiều người nhận trong thời gian ngắn."""

from __future__ import annotations

import numpy as np

from src.data.scenarios.stream import DAY, LOG_SCALE, Stream, log1p_f32

NAME = "fan_out"
VERSION = 1
TARGETS = ("FAN-OUT",)
FEATURES = ["fo_log_n_recv_24h", "fo_log_n_new_recv_24h", "fo_amt_std_24h", "fo_amt_dev_24h"]


def compute(s: Stream) -> dict[str, np.ndarray]:
    lo, hi = s.out.bounds(s.q_s, DAY)
    count, mean, std = s.window_stats(s.out, lo, hi)
    return {
        "fo_log_n_recv_24h": log1p_f32(s.out_distinct.count(s.q_s, DAY)),
        "fo_log_n_new_recv_24h": log1p_f32(s.out.sum(s.first_pair, lo, hi)),
        # −1: chưa đủ giao dịch để nói về độ đều. 0 có nghĩa là các khoản giống hệt nhau.
        "fo_amt_std_24h": np.where(count > 1, std, -1.0).astype(np.float32),
        "fo_amt_dev_24h": np.where(
            count > 0, np.abs(s.log_usd / LOG_SCALE - mean), -1.0).astype(np.float32),
    }
