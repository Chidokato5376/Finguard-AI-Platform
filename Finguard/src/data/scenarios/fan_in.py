"""FAN-IN: nhiều nguồn cùng đổ tiền về một người nhận."""

from __future__ import annotations

import numpy as np

from src.data.scenarios.stream import DAY, MONEY_SCALE, Stream, log1p_f32

NAME = "fan_in"
VERSION = 1
TARGETS = ("FAN-IN",)
FEATURES = ["fi_log_n_send_24h", "fi_log_amt_in_24h", "fi_new_sender_ratio_24h"]


def compute(s: Stream) -> dict[str, np.ndarray]:
    lo, hi = s.into.bounds(s.q_r, DAY)
    count = (hi - lo).astype(np.float64)
    new_senders = s.into.sum(s.first_pair, lo, hi).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(count > 0, new_senders / count, 0.0)
    return {
        "fi_log_n_send_24h": log1p_f32(s.in_distinct.count(s.q_r, DAY)),
        "fi_log_amt_in_24h": log1p_f32(s.into.sum(s.cents, lo, hi) / MONEY_SCALE),
        "fi_new_sender_ratio_24h": ratio.astype(np.float32),
    }
