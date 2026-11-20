"""Chia nhỏ tiền mặt để lách ngưỡng báo cáo (mốc 10.000 USD)."""

from __future__ import annotations

import numpy as np

from src.data.scenarios.stream import DAY, MONEY_SCALE, WEEK, DistinctIndex, Stream, log1p_f32

NAME = "cash_structuring"
VERSION = 1
TARGETS = ("RANDOM", "FAN-OUT")
FEATURES = [
    "cs_log_cash_24h", "cs_cash_ratio_ctr", "cs_log_n_near_ctr_7d", "cs_log_n_channels_7d",
]
THRESHOLD = 10_000.0
NEAR_BELOW = 8_000.0


def compute(s: Stream) -> dict[str, np.ndarray]:
    code = s.channel_names.index("cash") if "cash" in s.channel_names else -1
    is_cash = s.channel == code
    lo, hi = s.out.bounds(s.q_s, DAY)
    prior = s.out.sum(s.cents * is_cash, lo, hi) / MONEY_SCALE
    # Khoản hiện tại đã biết tại thời điểm chấm nên được cộng vào.
    total = prior + np.where(is_cash, s.usd, 0.0)

    near = ((s.usd >= NEAR_BELOW) & (s.usd < THRESHOLD)).astype(np.int64)
    week_lo, week_hi = s.out.bounds(s.q_s, WEEK)
    channels = DistinctIndex(s.s, s.channel, s.t, s.base).count(s.q_s, WEEK)
    return {
        "cs_log_cash_24h": log1p_f32(prior),
        "cs_cash_ratio_ctr": np.clip(total / THRESHOLD, 0, 5).astype(np.float32),
        "cs_log_n_near_ctr_7d": log1p_f32(s.out.sum(near, week_lo, week_hi)),
        "cs_log_n_channels_7d": log1p_f32(channels),
    }
