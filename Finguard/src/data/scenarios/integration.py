"""Khoản hợp nhất: lệch khỏi nhịp chi đều đặn, khoản lớn lần đầu cho đối tác mới."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.scenarios.stream import LOG_SCALE, Stream, window_stats_sorted

NAME = "integration"
VERSION = 1
TARGETS = ("RANDOM", "GATHER-SCATTER")
FEATURES = ["in_gap_z", "in_log_ratio_prior_max", "in_large_first_time"]
MIN_HISTORY = 5


def compute(s: Stream) -> dict[str, np.ndarray]:
    start, end = s.out.bounds(s.q_s, None)
    has_prior = end > start
    last = np.maximum(end - 1, 0)

    same_entity = np.r_[False, s.out.entity[1:] == s.out.entity[:-1]]
    gaps = np.r_[0, np.diff(s.out.time)] * same_entity
    log_gaps = np.rint(np.log1p(gaps) * LOG_SCALE).astype(np.int64)
    # Sự kiện đầu của mỗi tài khoản không có khoảng cách nào.
    count, mean, std = window_stats_sorted(log_gaps, np.minimum(start + 1, end), end)
    current = np.log1p(np.maximum(s.t - s.out.time[last], 0))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(
            has_prior & (count >= 3) & (std > 0), np.clip((current - mean) / std, -5, 5), 0.0)

    ordered = s.log_usd[s.out.order]
    prior_max = pd.Series(ordered).groupby(s.out.entity, sort=False).cummax().to_numpy()[last]
    ratio = np.where(has_prior, np.clip((s.log_usd - prior_max) / LOG_SCALE, -10, 10), 0.0)
    established = (end - start) >= MIN_HISTORY
    return {
        "in_gap_z": z.astype(np.float32),
        "in_log_ratio_prior_max": ratio.astype(np.float32),
        "in_large_first_time": (
            established & (s.first_pair == 1) & (s.log_usd > prior_max)).astype(np.float32),
    }
