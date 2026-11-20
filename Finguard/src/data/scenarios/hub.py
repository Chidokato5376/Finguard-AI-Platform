"""Tài khoản trung tâm: số đối tác khác nhau, tính cả hai chiều, của hai đầu giao dịch."""

from __future__ import annotations

import numpy as np

from src.data.scenarios.stream import WEEK, DistinctIndex, Stream, log1p_f32

NAME = "hub"
VERSION = 1
TARGETS = ("FAN-OUT", "FAN-IN", "GATHER-SCATTER", "SCATTER-GATHER")
FEATURES = ["hub_log_deg_sender_7d", "hub_log_deg_receiver_7d"]


def compute(s: Stream) -> dict[str, np.ndarray]:
    index = DistinctIndex(np.r_[s.s, s.r], np.r_[s.r, s.s], np.r_[s.t, s.t], s.base)
    return {
        "hub_log_deg_sender_7d": log1p_f32(index.count(s.q_s, WEEK)),
        "hub_log_deg_receiver_7d": log1p_f32(index.count(s.q_r, WEEK)),
    }
