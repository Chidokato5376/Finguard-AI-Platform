"""Chuỗi mule: tiền vừa vào đã bị chuyển tiếp gần như nguyên vẹn."""

from __future__ import annotations

import numpy as np

from src.data.scenarios.stream import DAY, LOG_SCALE, Stream, log1p_f32

NAME = "mule_chain"
#: v2: tuổi tài khoản chặn ở AGE_CAP. Bản v1 không chặn nên với tài khoản có từ ngày đầu,
#: tuổi = thời gian kể từ lúc bộ dữ liệu bắt đầu (trung vị 2,6 ngày ở train, 7,3 ngày ở val):
#: một chiếc đồng hồ, không phải hành vi. Tín hiệu thật là "tài khoản mới" (R1, 10/11/2026).
VERSION = 2
TARGETS = ("STACK", "GATHER-SCATTER", "RANDOM")
AGE_CAP = 3 * DAY
FEATURES = [
    "mc_forward_ratio_24h", "mc_amt_sim_last_in", "mc_log_min_since_in", "mc_log_age_days_cap3",
]


def compute(s: Stream) -> dict[str, np.ndarray]:
    lo, hi = s.into.bounds(s.q_s, DAY)
    received = s.into.sum(s.cents, lo, hi).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        forward = np.where(received > 0, np.clip(s.cents / received, 0, 10), -1.0)

    start, end = s.into.bounds(s.q_s, None)
    has_in = end > start
    last = np.maximum(end - 1, 0)
    last_log = s.log_usd[s.into.order][last]
    distance = np.abs(s.log_usd - last_log) / LOG_SCALE
    minutes = np.log1p(np.maximum(s.t - s.into.time[last], 0) / 60.0)

    first_seen = np.full(s.n_entities, np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(first_seen, s.s, s.t)
    np.minimum.at(first_seen, s.r, s.t)
    return {
        "mc_forward_ratio_24h": forward.astype(np.float32),
        "mc_amt_sim_last_in": np.where(has_in, np.exp(-distance), 0.0).astype(np.float32),
        "mc_log_min_since_in": np.where(has_in, minutes, -1.0).astype(np.float32),
        "mc_log_age_days_cap3": log1p_f32(np.minimum(s.t - first_seen[s.s], AGE_CAP) / float(DAY)),
    }
