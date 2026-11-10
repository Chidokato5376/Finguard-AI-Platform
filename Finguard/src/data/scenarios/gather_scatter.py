"""GATHER-SCATTER: gom từ nhiều nguồn vào một tài khoản rồi phân tán đi nhiều nơi."""

from __future__ import annotations

import numpy as np

from src.data.scenarios.stream import DAY, Stream, log1p_f32

NAME = "gather_scatter"
VERSION = 1
TARGETS = ("GATHER-SCATTER",)
FEATURES = ["gs_passthrough_24h", "gs_log_min_since_first_in_24h", "gs_log_deg_product_24h"]


def compute(s: Stream) -> dict[str, np.ndarray]:
    in_lo, in_hi = s.into.bounds(s.q_s, DAY)
    out_lo, out_hi = s.out.bounds(s.q_s, DAY)
    amount_in = s.into.sum(s.cents, in_lo, in_hi).astype(np.float64)
    amount_out = s.out.sum(s.cents, out_lo, out_hi).astype(np.float64)
    top = np.maximum(amount_in, amount_out)
    with np.errstate(divide="ignore", invalid="ignore"):
        passthrough = np.where(top > 0, np.minimum(amount_in, amount_out) / top, 0.0)

    # Giai đoạn gom kéo dài bao lâu: tính từ khoản thu đầu tiên trong 24 giờ.
    has_in = in_hi > in_lo
    first_in = s.into.time[np.minimum(in_lo, len(s.into.time) - 1)]
    minutes = np.where(has_in, np.log1p(np.maximum(s.t - first_in, 0) / 60.0), -1.0)

    degree_in = s.in_distinct.count(s.q_s, DAY).astype(np.float64)
    degree_out = s.out_distinct.count(s.q_s, DAY).astype(np.float64)
    return {
        "gs_passthrough_24h": passthrough.astype(np.float32),
        "gs_log_min_since_first_in_24h": minutes.astype(np.float32),
        "gs_log_deg_product_24h": log1p_f32(degree_in * degree_out),
    }
