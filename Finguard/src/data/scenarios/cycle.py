"""CYCLE: tiền quay về nơi xuất phát. Thủ công chỉ đo vòng hai bước; vòng dài giao cho DRRE."""

from __future__ import annotations

import numpy as np

from src.data.scenarios.stream import WEEK, EventIndex, Query, Stream, log1p_f32

NAME = "cycle"
VERSION = 1
TARGETS = ("CYCLE",)
FEATURES = ["cy_reciprocity_7d", "cy_log_n_reverse_7d", "cy_log_min_since_reverse"]


def compute(s: Stream) -> dict[str, np.ndarray]:
    width = s.n_entities
    pair = s.s * width + s.r
    unique_pairs = np.unique(pair)
    index = EventIndex(np.searchsorted(unique_pairs, pair), s.t, s.base, s.cents)

    reverse = s.r * width + s.s
    position = np.searchsorted(unique_pairs, reverse)
    position_safe = np.minimum(position, len(unique_pairs) - 1)
    # Cặp ngược chưa từng tồn tại nhận mã nằm ngoài chỉ mục, nên đếm ra 0.
    reverse_id = np.where(unique_pairs[position_safe] == reverse, position_safe, len(unique_pairs))

    query = Query(reverse_id, s.t, s.base)
    lo, hi = index.bounds(query, WEEK)
    count = hi - lo
    last = index.time[np.maximum(hi - 1, 0)]
    minutes = np.where(count > 0, np.log1p(np.maximum(s.t - last, 0) / 60.0), -1.0)
    return {
        "cy_reciprocity_7d": (count > 0).astype(np.float32),
        "cy_log_n_reverse_7d": log1p_f32(count),
        "cy_log_min_since_reverse": minutes.astype(np.float32),
    }
