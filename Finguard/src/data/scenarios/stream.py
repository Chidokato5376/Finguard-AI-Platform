"""Bộ công cụ cửa sổ thời gian cho đặc trưng theo kịch bản.

=== BẤT BIẾN NHÂN QUẢ ===

Mọi cửa sổ có dạng [t − W, t): chỉ tính sự kiện có thời điểm NHỎ HƠN HẲN t. AMLWorld
chỉ ghi tới phút nên nhiều giao dịch trùng thời điểm, và thứ tự của chúng trong file
không mang ý nghĩa. Loại cả các dòng cùng phút khiến kết quả không phụ thuộc thứ tự
dòng, và thêm dòng tương lai không thể đổi giá trị cũ
(tests/test_scenarios.py kiểm chứng cả hai).

=== VÌ SAO CỘNG DỒN BẰNG SỐ NGUYÊN ===

Tổng trong cửa sổ = hiệu hai tổng tích lũy. Với số thực, sai số làm tròn của tổng tích
lũy phụ thuộc mọi phần tử đứng trước, kể cả phần tử của tương lai được chèn vào, nên
giá trị cũ đổi ở chữ số cuối. Tiền được lưu bằng cent, log bằng phần nghìn, trên
int64: tràn số quay vòng không ảnh hưởng vì hiệu hai tổng vẫn đúng.

=== QUY MÔ ===

Không có vòng lặp theo dòng: HI-Medium có 32 triệu dòng. Truy vấn là tìm nhị phân trên
khóa (thực thể, thời điểm), đếm khác biệt trong cửa sổ quy về đếm khoảng phủ.
"""

from __future__ import annotations

from functools import cached_property

import numpy as np
import pandas as pd

HOUR = 3600
DAY = 24 * HOUR
WEEK = 7 * DAY

#: Cửa sổ dài nhất module nào cũng phải nhỏ hơn. Dịch trục thời gian đi chừng này để
#: t − W luôn không âm, nhờ đó khóa (thực thể, t − W) không lấn sang thực thể trước.
MAX_WINDOW = 8 * DAY

MONEY_SCALE = 100
LOG_SCALE = 1000
#: Chặn số tiền cực đại để tổng trong một cửa sổ luôn vừa int64.
MONEY_CAP = 1e13
NEVER = -(10 ** 12)


def log1p_f32(values) -> np.ndarray:
    return np.log1p(np.asarray(values, dtype=np.float64).clip(min=0)).astype(np.float32)


class Query:
    """Một loạt truy vấn (thực thể, thời điểm); giữ sẵn thứ tự sắp để tìm nhị phân nhanh."""

    def __init__(self, entity: np.ndarray, time: np.ndarray, base: int):
        self.key = np.asarray(entity, dtype=np.int64) * base + time
        # Trừ cùng một W không đổi thứ tự của khóa, vì mọi t đều lớn hơn W.
        self.order = np.argsort(self.key, kind="stable")

    def find(self, keys: np.ndarray, shift: int = 0) -> np.ndarray:
        """Số phần tử của `keys` nhỏ hơn hẳn (khóa truy vấn − shift)."""
        out = np.empty(len(self.key), dtype=np.int64)
        out[self.order] = np.searchsorted(keys, self.key[self.order] - shift, side="left")
        return out


class EventIndex:
    """Sự kiện của từng thực thể, sắp theo thời gian, truy vấn được theo cửa sổ."""

    def __init__(self, entity: np.ndarray, time: np.ndarray, base: int, tiebreak: np.ndarray):
        self.base = base
        keys = np.asarray(entity, dtype=np.int64) * base + time
        # Khóa phụ để hàng "sự kiện gần nhất" trong cùng một phút không phụ thuộc thứ tự dòng.
        self.order = np.lexsort((tiebreak, keys))
        self.keys = keys[self.order]
        self.entity = np.asarray(entity, dtype=np.int64)[self.order]
        self.time = np.asarray(time, dtype=np.int64)[self.order]

    def bounds(self, query: Query, window: int | None) -> tuple[np.ndarray, np.ndarray]:
        """(lo, hi) sao cho [lo, hi) là các sự kiện trong cửa sổ; window=None là toàn bộ quá khứ."""
        hi = query.find(self.keys)
        if window is None:
            lo = self.entity_start(query)
        else:
            lo = query.find(self.keys, shift=window)
        return lo, hi

    def entity_start(self, query: Query) -> np.ndarray:
        """Vị trí sự kiện đầu tiên của thực thể được hỏi."""
        floor = (query.key // self.base) * self.base
        out = np.empty(len(floor), dtype=np.int64)
        out[query.order] = np.searchsorted(self.keys, floor[query.order], side="left")
        return out

    def prefix(self, row_values: np.ndarray) -> np.ndarray:
        """Tổng tích lũy theo thứ tự chỉ mục của một cột số nguyên xếp theo dòng gốc."""
        return self.sorted_prefix(np.asarray(row_values, dtype=np.int64)[self.order])

    @staticmethod
    def sorted_prefix(sorted_values: np.ndarray) -> np.ndarray:
        return np.concatenate(([0], np.cumsum(sorted_values, dtype=np.int64)))

    def sum(self, row_values: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        cumulative = self.prefix(row_values)
        return cumulative[hi] - cumulative[lo]


class DistinctIndex:
    """Đếm số đối tác KHÁC NHAU của một thực thể trong cửa sổ.

    Sự kiện j được tính cho truy vấn t khi nó là lần xuất hiện đầu tiên của đối tác đó
    trong cửa sổ: t_j ≥ t − W, t_j < t, và lần trước của cùng đối tác (prev) < t − W.
    Đó là t thuộc nửa khoảng (max(t_j, prev + W), t_j + W], nên đáp án là số khoảng
    phủ t, tính bằng hai lần tìm nhị phân.
    """

    def __init__(self, entity: np.ndarray, partner: np.ndarray, time: np.ndarray, base: int):
        self.base = base
        width = int(max(np.max(partner, initial=0), np.max(entity, initial=0))) + 1
        pair = np.asarray(entity, dtype=np.int64) * width + partner
        order = np.lexsort((time, pair))
        pair, sorted_time = pair[order], np.asarray(time, dtype=np.int64)[order]
        fresh_pair = np.r_[True, pair[1:] != pair[:-1]]
        keep = fresh_pair | np.r_[True, sorted_time[1:] != sorted_time[:-1]]
        self.entity = (pair[keep] // width).astype(np.int64)
        self.time = sorted_time[keep]
        self.prev = np.r_[NEVER, self.time[:-1]]
        self.prev[fresh_pair[keep]] = NEVER

    def count(self, query: Query, window: int) -> np.ndarray:
        start = np.maximum(self.time, self.prev + window)
        stop = self.time + window
        starts = np.sort(self.entity * self.base + start)
        stops = np.sort(self.entity * self.base + stop)
        return query.find(starts) - query.find(stops)


class Stream:
    """Luồng giao dịch đã mã hóa. Chỉ chứa dữ liệu quan sát được, KHÔNG có nhãn."""

    def __init__(
        self,
        sender: np.ndarray,
        receiver: np.ndarray,
        seconds: np.ndarray,
        amount_usd: np.ndarray,
        channel: np.ndarray,
        channel_names: list[str],
        entity_names: np.ndarray,
    ):
        self.n = len(sender)
        self.s = np.asarray(sender, dtype=np.int64)
        self.r = np.asarray(receiver, dtype=np.int64)
        seconds = np.asarray(seconds, dtype=np.int64)
        self.t = seconds - (seconds.min() if self.n else 0) + MAX_WINDOW
        self.base = int(self.t.max(initial=MAX_WINDOW)) + 2 * MAX_WINDOW + 1
        self.usd = np.asarray(amount_usd, dtype=np.float64).clip(0, MONEY_CAP)
        self.channel = np.asarray(channel, dtype=np.int64)
        self.channel_names = list(channel_names)
        self.entity_names = np.asarray(entity_names, dtype=object)
        self.n_entities = len(self.entity_names)
        self.cents = np.rint(self.usd * MONEY_SCALE).astype(np.int64)
        self.log_usd = np.rint(np.log1p(self.usd) * LOG_SCALE).astype(np.int64)

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> Stream:
        """Từ khung có account_id, counterparty_id, timestamp, amount_usd, channel."""
        ids = pd.concat([frame["account_id"].astype(str), frame["counterparty_id"].astype(str)],
                        ignore_index=True).astype("category")
        n = len(frame)
        codes = ids.cat.codes.to_numpy()
        channel = frame["channel"].astype(str).astype("category")
        return cls(
            codes[:n], codes[n:],
            pd.to_datetime(frame["timestamp"]).dt.as_unit("s").astype("int64").to_numpy(),
            frame["amount_usd"].to_numpy(dtype=np.float64),
            channel.cat.codes.to_numpy(), list(channel.cat.categories),
            ids.cat.categories.to_numpy(dtype=object),
        )

    def event_index(self, entity: np.ndarray) -> EventIndex:
        return EventIndex(entity, self.t, self.base, self.cents)

    def query(self, entity: np.ndarray) -> Query:
        return Query(entity, self.t, self.base)

    @cached_property
    def out(self) -> EventIndex:
        """Các khoản CHI, theo người gửi."""
        return self.event_index(self.s)

    @cached_property
    def into(self) -> EventIndex:
        """Các khoản THU, theo người nhận."""
        return self.event_index(self.r)

    @cached_property
    def q_s(self) -> Query:
        return self.query(self.s)

    @cached_property
    def q_r(self) -> Query:
        return self.query(self.r)

    @cached_property
    def out_distinct(self) -> DistinctIndex:
        return DistinctIndex(self.s, self.r, self.t, self.base)

    @cached_property
    def in_distinct(self) -> DistinctIndex:
        return DistinctIndex(self.r, self.s, self.t, self.base)

    @cached_property
    def first_pair(self) -> np.ndarray:
        """1 nếu cặp (gửi, nhận) chưa từng xuất hiện ở thời điểm NHỎ HƠN dòng này."""
        pair = self.s * self.n_entities + self.r
        order = np.lexsort((self.t, pair))
        run = np.cumsum(np.r_[True, pair[order][1:] != pair[order][:-1]]) - 1
        earliest = self.t[order][np.r_[True, run[1:] != run[:-1]]]
        flag = np.zeros(self.n, dtype=np.int64)
        flag[order] = (self.t[order] == earliest[run]).astype(np.int64)
        return flag

    @cached_property
    def bank(self) -> np.ndarray:
        """Mã ngân hàng của từng thực thể, tách từ định danh 'B<ngân hàng>_<tài khoản>'."""
        names = pd.Series(self.entity_names, dtype=str)
        return pd.factorize(names.str.split("_", n=1).str[0])[0].astype(np.int64)

    def window_stats(
        self, index: EventIndex, lo: np.ndarray, hi: np.ndarray, row_values: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(số sự kiện, trung bình, độ lệch chuẩn) trên thang log tiền của cửa sổ."""
        row_values = self.log_usd if row_values is None else row_values
        ordered = np.asarray(row_values, dtype=np.int64)[index.order]
        return window_stats_sorted(ordered, lo, hi)


def window_stats_sorted(
    ordered: np.ndarray, lo: np.ndarray, hi: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Như Stream.window_stats, với giá trị (fixed-point) đã theo thứ tự chỉ mục."""
    first = EventIndex.sorted_prefix(ordered)
    second = EventIndex.sorted_prefix(ordered * ordered)
    count = (hi - lo).astype(np.float64)
    total = (first[hi] - first[lo]).astype(np.float64) / LOG_SCALE
    squares = (second[hi] - second[lo]).astype(np.float64) / LOG_SCALE ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = np.where(count > 0, total / count, 0.0)
        variance = np.where(count > 1, (squares - total * total / count) / (count - 1), 0.0)
    return count, mean, np.sqrt(np.maximum(variance, 0.0))


def last_before(index: EventIndex, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Vị trí sự kiện gần nhất trong [lo, hi), hoặc −1 nếu cửa sổ rỗng."""
    return np.where(hi > lo, hi - 1, -1)
