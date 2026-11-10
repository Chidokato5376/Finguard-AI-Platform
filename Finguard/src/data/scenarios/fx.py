"""Tỷ giá về USD, học từ giao dịch đổi tiền của RIÊNG giai đoạn train.

Cột `amount` là số tiền theo loại tiền thanh toán; cộng thẳng Yên với Đô-la thì mọi tổng
trong cửa sổ vô nghĩa. Parquet đã xử lý không còn số tiền nhận nên phải đọc CSV gốc.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

#: Timestamp, Amount Received, Receiving Currency, Amount Paid, Payment Currency.
_RAW_COLUMNS = [0, 5, 6, 7, 8]
_NAMES = ["timestamp", "received", "receive_currency", "paid", "pay_currency"]
ANCHOR = "US Dollar"


def _median_pair_rates(raw_csv: str | Path, train_end: pd.Timestamp) -> dict[tuple[str, str], float]:
    parts = []
    reader = pd.read_csv(raw_csv, skiprows=1, header=None, usecols=_RAW_COLUMNS, names=_NAMES,
                         dtype={"receive_currency": str, "pay_currency": str},
                         chunksize=5_000_000)
    for chunk in reader:
        fx = chunk[chunk["receive_currency"] != chunk["pay_currency"]]
        fx = fx[(fx["paid"] > 0) & (fx["received"] > 0)].copy()
        fx["timestamp"] = pd.to_datetime(fx["timestamp"], format="%Y/%m/%d %H:%M")
        parts.append(fx[fx["timestamp"] <= train_end])
    data = pd.concat(parts, ignore_index=True)
    data["rate"] = data["received"] / data["paid"]
    medians = data.groupby(["pay_currency", "receive_currency"], observed=True)["rate"].median()
    return {pair: float(rate) for pair, rate in medians.items()}


def rates_from_pairs(pairs: dict[tuple[str, str], float], currencies: list[str]) -> dict[str, float]:
    """Một đơn vị mỗi loại tiền đổi được bao nhiêu USD; thiếu đường quy đổi thì báo lỗi."""
    edges: dict[str, list[tuple[str, float]]] = {}
    for (pay, receive), rate in pairs.items():
        edges.setdefault(receive, []).append((pay, rate))
        edges.setdefault(pay, []).append((receive, 1.0 / rate))

    # Duyệt theo chiều rộng từ USD, nên mỗi loại tiền đi đường ngắn nhất.
    rates = {ANCHOR: 1.0}
    pending = deque([ANCHOR])
    while pending:
        current = pending.popleft()
        for other, rate in edges.get(current, []):
            if other not in rates:
                rates[other] = rates[current] * rate
                pending.append(other)
    missing = sorted(set(currencies) - set(rates))
    if missing:
        raise RuntimeError(f"Không dựng được tỷ giá về USD cho {missing} từ giai đoạn train.")
    return rates


def usd_rates(raw_csv: str | Path, train_end: pd.Timestamp, currencies: list[str]) -> dict[str, float]:
    return rates_from_pairs(_median_pair_rates(raw_csv, train_end), currencies)


def to_usd(amount: np.ndarray, currency: pd.Series, rates: dict[str, float]) -> np.ndarray:
    factor = currency.astype(str).map(rates).to_numpy(dtype=np.float64)
    return np.asarray(amount, dtype=np.float64) * factor
