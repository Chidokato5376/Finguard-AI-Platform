"""Unified Schema — chuẩn hóa dữ liệu từ PaySim / IEEE-CIS / Synthetic VN.

Xem docs/architecture.md và data/README.md để biết định nghĩa đầy đủ
và các cảnh báo về nguồn gốc trường dữ liệu (đặc biệt device_id/ip_country
đối với PaySim).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional


class Channel(str, Enum):
    DOMESTIC = "domestic"
    INTERNATIONAL = "international"
    E_WALLET = "e_wallet"
    UNKNOWN = "unknown"


class DataSource(str, Enum):
    """Nguồn gốc bản ghi — bắt buộc để phân biệt dữ liệu thật vs tổng hợp."""

    PAYSIM = "paysim"
    IEEE_CIS = "ieee_cis"
    SYNTHETIC_VN = "synthetic_vn"


@dataclass(frozen=True, slots=True)
class TransactionRecord:
    """Một bản ghi giao dịch đã chuẩn hóa theo Unified Schema (Mục 3.3).

    Attributes:
        account_id: Định danh tài khoản gửi — khóa tích lũy Behavior Memory.
        counterparty_id: Định danh tài khoản nhận — đỉnh thứ hai trong đồ thị.
        timestamp: Thời điểm giao dịch (trục thời gian cho GRU và decay hành vi).
        amount: Số tiền giao dịch.
        channel: Kênh giao dịch.
        device_id: Tín hiệu thiết bị. CẢNH BÁO: không có sẵn trong PaySim gốc —
            nếu source == PAYSIM, trường này PHẢI được đánh dấu is_synthetic=True.
        ip_country: Tín hiệu định vị. Cùng cảnh báo như device_id.
        label: Nhãn gian lận (nếu có) — chỉ dùng cho fine-tune bán giám sát (L_BCE).
        source: Nguồn dữ liệu gốc — dùng để audit tính xác thực của từng trường.
        is_synthetic_field: Danh sách tên trường bị sinh bổ sung (không phải dữ liệu gốc).
    """

    account_id: str
    counterparty_id: str
    timestamp: datetime
    amount: float
    channel: Channel
    device_id: Optional[str]
    ip_country: Optional[str]
    label: Optional[bool]
    source: DataSource
    is_synthetic_field: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.amount < 0:
            raise ValueError(f"amount must be non-negative, got {self.amount}")
        if self.source is DataSource.PAYSIM:
            missing = {"device_id", "ip_country"} - set(self.is_synthetic_field)
            if (self.device_id is not None or self.ip_country is not None) and missing:
                raise ValueError(
                    "PaySim không có device_id/ip_country gốc — nếu điền giá trị, "
                    "phải khai báo trong is_synthetic_field. Xem data/README.md."
                )
