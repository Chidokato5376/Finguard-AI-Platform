"""Đặc trưng theo từng kịch bản rửa tiền: mỗi file một kịch bản, bật/tắt độc lập.

Mọi mô-đun có cùng giao diện: NAME, VERSION, TARGETS (kiểu hình nhắm tới), FEATURES và
compute(Stream) -> {tên: mảng float32}. Stream không có nhãn, nên mô-đun không thể đọc
nhãn. Tăng VERSION khi đổi công thức để scripts/build_scenarios.py tính lại mô-đun đó.
"""

from __future__ import annotations

from types import ModuleType

from src.data.scenarios import (
    cash_structuring, cross_bank_fx, cycle, fan_in, fan_out, gather_scatter, hub, integration,
    mule_chain, scatter_gather, synchronized, temporal_windows,
)

MODULES: tuple[ModuleType, ...] = (
    fan_out, fan_in, gather_scatter, scatter_gather, cycle, mule_chain, synchronized, hub,
    cross_bank_fx, cash_structuring, integration, temporal_windows,
)
REGISTRY: dict[str, ModuleType] = {module.NAME: module for module in MODULES}

#: Hai cột đo được là vô dụng trên AMLWorld (lộ trình đợt 2, bảng Giai đoạn 1).
DROPPED_BASE_FEATURES = ("is_round_amount", "hour_sin")


def _require(name: str) -> None:
    if name not in REGISTRY:
        raise ValueError(f"Không có mô-đun kịch bản '{name}'. Có: {sorted(REGISTRY)}")


def resolve(spec: str) -> list[str]:
    """Cờ bật/tắt: 'all', 'none', 'fan_in,cycle', hoặc 'all,-hub' để loại bớt."""
    chosen: list[str] = []
    for part in (piece.strip() for piece in spec.split(",") if piece.strip()):
        if part == "all":
            chosen = list(REGISTRY)
        elif part == "none":
            chosen = []
        elif part.startswith("-"):
            _require(part[1:])
            chosen = [name for name in chosen if name != part[1:]]
        else:
            _require(part)
            if part not in chosen:
                chosen.append(part)
    return chosen


def features_of(names: list[str]) -> list[str]:
    return [feature for name in names for feature in REGISTRY[name].FEATURES]
