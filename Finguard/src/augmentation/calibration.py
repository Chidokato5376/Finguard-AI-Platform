"""Calibration — ước lượng biên tham số của từng kiểu hình (Mục 6.3).

Nguyên tắc "neo thực nghiệm" (Mục 6.2): mọi tham số số học mà LLM được
phép đề xuất đều phải nằm trong biên do file này sinh ra. Biên có hai
nguồn, và nguồn của TỪNG tham số được ghi lại để manifest truy vết được:

  - "du_lieu_thuc": suy ra từ phân vị của dữ liệu thực CÓ NHÃN.
  - "tri_thuc_mien": hằng số có dẫn nguồn, dùng khi dữ liệu thực không
    chứa thông tin đó. Ví dụ đã kiểm chứng: PaySim KHÔNG có chuỗi money
    mule (99,85% người gửi chỉ có 1 giao dịch), nên chain_length và
    amount_decay KHÔNG thể suy ra từ PaySim — phải lấy từ tri thức miền.

ĐƠN VỊ SỐ TIỀN: biên amount được hiệu chỉnh theo ĐƠN VỊ CỦA DỮ LIỆU NỀN
đang dùng, không mặc định là VND. Khóa đặc tả tên `amount_vnd_*` chỉ đúng
nghĩa VND khi dữ liệu nền là synthetic_vn. Trường Calibration.unit ghi rõ
điều này và được đưa vào manifest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.augmentation.spec_schema import HARD_LIMITS

#: Tri thức miền — chuỗi trung gian money mule thường 3–6 chặng, mỗi chặng
#: giữ lại 92–99% số tiền (phần hụt là "phí" của tài khoản trung gian).
#: Nguồn: mô tả điển hình hóa trong các vụ án rửa tiền qua tài khoản ngân
#: hàng tại Việt Nam (vụ Mr Pips — VTV, VnExpress, Tuổi Trẻ) và tài liệu
#: typology của FATF về layering. ĐÂY LÀ THAM SỐ GIẢ ĐỊNH, không phải số
#: đo thực nghiệm — phải nêu rõ khi báo cáo.
DOMAIN_CHAIN_LENGTH: tuple[float, float] = (3.0, 6.0)
DOMAIN_AMOUNT_DECAY: tuple[float, float] = (0.92, 0.99)

#: Khoảng cách thời gian giữa hai chặng của chuỗi mule: vài phút tới vài
#: giờ. Khi dữ liệu thực có chuỗi (vd. synthetic_vn, AMLSim) thì hàm
#: calibrate_from_real() sẽ ghi đè bằng phân vị thực đo.
DOMAIN_HOP_GAP_MINUTES: tuple[float, float] = (2.0, 180.0)


@dataclass(frozen=True)
class Calibration:
    """Biên tham số đã hiệu chỉnh + nguồn của từng biên."""

    bounds: dict[str, tuple[float, float]]
    sources: dict[str, str]
    unit: str
    n_real_rows: int
    n_real_fraud: int
    notes: list[str] = field(default_factory=list)

    def bounds_for(self, archetype: str) -> dict[str, tuple[float, float]]:
        """Biên áp dụng cho một kiểu hình. Hiện dùng chung một bộ biên cho
        mọi kiểu hình; tách theo kiểu hình khi có dữ liệu thực đủ nhiều cho
        từng kiểu (xem notes trong manifest).
        """
        return dict(self.bounds)

    def to_json(self) -> dict[str, Any]:
        return {
            "bounds": {k: list(v) for k, v in self.bounds.items()},
            "sources": dict(self.sources),
            "unit": self.unit,
            "n_real_rows": self.n_real_rows,
            "n_real_fraud": self.n_real_fraud,
            "notes": list(self.notes),
        }


def _time_seconds(df: pd.DataFrame) -> np.ndarray:
    """Trục thời gian theo giây, chấp nhận cả `timestamp` và `step`."""
    if "timestamp" in df.columns:
        return pd.to_datetime(df["timestamp"]).astype("int64").to_numpy() / 1e9
    if "step" in df.columns:
        step = df["step"].astype(float).to_numpy()
        # PaySim: step = 1 giờ. IEEE-CIS: TransactionDT = giây.
        unit_seconds = 3600.0 if step.max() < 1e5 else 1.0
        return step * unit_seconds
    raise KeyError("DataFrame cần có cột 'timestamp' hoặc 'step' để hiệu chỉnh nhịp thời gian")


def _clip(bounds: tuple[float, float], param: str) -> tuple[float, float]:
    lo, hi = HARD_LIMITS[param]
    return (float(np.clip(bounds[0], lo, hi)), float(np.clip(bounds[1], lo, hi)))


def calibrate_from_real(
    df: pd.DataFrame,
    unit: str,
    target_cases_per_spec: tuple[int, int] = (3, 40),
    amount_percentiles: tuple[float, float] = (5.0, 95.0),
) -> Calibration:
    """Suy biên tham số từ dữ liệu thực CÓ NHÃN.

    Args:
        df: dữ liệu thực đã chuẩn hóa (cần `amount`, `label`, và cột thời gian).
        unit: đơn vị của cột amount, vd. "VND", "USD (IEEE-CIS)",
            "đơn vị PaySim". Ghi thẳng vào manifest.
        target_cases_per_spec: biên cho n_cases — đây là tham số VẬN HÀNH
            (muốn sinh bao nhiêu ca mỗi đặc tả), không suy từ dữ liệu.
        amount_percentiles: phân vị dùng làm biên số tiền của lớp dương.

    Raises:
        ValueError: nếu không có dòng nào mang nhãn dương — không thể neo
            biên số tiền vào dữ liệu thực, phải truyền tay.
    """
    if "label" not in df.columns:
        raise ValueError("Thiếu cột 'label' — tầng tăng cường phải neo vào dữ liệu CÓ NHÃN")

    label = df["label"].astype(bool).to_numpy()
    fraud = df.loc[label, "amount"].astype(float).to_numpy()
    if fraud.size == 0:
        raise ValueError(
            "Dữ liệu thực không có nhãn dương nào — không thể hiệu chỉnh biên số tiền. "
            "Hãy truyền một lát dữ liệu có gian lận, hoặc đặt biên bằng tay."
        )

    bounds: dict[str, tuple[float, float]] = {}
    sources: dict[str, str] = {}
    notes: list[str] = []

    lo, hi = np.percentile(fraud, amount_percentiles)
    if hi <= lo:  # mọi giao dịch gian lận cùng một mức
        hi = lo * 1.5 + 1.0
    bounds["amount_vnd"] = _clip((float(lo), float(hi)), "amount_vnd")
    sources["amount_vnd"] = f"du_lieu_thuc (phân vị {amount_percentiles[0]:g}–{amount_percentiles[1]:g} của lớp dương, n={fraud.size})"

    # Nhịp thời gian: khoảng cách giữa hai giao dịch liên tiếp của CÙNG tài khoản
    t = _time_seconds(df)
    order = np.argsort(t, kind="stable")
    acc = df["account_id"].to_numpy()[order]
    t_sorted = t[order]
    same = acc[1:] == acc[:-1]
    gaps_min = (t_sorted[1:] - t_sorted[:-1])[same] / 60.0
    gaps_min = gaps_min[gaps_min > 0]
    if gaps_min.size >= 30:
        g_lo, g_hi = np.percentile(gaps_min, [10.0, 75.0])
        bounds["hop_gap_minutes"] = _clip((float(g_lo), float(g_hi)), "hop_gap_minutes")
        sources["hop_gap_minutes"] = f"du_lieu_thuc (phân vị 10–75 khoảng cách cùng tài khoản, n={gaps_min.size})"
    else:
        bounds["hop_gap_minutes"] = _clip(DOMAIN_HOP_GAP_MINUTES, "hop_gap_minutes")
        sources["hop_gap_minutes"] = "tri_thuc_mien (dữ liệu thực có < 30 cặp giao dịch liên tiếp cùng tài khoản)"
        notes.append(
            "Dữ liệu nền gần như không có tài khoản nào xuất hiện nhiều lần, nên nhịp "
            "thời gian giữa các chặng phải lấy từ tri thức miền thay vì đo được."
        )

    # Số giao dịch mỗi ca: suy từ số giao dịch của các tài khoản CÓ gian lận
    fraud_accounts = set(df.loc[label, "account_id"])
    if fraud_accounts:
        per_acc = (
            df[df["account_id"].isin(fraud_accounts)]
            .groupby("account_id", observed=True).size()
        )
        b_lo, b_hi = np.percentile(per_acc.to_numpy(), [50.0, 90.0])
        bounds["burst_count"] = _clip((max(1.0, float(b_lo)), max(2.0, float(b_hi))), "burst_count")
        sources["burst_count"] = f"du_lieu_thuc (phân vị 50–90 số giao dịch của tài khoản có gian lận, n={len(per_acc)})"
    else:  # pragma: no cover — đã chặn ở trên bằng ValueError
        bounds["burst_count"] = _clip((1.0, 5.0), "burst_count")
        sources["burst_count"] = "tri_thuc_mien"

    # Hai tham số KHÔNG suy được từ dữ liệu phẳng không có chuỗi
    bounds["chain_length"] = _clip(DOMAIN_CHAIN_LENGTH, "chain_length")
    sources["chain_length"] = "tri_thuc_mien (typology layering; PaySim/IEEE-CIS không có chuỗi mule để đo)"
    bounds["amount_decay"] = _clip(DOMAIN_AMOUNT_DECAY, "amount_decay")
    sources["amount_decay"] = "tri_thuc_mien (phí tài khoản trung gian 1–8% mỗi chặng)"

    # Tham số vận hành
    bounds["n_cases"] = _clip((float(target_cases_per_spec[0]), float(target_cases_per_spec[1])), "n_cases")
    sources["n_cases"] = "tham_so_van_hanh (khối lượng muốn sinh, không suy từ dữ liệu)"

    return Calibration(
        bounds=bounds,
        sources=sources,
        unit=unit,
        n_real_rows=int(len(df)),
        n_real_fraud=int(label.sum()),
        notes=notes,
    )
