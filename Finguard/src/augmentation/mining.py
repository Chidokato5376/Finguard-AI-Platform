"""Adversarial mining — thu biến thể mà mô hình hiện tại bỏ sót (Mục 6.3).

=== MẶC ĐỊNH TẮT — ĐỌC TRƯỚC KHI BẬT ===

Vòng khai thác mẫu đối kháng làm dữ liệu huấn luyện PHỤ THUỘC VÀO MÔ HÌNH
(Mục 6.6). Khi bật, tập dữ liệu của vòng sau được chọn bởi điểm yếu của
mô hình vòng trước, nên:

  - Không được bật trong các cấu hình đối chứng (i)/(ii)/(iii) ban đầu.
    Nếu bật, phần cải thiện không còn quy được cho LLM hay cho việc tăng
    cường dữ liệu nói chung.
  - Chỉ bật khi nghiên cứu RIÊNG hiệu quả của đối kháng, và phải ghi rõ
    trong manifest (notes) rằng vòng này có mining.

=== BẤT BIẾN: MINING CHỈ ĐƯỢC THU HẸP BIÊN ===

apply_focus() chỉ được phép làm biên HẸP HƠN biên hiệu chỉnh gốc, không
bao giờ rộng hơn. Nhờ vậy nguyên tắc "neo thực nghiệm" (Mục 6.2) vẫn giữ:
mô hình có thể hướng bộ sinh chú ý vào vùng tham số nào, nhưng không thể
đẩy bộ sinh ra ngoài vùng đã hiệu chỉnh từ dữ liệu thực.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.augmentation.calibration import Calibration
from src.augmentation.spec_schema import ScenarioSpec


@dataclass(frozen=True)
class MiningConfig:
    """Cấu hình khai thác đối kháng. enabled=False là mặc định có chủ ý."""

    enabled: bool = False
    #: Giao dịch được coi là "đã cảnh báo" nếu điểm nằm trong phân vị trên này
    alert_quantile: float = 0.99
    #: Số đặc tả bị bỏ sót nhiều nhất được đưa trở lại bộ sinh
    top_k: int = 5
    #: Hệ số thu hẹp biên quanh vùng bị bỏ sót (0 < factor <= 1)
    narrow_factor: float = 0.6


@dataclass
class MiningReport:
    """Kết quả một vòng khai thác — đưa vào manifest.notes của vòng sau."""

    enabled: bool
    alert_threshold: float
    miss_rate_by_spec: dict[str, float] = field(default_factory=dict)
    focus_specs: list[str] = field(default_factory=list)
    focus_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "alert_threshold": self.alert_threshold,
            "miss_rate_by_spec": self.miss_rate_by_spec,
            "focus_specs": self.focus_specs,
            "focus_bounds": {k: list(v) for k, v in self.focus_bounds.items()},
            "notes": list(self.notes),
        }


def mine_missed_specs(
    scored: pd.DataFrame,
    specs: Sequence[ScenarioSpec],
    config: MiningConfig = MiningConfig(),
    score_col: str = "risk_score",
) -> MiningReport:
    """Tìm các đặc tả mà mô hình hiện tại bỏ sót nhiều nhất.

    Args:
        scored: tập đã chấm điểm, cần có `risk_score`, `label`, `spec_id`.
        specs: các đặc tả đã dùng để sinh dữ liệu.
        config: xem MiningConfig — enabled=False thì trả báo cáo rỗng.
        score_col: tên cột điểm rủi ro.

    Returns:
        MiningReport. Khi config.enabled=False, trả báo cáo với
        enabled=False và không tính toán gì — để gọi hàm này vô điều kiện
        trong pipeline mà vẫn an toàn cho cấu hình đối chứng.
    """
    if not config.enabled:
        return MiningReport(
            enabled=False,
            alert_threshold=float("nan"),
            notes=["Mining đang TẮT (mặc định) — không có ảnh hưởng tới dữ liệu huấn luyện."],
        )

    missing = {score_col, "label", "spec_id"} - set(scored.columns)
    if missing:
        raise KeyError(f"Thiếu cột để khai thác đối kháng: {sorted(missing)}")

    scores = scored[score_col].astype(float).to_numpy()
    threshold = float(np.quantile(scores, config.alert_quantile))
    fraud = scored["label"].fillna(False).astype(bool)
    synthetic_fraud = scored[fraud & scored["spec_id"].notna()]

    miss_rate: dict[str, float] = {}
    for spec_id, group in synthetic_fraud.groupby("spec_id"):
        missed = (group[score_col].astype(float) < threshold).mean()
        miss_rate[str(spec_id)] = float(missed)

    focus = sorted(miss_rate, key=lambda k: miss_rate[k], reverse=True)[: config.top_k]
    by_id = {s.spec_id: s for s in specs}
    focus_specs = [f for f in focus if f in by_id and miss_rate[f] > 0.0]

    focus_bounds: dict[str, tuple[float, float]] = {}
    if focus_specs:
        amounts = [by_id[f].amount_vnd for f in focus_specs]
        gaps = [by_id[f].hop_gap_minutes for f in focus_specs]
        focus_bounds["amount_vnd"] = (
            float(min(a[0] for a in amounts)), float(max(a[1] for a in amounts)))
        focus_bounds["hop_gap_minutes"] = (
            float(min(g[0] for g in gaps)), float(max(g[1] for g in gaps)))

    return MiningReport(
        enabled=True,
        alert_threshold=threshold,
        miss_rate_by_spec=miss_rate,
        focus_specs=focus_specs,
        focus_bounds=focus_bounds,
        notes=[
            f"Vòng này CÓ khai thác đối kháng: {len(focus_specs)} đặc tả bị bỏ sót nhiều nhất "
            "được đưa trở lại bộ sinh. Dữ liệu huấn luyện vòng sau phụ thuộc vào mô hình "
            "vòng này — KHÔNG dùng kết quả này cho cấu hình đối chứng (Mục 6.6)."
        ],
    )


def apply_focus(
    calibration: Calibration, report: MiningReport, config: MiningConfig = MiningConfig()
) -> Calibration:
    """Thu hẹp biên hiệu chỉnh về vùng bị bỏ sót — KHÔNG BAO GIỜ mở rộng.

    Biên mới = giao của (biên gốc) và (vùng bị bỏ sót đã nới theo
    narrow_factor). Nếu phép giao rỗng thì giữ nguyên biên gốc cho tham số
    đó và ghi chú lại.
    """
    if not report.enabled or not report.focus_bounds:
        return calibration

    new_bounds = dict(calibration.bounds)
    new_sources = dict(calibration.sources)
    notes = list(calibration.notes)

    for param, (lo, hi) in report.focus_bounds.items():
        orig = calibration.bounds.get(param)
        if orig is None:
            continue
        center = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo) / max(config.narrow_factor, 1e-6)
        cand = (center - half, center + half)
        inter = (max(orig[0], cand[0]), min(orig[1], cand[1]))
        if inter[0] >= inter[1]:
            notes.append(
                f"Mining đề xuất vùng cho '{param}' không giao với biên gốc — giữ biên gốc.")
            continue
        new_bounds[param] = (float(inter[0]), float(inter[1]))
        new_sources[param] = (
            f"{calibration.sources.get(param, '?')} + thu hẹp bởi adversarial mining")

    notes.extend(report.notes)
    return Calibration(
        bounds=new_bounds,
        sources=new_sources,
        unit=calibration.unit,
        n_real_rows=calibration.n_real_rows,
        n_real_fraud=calibration.n_real_fraud,
        notes=notes,
    )
