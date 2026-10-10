"""Đặc tả theo LUẬT — đường cơ sở bắt buộc của cấu hình (ii) (Mục 6.5).

Vì sao file này tồn tại: nếu chỉ so "chỉ dữ liệu thực" với "có thêm dữ
liệu do LLM đề xuất", phần cải thiện có thể đến từ BẢN THÂN việc tăng
cường dữ liệu, không phải từ LLM. Cấu hình (ii) tách hai nguyên nhân đó
ra. Chỉ khi (iii) vượt (ii) có ý nghĩa thống kê thì mới được quy công cho
LLM (xem protocol.CLAIM_GATE).

Phân biệt với llm_adapter.OfflineVariantTransport: transport đó là bản
giả của đường gọi mạng, dùng để kiểm thử đường ống; còn file này là ĐƯỜNG
CƠ SỞ KHOA HỌC — liệt kê có hệ thống các biến thể trong biên hiệu chỉnh,
không ngẫu nhiên tùy ý, nên tái lập được mà không cần seed.
"""

from __future__ import annotations

from src.augmentation.calibration import Calibration
from src.augmentation.spec_schema import (
    ARCHETYPES,
    CHAIN_ARCHETYPES,
    ScenarioSpec,
    SpecRejection,
    validate_spec,
)

#: Ba mức số tiền quét qua biên hiệu chỉnh: thấp / giữa / cao.
AMOUNT_TERCILES: tuple[tuple[float, float], ...] = ((0.0, 0.34), (0.33, 0.67), (0.66, 1.0))

#: Khung giờ gắn với từng kiểu hình theo tri thức miền. Chiếm quyền tài
#: khoản thường xảy ra lúc chủ tài khoản không để ý (rạng sáng, đêm);
#: chuỗi mule chạy liên tục nên quét cả ngày.
ARCHETYPE_DAY_PARTS: dict[str, tuple[str, ...]] = {
    "money_mule_chain": ("sang", "chieu", "dem"),
    "account_takeover": ("rang_sang", "dem"),
    "investment_romance_scam": ("toi", "chieu"),
    "smurfing_structuring": ("sang", "trua", "chieu"),
}

ARCHETYPE_FLAGS: dict[str, tuple[bool, bool]] = {
    # (new_device, foreign_ip)
    "money_mule_chain": (True, False),
    "account_takeover": (True, True),
    "investment_romance_scam": (False, False),
    "smurfing_structuring": (False, False),
}

ARCHETYPE_CHANNEL: dict[str, str] = {
    "money_mule_chain": "domestic",
    "account_takeover": "domestic",
    "investment_romance_scam": "e_wallet",
    "smurfing_structuring": "domestic",
}


def _lerp(bounds: tuple[float, float], frac: float) -> float:
    return bounds[0] + (bounds[1] - bounds[0]) * frac


def _lerp_round(bounds: tuple[float, float], frac: float, digits: int = 2) -> float:
    """Nội suy rồi làm tròn, nhưng KẸP LẠI vào trong biên.

    Làm tròn có thể đẩy giá trị ra ngoài biên vài phần nghìn (biên số tiền
    có độ lớn ~1e8), và khi đó validator loại đặc tả của chính đường cơ sở.
    Kẹp lại để đường cơ sở (ii) không bao giờ phụ thuộc vào sai số cho phép
    của validator.
    """
    return float(min(max(round(_lerp(bounds, frac), digits), bounds[0]), bounds[1]))


def build_rule_specs(
    calibration: Calibration,
    archetypes: tuple[str, ...] = ARCHETYPES,
    n_cases_per_spec: int | None = None,
) -> tuple[list[ScenarioSpec], list[SpecRejection]]:
    """Liệt kê đặc tả theo luật: mỗi kiểu hình × mỗi mức số tiền × mỗi khung giờ.

    Args:
        calibration: biên tham số đã hiệu chỉnh.
        archetypes: các kiểu hình cần phủ.
        n_cases_per_spec: số ca mỗi đặc tả. None → lấy giữa biên n_cases.

    Returns:
        (specs hợp lệ, danh sách bị loại). Danh sách bị loại phải RỖNG với
        calibration hợp lý — nếu không rỗng tức là biên hiệu chỉnh xung đột
        với biên tuyệt đối, và manifest sẽ ghi lại để điều tra.
    """
    bounds = calibration.bounds
    n_cases = (
        n_cases_per_spec
        if n_cases_per_spec is not None
        else int(round(_lerp(bounds["n_cases"], 0.5)))
    )

    specs: list[ScenarioSpec] = []
    rejections: list[SpecRejection] = []

    for archetype in archetypes:
        is_chain = archetype in CHAIN_ARCHETYPES
        new_device, foreign_ip = ARCHETYPE_FLAGS.get(archetype, (True, False))
        for lo_frac, hi_frac in AMOUNT_TERCILES:
            for day_part in ARCHETYPE_DAY_PARTS.get(archetype, ("chieu",)):
                raw = {
                    "archetype": archetype,
                    "description": (
                        f"luat: {archetype}, so tien {lo_frac:.0%}-{hi_frac:.0%} bien hieu chinh, "
                        f"khung gio {day_part}"
                    ),
                    "n_cases": n_cases,
                    "chain_length": (
                        int(round(_lerp(bounds["chain_length"], 0.5))) if is_chain else 2
                    ),
                    "amount_vnd_min": _lerp_round(bounds["amount_vnd"], lo_frac),
                    "amount_vnd_max": _lerp_round(bounds["amount_vnd"], hi_frac),
                    "hop_gap_minutes_min": _lerp_round(bounds["hop_gap_minutes"], 0.1, 3),
                    "hop_gap_minutes_max": _lerp_round(bounds["hop_gap_minutes"], 0.6, 3),
                    "amount_decay": _lerp_round(bounds["amount_decay"], 0.5, 4),
                    "day_part": day_part,
                    "channel": ARCHETYPE_CHANNEL.get(archetype, "domestic"),
                    "new_device": new_device,
                    "foreign_ip": foreign_ip,
                    "burst_count": int(round(_lerp(bounds["burst_count"], 0.5))),
                }
                spec, rejection = validate_spec(
                    raw, bounds, origin="rule",
                    provenance={"provider": "rule", "generator": "rule_specs.build_rule_specs"},
                )
                if spec is not None:
                    specs.append(spec)
                else:
                    rejections.append(rejection)  # type: ignore[arg-type]

    return specs, rejections
