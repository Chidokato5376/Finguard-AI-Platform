"""Spec validator + từ vựng đóng cho đặc tả kịch bản (Mục 6.3).

Đặc tả kịch bản là hợp đồng duy nhất giữa LLM và phần còn lại của hệ
thống. Mọi thứ LLM trả về PHẢI đi qua validate_spec() trước khi tới
simulator — đây là chỗ cô lập sự không tất định của LLM.

Ba loại vi phạm bị loại (Mục 6.4, lớp "trước khi sinh"):
  1. Vi phạm từ vựng đóng  — archetype/channel/day_part ngoài danh sách.
  2. Vượt biên giá trị      — tham số ngoài biên do calibration.py suy ra.
  3. Khóa không thuộc mẫu   — thêm khóa lạ, hoặc thiếu khóa bắt buộc.

Tỷ lệ bị loại được dataset.py ghi vào manifest như chỉ báo độ ổn định của
bộ sinh (Mục 6.4).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

from src.data.schema import Channel

# --------------------------------------------------------------------------
# Từ vựng ĐÓNG — LLM không được phát minh giá trị mới ở các trường này
# --------------------------------------------------------------------------

#: Các kiểu hình gian lận được mô tả trong tri thức miền. Ba kiểu đầu trùng
#: với FRAUD_PATTERN_NAMES của src/data/synthetic_generator.py (giữ tương
#: thích để phân tích concept drift dùng chung mã kiểu hình). Kiểu thứ tư
#: (smurfing) là kiểu hình bổ sung cho tầng tăng cường.
ARCHETYPES: tuple[str, ...] = (
    "money_mule_chain",
    "account_takeover",
    "investment_romance_scam",
    "smurfing_structuring",
)

#: Kiểu hình có cấu trúc chuỗi nhiều chặng (dùng chain_length, amount_decay)
CHAIN_ARCHETYPES: frozenset[str] = frozenset({"money_mule_chain", "smurfing_structuring"})

CHANNEL_VOCAB: tuple[str, ...] = tuple(c.value for c in Channel)

DAY_PART_VOCAB: tuple[str, ...] = ("rang_sang", "sang", "trua", "chieu", "toi", "dem")

#: Giờ bắt đầu của từng khung thời gian trong ngày (giờ địa phương)
DAY_PART_HOURS: dict[str, tuple[int, int]] = {
    "rang_sang": (0, 5),
    "sang": (5, 11),
    "trua": (11, 13),
    "chieu": (13, 18),
    "toi": (18, 22),
    "dem": (22, 24),
}

#: Cột chỉ phục vụ truy vết — KHÔNG được dùng làm đặc trưng đầu vào.
#: Dùng làm NHÃN cho đầu phân loại tình huống thì hợp lệ (xem __init__.py).
TRACE_ONLY_COLUMNS: tuple[str, ...] = ("fraud_scenario_type", "scenario_code", "spec_id")

#: Khóa bắt buộc trong JSON do LLM trả về. LLM KHÔNG được thêm khóa khác.
REQUIRED_SPEC_KEYS: frozenset[str] = frozenset({
    "archetype", "description", "n_cases", "chain_length", "amount_vnd_min",
    "amount_vnd_max", "hop_gap_minutes_min", "hop_gap_minutes_max",
    "amount_decay", "day_part", "channel", "new_device", "foreign_ip", "burst_count",
})

#: Biên tuyệt đối — chặn trên cùng mọi biên hiệu chỉnh. Kể cả khi
#: calibration đề xuất rộng hơn, vượt các mức này là lỗi cấu hình.
HARD_LIMITS: dict[str, tuple[float, float]] = {
    "n_cases": (1, 500),
    "chain_length": (2, 12),
    "burst_count": (1, 30),
    "amount_vnd": (10_000.0, 10_000_000_000.0),
    "hop_gap_minutes": (0.1, 10_080.0),  # tối đa 7 ngày mỗi chặng
    "amount_decay": (0.50, 1.0),
}

MAX_DESCRIPTION_CHARS = 400


@dataclass(frozen=True)
class SpecRejection:
    """Một đặc tả bị loại, kèm lý do để ghi vào manifest."""

    reason_code: str  # "closed_vocabulary" | "out_of_bounds" | "schema" | "parse"
    detail: str
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScenarioSpec:
    """Đặc tả kịch bản đã hợp lệ — đầu vào duy nhất của simulator.

    Mọi trường số học đã được validate_spec() kiểm tra nằm trong biên hiệu
    chỉnh, nên simulator KHÔNG cần kiểm tra lại.
    """

    archetype: str
    description: str
    n_cases: int
    chain_length: int
    amount_vnd: tuple[float, float]
    hop_gap_minutes: tuple[float, float]
    amount_decay: float
    day_part: str
    channel: str
    new_device: bool
    foreign_ip: bool
    burst_count: int
    origin: str  # "rule" | "llm"
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def spec_id(self) -> str:
        """Mã băm ổn định của nội dung đặc tả — dùng để truy vết và tái lập."""
        payload = {k: v for k, v in asdict(self).items() if k != "provenance"}
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]

    @property
    def scenario_code(self) -> str:
        """Mã kịch bản ghi vào từng dòng giao dịch (cột truy vết)."""
        return f"{self.archetype}:{self.origin}:{self.spec_id}"

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["spec_id"] = self.spec_id
        d["scenario_code"] = self.scenario_code
        return d


def _as_float_pair(raw: dict[str, Any], lo_key: str, hi_key: str) -> tuple[float, float]:
    return float(raw[lo_key]), float(raw[hi_key])


def _within(value: float, bounds: tuple[float, float]) -> bool:
    """So sánh có sai số TƯƠNG ĐỐI.

    Sai số tuyệt đối không dùng được ở đây: biên số tiền có độ lớn ~1e8,
    nên một giá trị đã làm tròn 2 chữ số thập phân có thể lệch khỏi biên
    tới ~1e-3 — với sai số 1e-9 tuyệt đối thì đặc tả hợp lệ bị loại oan
    (lỗi đã gặp thật: amount_vnd_min=9536484.4 bị loại vì biên là
    9536484.40110087).
    """
    scale = max(1.0, abs(bounds[0]), abs(bounds[1]))
    tol = 1e-9 * scale
    return bounds[0] - tol <= value <= bounds[1] + tol


def validate_spec(
    raw: dict[str, Any],
    bounds: dict[str, tuple[float, float]],
    origin: str,
    provenance: dict[str, Any] | None = None,
) -> tuple[ScenarioSpec | None, SpecRejection | None]:
    """Kiểm tra một đặc tả thô (dict từ JSON của LLM hoặc từ rule_specs).

    Args:
        raw: dict đặc tả thô.
        bounds: biên hiệu chỉnh, khóa theo tên tham số — xem
            calibration.Calibration.bounds_for().
        origin: "rule" hoặc "llm" — ghi vào ScenarioSpec để giao thức đánh
            giá phân biệt được cấu hình (ii) và (iii).
        provenance: thông tin nhà cung cấp/model/phiên bản prompt/seed.

    Returns:
        (spec, None) nếu hợp lệ, hoặc (None, rejection) kèm lý do.
    """
    if not isinstance(raw, dict):
        return None, SpecRejection("schema", f"đặc tả không phải object JSON: {type(raw).__name__}")

    keys = set(raw)
    extra = keys - REQUIRED_SPEC_KEYS
    missing = REQUIRED_SPEC_KEYS - keys
    if extra:
        return None, SpecRejection("schema", f"khóa không thuộc mẫu: {sorted(extra)}", raw)
    if missing:
        return None, SpecRejection("schema", f"thiếu khóa bắt buộc: {sorted(missing)}", raw)

    # --- Lớp 1: từ vựng đóng ---
    if raw["archetype"] not in ARCHETYPES:
        return None, SpecRejection(
            "closed_vocabulary", f"archetype '{raw['archetype']}' ngoài từ vựng", raw)
    if raw["channel"] not in CHANNEL_VOCAB:
        return None, SpecRejection(
            "closed_vocabulary", f"channel '{raw['channel']}' ngoài từ vựng", raw)
    if raw["day_part"] not in DAY_PART_VOCAB:
        return None, SpecRejection(
            "closed_vocabulary", f"day_part '{raw['day_part']}' ngoài từ vựng", raw)
    for flag in ("new_device", "foreign_ip"):
        if not isinstance(raw[flag], bool):
            return None, SpecRejection("schema", f"{flag} phải là boolean", raw)

    if not isinstance(raw["description"], str) or len(raw["description"]) > MAX_DESCRIPTION_CHARS:
        return None, SpecRejection(
            "schema", f"description phải là chuỗi ≤ {MAX_DESCRIPTION_CHARS} ký tự", raw)

    # --- Lớp 2: biên giá trị ---
    try:
        amount = _as_float_pair(raw, "amount_vnd_min", "amount_vnd_max")
        gap = _as_float_pair(raw, "hop_gap_minutes_min", "hop_gap_minutes_max")
        n_cases = int(raw["n_cases"])
        chain_length = int(raw["chain_length"])
        burst_count = int(raw["burst_count"])
        decay = float(raw["amount_decay"])
    except (TypeError, ValueError) as exc:
        return None, SpecRejection("schema", f"tham số không phải số: {exc}", raw)

    if amount[0] > amount[1] or gap[0] > gap[1]:
        return None, SpecRejection("out_of_bounds", "khoảng giá trị có min > max", raw)

    is_chain = raw["archetype"] in CHAIN_ARCHETYPES
    if not is_chain and chain_length != 2:
        # Kiểu hình không phải chuỗi: chain_length không có nghĩa, buộc = 2
        # để đặc tả không mang tham số gây nhầm lẫn.
        return None, SpecRejection(
            "schema", f"archetype '{raw['archetype']}' không phải chuỗi, chain_length phải = 2", raw)

    checks: list[tuple[str, float, str]] = [
        ("amount_vnd", amount[0], "amount_vnd_min"),
        ("amount_vnd", amount[1], "amount_vnd_max"),
        ("hop_gap_minutes", gap[0], "hop_gap_minutes_min"),
        ("hop_gap_minutes", gap[1], "hop_gap_minutes_max"),
        ("amount_decay", decay, "amount_decay"),
        ("n_cases", n_cases, "n_cases"),
        ("burst_count", burst_count, "burst_count"),
    ]
    # chain_length chỉ được đối chiếu với biên hiệu chỉnh khi kiểu hình THỰC SỰ
    # là chuỗi. Với kiểu hình không phải chuỗi, giá trị bắt buộc là 2 và biên
    # hiệu chỉnh (vd. 3–6 chặng từ tri thức miền) không áp dụng — nếu vẫn đối
    # chiếu thì mọi đặc tả không phải chuỗi đều bị loại oan.
    if is_chain:
        checks.append(("chain_length", chain_length, "chain_length"))
    elif not _within(chain_length, HARD_LIMITS["chain_length"]):  # pragma: no cover
        return None, SpecRejection(
            "out_of_bounds", f"chain_length={chain_length} vượt biên tuyệt đối", raw)

    for param, value, label in checks:
        hard = HARD_LIMITS[param]
        if not _within(value, hard):
            return None, SpecRejection(
                "out_of_bounds", f"{label}={value} vượt biên tuyệt đối {hard}", raw)
        soft = bounds.get(param)
        if soft is not None and not _within(value, soft):
            return None, SpecRejection(
                "out_of_bounds", f"{label}={value} ngoài biên hiệu chỉnh {soft}", raw)

    if origin not in ("rule", "llm"):
        raise ValueError(f"origin phải là 'rule' hoặc 'llm', nhận '{origin}'")

    spec = ScenarioSpec(
        archetype=raw["archetype"],
        description=raw["description"].strip(),
        n_cases=n_cases,
        chain_length=chain_length,
        amount_vnd=amount,
        hop_gap_minutes=gap,
        amount_decay=decay,
        day_part=raw["day_part"],
        channel=raw["channel"],
        new_device=raw["new_device"],
        foreign_ip=raw["foreign_ip"],
        burst_count=burst_count,
        origin=origin,
        provenance=dict(provenance or {}),
    )
    return spec, None


def spec_template_for_prompt(bounds: dict[str, tuple[float, float]]) -> dict[str, Any]:
    """Mẫu JSON + biên hợp lệ, dùng để mô tả hợp đồng cho LLM.

    CHỈ chứa tên cột, từ vựng đóng và biên tham số TỔNG HỢP — tuyệt đối
    không chứa dòng giao dịch thật nào (Mục 6.6).
    """
    return {
        "archetype": list(ARCHETYPES),
        "description": f"chuỗi ≤ {MAX_DESCRIPTION_CHARS} ký tự, mô tả kiểu hình",
        "n_cases": list(bounds.get("n_cases", HARD_LIMITS["n_cases"])),
        "chain_length": list(bounds.get("chain_length", HARD_LIMITS["chain_length"])),
        "amount_vnd_min/amount_vnd_max": list(bounds.get("amount_vnd", HARD_LIMITS["amount_vnd"])),
        "hop_gap_minutes_min/hop_gap_minutes_max": list(
            bounds.get("hop_gap_minutes", HARD_LIMITS["hop_gap_minutes"])),
        "amount_decay": list(bounds.get("amount_decay", HARD_LIMITS["amount_decay"])),
        "day_part": list(DAY_PART_VOCAB),
        "channel": list(CHANNEL_VOCAB),
        "new_device": [True, False],
        "foreign_ip": [True, False],
        "burst_count": list(bounds.get("burst_count", HARD_LIMITS["burst_count"])),
    }
