"""Spec generator — adapter gọi LLM độc lập nhà cung cấp (Mục 6.3).

=== BA CAM KẾT CỦA FILE NÀY ===

1. KHÔNG GỬI DỮ LIỆU GIAO DỊCH THẬT (Mục 6.6). Prompt chỉ được dựng từ:
   tên cột của schema, tên kiểu hình, từ vựng đóng, và biên tham số TỔNG
   HỢP do calibration.py sinh ra. Hàm assert_prompt_free_of_raw_data()
   kiểm tra điều này bằng cách đối chiếu với dữ liệu thật, và
   tests/test_augmentation.py chạy kiểm tra đó tự động.

2. KHÔNG ĐỌC KHÓA API TỪ MÃ NGUỒN. Khóa chỉ lấy từ biến môi trường do
   người dùng tự đặt (mặc định DEEPSEEK_API_KEY / OPENAI_API_KEY). Khóa
   không bao giờ được ghi log, in ra, hay lưu vào manifest — manifest chỉ
   ghi TÊN biến môi trường đã dùng.

3. KHÔNG LÀM BỘ SINH TRỞ THÀNH PHỤ THUỘC VẬN HÀNH (Mục 6.2). File này
   KHÔNG được import từ src/drre/ hoặc src/dashboard/. Sau khi dataset đã
   có phiên bản, huấn luyện và vận hành không gọi API nữa.

=== GÁN NHÃN NGUỒN GỐC ===

Transport nào không thực sự gọi model (OfflineVariantTransport dùng cho
kiểm thử và cho trường hợp không có khóa) sẽ khai `is_model_backed=False`,
và SpecGenerator tự gán origin="rule". Nhờ vậy KHÔNG THỂ vô tình dán nhãn
"llm" cho đặc tả không do LLM sinh — điều kiện tiên quyết để giao thức
đánh giá so sánh được cấu hình (ii) và (iii).
"""

from __future__ import annotations

import json
import os
import random
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import pandas as pd

from src.augmentation.calibration import Calibration
from src.augmentation.spec_schema import (
    ARCHETYPES,
    CHAIN_ARCHETYPES,
    ScenarioSpec,
    SpecRejection,
    spec_template_for_prompt,
    validate_spec,
)

#: Tăng khi đổi nội dung prompt — ghi vào manifest để tái lập được.
PROMPT_VERSION = "spec-gen-v1"

#: Dấu mốc bao quanh phần payload JSON trong prompt. Cần mốc tường minh vì
#: prompt còn chứa một ví dụ JSON cho model đọc; nếu bóc payload bằng cách
#: tìm dấu ngoặc đầu tiên thì sẽ lấy nhầm ví dụ đó.
PAYLOAD_BEGIN = "<<<PAYLOAD"
PAYLOAD_END = "PAYLOAD>>>"

PROVIDER_BASE_URLS: dict[str, str] = {
    "deepseek": "https://api.deepseek.com",
    "openai": "https://api.openai.com/v1",
}

PROVIDER_DEFAULT_KEY_ENV: dict[str, str] = {
    "deepseek": "DEEPSEEK_API_KEY",
    "openai": "OPENAI_API_KEY",
}

SYSTEM_PROMPT = (
    "Bạn là chuyên gia điển hình hóa (typology) gian lận ngân hàng. Nhiệm vụ của bạn là "
    "đề xuất BIẾN THỂ của các kiểu hình gian lận đã biết, dưới dạng ĐẶC TẢ THAM SỐ JSON. "
    "Bạn KHÔNG sinh dữ liệu giao dịch. Bạn KHÔNG được phát minh tên trường mới, "
    "không được dùng giá trị ngoài danh sách cho phép, và mọi tham số số học PHẢI nằm "
    "trong biên được cung cấp. Chỉ trả về JSON hợp lệ, không giải thích thêm."
)


class SpecTransport(Protocol):
    """Lớp vận chuyển prompt → chuỗi JSON. Cho phép thay nhà cung cấp hoặc
    chạy offline mà không đổi phần còn lại của tầng."""

    is_model_backed: bool
    name: str

    def __call__(self, system_prompt: str, user_prompt: str) -> str: ...


@dataclass
class LLMConfig:
    """Cấu hình nhà cung cấp. KHÔNG chứa khóa API, chỉ chứa TÊN biến môi trường."""

    provider: str = "deepseek"
    model: str = "deepseek-chat"
    base_url: str | None = None
    api_key_env: str | None = None
    temperature: float = 0.8
    max_tokens: int = 2000
    timeout_seconds: float = 60.0

    def resolved_base_url(self) -> str:
        if self.base_url:
            return self.base_url.rstrip("/")
        try:
            return PROVIDER_BASE_URLS[self.provider]
        except KeyError:
            raise ValueError(
                f"Không biết base_url cho provider '{self.provider}'. "
                f"Hãy truyền base_url, hoặc dùng một trong {sorted(PROVIDER_BASE_URLS)}."
            ) from None

    def resolved_key_env(self) -> str:
        return self.api_key_env or PROVIDER_DEFAULT_KEY_ENV.get(
            self.provider, "LLM_API_KEY")

    def to_json(self) -> dict[str, Any]:
        """Thông tin ghi vào manifest — tuyệt đối KHÔNG chứa giá trị khóa."""
        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.resolved_base_url(),
            "api_key_env": self.resolved_key_env(),
            "temperature": self.temperature,
            "prompt_version": PROMPT_VERSION,
        }


class HttpChatTransport:
    """Gọi endpoint chat completions tương thích OpenAI bằng thư viện chuẩn.

    Dùng urllib thay vì `requests`/`openai` để tầng này không thêm phụ
    thuộc mới vào requirements.txt.
    """

    is_model_backed = True

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.name = f"{config.provider}:{config.model}"
        key_env = config.resolved_key_env()
        key = os.environ.get(key_env, "").strip()
        if not key:
            raise RuntimeError(
                f"Chưa có khóa API trong biến môi trường {key_env}. "
                f"Hãy tự đặt biến này trên máy bạn (PowerShell: $env:{key_env}='...'), "
                "KHÔNG ghi khóa vào mã nguồn hay file cấu hình trong repo."
            )
        self._key = key  # chỉ giữ trong bộ nhớ, không log, không ghi manifest

    def __call__(self, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "response_format": {"type": "json_object"},
        }
        request = urllib.request.Request(
            f"{self.config.resolved_base_url()}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:  # không in body vì có thể vọng lại header
            raise RuntimeError(
                f"Nhà cung cấp trả lỗi HTTP {exc.code} khi gọi {self.name}. "
                "Kiểm tra khóa API, hạn mức, và tên model."
            ) from None
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Không kết nối được tới nhà cung cấp ({exc.reason}). Nếu máy không có "
                "internet, dùng OfflineVariantTransport để chạy cấu hình (ii)."
            ) from None
        return body["choices"][0]["message"]["content"]


class OfflineVariantTransport:
    """Transport KHÔNG gọi model — sinh biến thể tất định trong biên.

    Dùng cho: kiểm thử (không được phụ thuộc mạng), và cho người chạy lại
    pipeline khi không có khóa API. Vì không có LLM tham gia, SpecGenerator
    sẽ gán origin="rule" cho mọi đặc tả từ transport này.
    """

    is_model_backed = False
    name = "offline:variant"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def __call__(self, system_prompt: str, user_prompt: str) -> str:
        spec_request = extract_payload(user_prompt)
        bounds = spec_request["bounds"]
        archetypes = spec_request["archetypes"]
        n = int(spec_request["n_specs"])
        rng = random.Random(self.seed)

        def pick(key: str, integer: bool = False) -> float:
            lo, hi = bounds[key]
            v = rng.uniform(float(lo), float(hi))
            return int(round(v)) if integer else round(v, 4)

        specs = []
        for i in range(n):
            arch = archetypes[i % len(archetypes)]
            a_lo, a_hi = bounds["amount_vnd"]
            lo = rng.uniform(float(a_lo), float(a_hi))
            hi = rng.uniform(lo, float(a_hi))
            g_lo, g_hi = bounds["hop_gap_minutes"]
            gl = rng.uniform(float(g_lo), float(g_hi))
            gh = rng.uniform(gl, float(g_hi))
            specs.append({
                "archetype": arch,
                "description": f"bien the offline tat dinh #{i} cho {arch}",
                "n_cases": pick("n_cases", integer=True),
                "chain_length": pick("chain_length", integer=True) if arch in CHAIN_ARCHETYPES else 2,
                "amount_vnd_min": round(lo, 2),
                "amount_vnd_max": round(hi, 2),
                "hop_gap_minutes_min": round(gl, 3),
                "hop_gap_minutes_max": round(gh, 3),
                "amount_decay": pick("amount_decay"),
                "day_part": ["rang_sang", "dem", "toi", "chieu"][i % 4],
                "channel": ["domestic", "e_wallet"][i % 2],
                "new_device": i % 2 == 0,
                "foreign_ip": i % 3 == 0,
                "burst_count": pick("burst_count", integer=True),
            })
        return json.dumps({"specs": specs}, ensure_ascii=False)


# --------------------------------------------------------------------------
# Dựng prompt + rào an toàn dữ liệu
# --------------------------------------------------------------------------

def build_user_prompt(
    calibration: Calibration,
    archetypes: tuple[str, ...],
    n_specs: int,
    schema_columns: tuple[str, ...],
) -> str:
    """Dựng prompt CHỈ từ metadata. Không có dòng giao dịch nào ở đây."""
    request = {
        "n_specs": n_specs,
        "archetypes": list(archetypes),
        "bounds": {k: list(v) for k, v in calibration.bounds.items()},
        "amount_unit": calibration.unit,
        "allowed_values": spec_template_for_prompt(calibration.bounds),
    }
    return (
        "Bối cảnh: hệ thống phát hiện gian lận giao dịch ngân hàng số tại Việt Nam.\n"
        f"Các cột của bảng giao dịch (chỉ để bạn biết ngữ cảnh, KHÔNG cần sinh dữ liệu): "
        f"{', '.join(schema_columns)}.\n\n"
        f"Hãy đề xuất {n_specs} đặc tả kịch bản khác biệt nhau, phủ các kiểu hình được yêu cầu. "
        "Mỗi đặc tả phải là một object có ĐÚNG các khóa trong allowed_values, mọi giá trị số "
        "nằm trong biên bounds, mọi giá trị chữ lấy từ danh sách cho phép. "
        "Với kiểu hình không phải chuỗi, đặt chain_length = 2.\n"
        'Trả về JSON đúng dạng: {"specs": [ {...}, {...} ]}\n\n'
        f"{PAYLOAD_BEGIN}\n" + json.dumps(request, ensure_ascii=False) + f"\n{PAYLOAD_END}"
    )


def extract_payload(user_prompt: str) -> dict[str, Any]:
    """Bóc phần payload JSON của prompt, dựa vào dấu mốc tường minh.

    Raises:
        ValueError: nếu prompt không có cặp dấu mốc hợp lệ.
    """
    try:
        start = user_prompt.index(PAYLOAD_BEGIN) + len(PAYLOAD_BEGIN)
        end = user_prompt.index(PAYLOAD_END, start)
    except ValueError:
        raise ValueError(
            f"Prompt thiếu cặp dấu mốc {PAYLOAD_BEGIN} ... {PAYLOAD_END}"
        ) from None
    return json.loads(user_prompt[start:end].strip())


def assert_prompt_free_of_raw_data(
    prompt: str, real_df: pd.DataFrame, sample_size: int = 200
) -> None:
    """Rào an toàn: khẳng định prompt không chứa định danh hay số tiền thật.

    Kiểm tra mẫu ngẫu nhiên các account_id/counterparty_id và chuỗi biểu
    diễn của amount trong dữ liệu thật. Đây là kiểm tra CHẶN, gọi trước
    mỗi lần gửi prompt (xem SpecGenerator.propose).

    Raises:
        AssertionError: nếu tìm thấy bất kỳ định danh thật nào trong prompt.
    """
    sample = real_df.head(sample_size) if len(real_df) > sample_size else real_df
    ids: set[str] = set()
    for col in ("account_id", "counterparty_id"):
        if col in sample.columns:
            ids |= {str(v) for v in sample[col].dropna().unique()}
    leaked = sorted(i for i in ids if len(i) >= 4 and i in prompt)
    if leaked:
        raise AssertionError(
            f"Prompt chứa {len(leaked)} định danh từ dữ liệu thật (vd. {leaked[:3]}). "
            "Tầng tăng cường KHÔNG được gửi dữ liệu giao dịch thật cho nhà cung cấp LLM."
        )
    if "amount" in sample.columns:
        amounts = {f"{float(v):.2f}" for v in sample["amount"].dropna().head(50)}
        leaked_amounts = sorted(a for a in amounts if a in prompt)
        if leaked_amounts:
            raise AssertionError(
                f"Prompt chứa số tiền của giao dịch thật (vd. {leaked_amounts[:3]}). "
                "Chỉ được gửi biên tham số tổng hợp."
            )


def _extract_specs(payload: str) -> tuple[list[dict[str, Any]], SpecRejection | None]:
    """Bóc danh sách đặc tả từ chuỗi model trả về, chịu được rào ```json."""
    text = payload.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        return [], SpecRejection("parse", f"không phải JSON hợp lệ: {exc}")
    if isinstance(obj, list):
        return obj, None
    if isinstance(obj, dict) and isinstance(obj.get("specs"), list):
        return obj["specs"], None
    return [], SpecRejection("parse", f"JSON không có mảng 'specs': khóa nhận được {sorted(obj)[:5]}")


@dataclass
class ProposalResult:
    """Kết quả một lượt đề xuất đặc tả, đủ thông tin cho manifest."""

    specs: list[ScenarioSpec] = field(default_factory=list)
    rejections: list[SpecRejection] = field(default_factory=list)
    transport_name: str = ""
    origin: str = "rule"
    n_requested: int = 0

    @property
    def rejection_rate(self) -> float:
        total = len(self.specs) + len(self.rejections)
        return len(self.rejections) / total if total else 0.0

    def rejection_histogram(self) -> dict[str, int]:
        hist: dict[str, int] = {}
        for r in self.rejections:
            hist[r.reason_code] = hist.get(r.reason_code, 0) + 1
        return hist


class SpecGenerator:
    """Yêu cầu bộ sinh (LLM hoặc offline) đề xuất đặc tả, rồi lọc qua validator."""

    def __init__(
        self,
        calibration: Calibration,
        transport: SpecTransport | Callable[[str, str], str] | None = None,
        config: LLMConfig | None = None,
    ) -> None:
        self.calibration = calibration
        self.config = config or LLMConfig()
        self.transport = transport or OfflineVariantTransport()

    @property
    def origin(self) -> str:
        """"llm" chỉ khi transport thực sự gọi model — xem docstring module."""
        return "llm" if getattr(self.transport, "is_model_backed", False) else "rule"

    def propose(
        self,
        n_specs: int,
        real_df: pd.DataFrame,
        archetypes: tuple[str, ...] = ARCHETYPES,
        schema_columns: tuple[str, ...] = (
            "account_id", "counterparty_id", "timestamp", "amount",
            "channel", "device_id", "ip_country", "label",
        ),
    ) -> ProposalResult:
        """Đề xuất n_specs đặc tả, trả về danh sách hợp lệ + danh sách bị loại.

        `real_df` CHỈ dùng cho rào an toàn assert_prompt_free_of_raw_data —
        không có nội dung nào của nó đi vào prompt.
        """
        prompt = build_user_prompt(self.calibration, archetypes, n_specs, schema_columns)
        assert_prompt_free_of_raw_data(prompt, real_df)

        result = ProposalResult(
            transport_name=getattr(self.transport, "name", type(self.transport).__name__),
            origin=self.origin,
            n_requested=n_specs,
        )
        raw_payload = self.transport(SYSTEM_PROMPT, prompt)
        raw_specs, parse_error = _extract_specs(raw_payload)
        if parse_error is not None:
            result.rejections.append(parse_error)
            return result

        provenance = {
            **(self.config.to_json() if self.origin == "llm" else {"provider": "offline"}),
            "transport": result.transport_name,
            "prompt_version": PROMPT_VERSION,
        }
        for raw in raw_specs:
            spec, rejection = validate_spec(
                raw, self.calibration.bounds, origin=self.origin, provenance=provenance)
            if spec is not None:
                result.specs.append(spec)
            else:
                result.rejections.append(rejection)  # type: ignore[arg-type]
        return result


def make_transport(config: LLMConfig, offline: bool = False, seed: int = 0) -> SpecTransport:
    """Chọn transport. offline=True (hoặc thiếu khóa API) → không gọi mạng."""
    if offline:
        return OfflineVariantTransport(seed=seed)
    return HttpChatTransport(config)
