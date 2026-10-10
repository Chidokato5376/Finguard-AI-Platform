"""Simulator tất định: đặc tả → dòng giao dịch đúng schema (Mục 6.3).

Vì sao phải tất định: nhãn được gán THEO CẤU TẠO (dòng nào do kịch bản
gian lận sinh ra thì label=True, không cần ai gán tay), phân phối kiểm
soát được, và với cùng (spec, seed) thì kết quả trùng khít — điều kiện để
manifest tái lập được dataset (Mục 6.2).

Tái sử dụng mã tài khoản (Mục 6.3): tài khoản "nạn nhân"/"nguồn" được lấy
từ chính tập giao dịch nền, nên khi dựng đồ thị ở bước thông thường, các
cạnh tổng hợp nối liền vào mạng lưới thật thay vì tạo một ốc đảo riêng.
Chỉ các tài khoản trung gian (mule) và tài khoản thụ hưởng mới là mã mới.

Dữ liệu phẳng: simulator chỉ phát ra các dòng theo đúng cột của bảng gốc.
Không có cấu trúc đồ thị nào được ghi ra — nó được suy ra sau.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Sequence

import numpy as np
import pandas as pd

from src.augmentation.spec_schema import DAY_PART_HOURS, ScenarioSpec
from src.data.schema import DataSource

#: Tiền tố mã tài khoản tổng hợp. Dùng tiền tố rõ ràng để audit được, và
#: vì cột account_id KHÔNG phải đặc trưng đầu vào nên không gây rò rỉ.
SYNTH_PREFIX = "SYN"

#: Mã quốc gia dùng khi đặc tả bật foreign_ip.
FOREIGN_IP_CODES: tuple[str, ...] = ("KH", "PH", "TW", "XX")


def infer_amount_decimals(amounts: Sequence[float], share: float = 0.99) -> int | None:
    """Suy độ lượng tử hóa số tiền của dữ liệu nền.

    Vì sao cần: PaySim có 100% số tiền khớp đúng 2 chữ số thập phân, trong
    khi simulator phát ra float đầy đủ sau các phép nhân suy giảm/nhiễu.
    Khi đó bộ phân biệt tách dữ liệu tổng hợp khỏi dữ liệu thật với AUC
    1,000 chỉ bằng phần thập phân của số tiền — đo thực tế, không phải giả
    thuyết. Làm tròn theo đúng độ lượng tử hóa của nền sẽ xóa dấu vết này.

    Returns:
        Số chữ số thập phân (0, 1 hoặc 2) nếu ≥ `share` số tiền của nền
        khớp mức đó, hoặc None nếu nền dùng độ chính xác đầy đủ (vd.
        synthetic_vn: 0% khớp 2 chữ số) — khi đó không làm tròn.
    """
    values = np.asarray(amounts, dtype=float)
    if values.size == 0:
        return None
    for decimals in (0, 1, 2):
        scale = 10.0 ** decimals
        if np.mean(np.abs(values * scale - np.round(values * scale)) < 1e-6) >= share:
            return decimals
    return None


def channel_override_needed(spec_channels: Sequence[str], background_channels: Sequence[str]) -> bool:
    """Nền có dùng cùng từ vựng kênh với đặc tả hay không.

    Vì sao cần: `src/data/schema.py` định nghĩa Channel =
    domestic/international/e_wallet/unknown, nhưng loader PaySim ghi thẳng
    loại giao dịch của PaySim (cash_out, transfer, payment, cash_in, debit)
    vào cột `channel`. Tức là cột này CHƯA thống nhất giữa các nguồn. Nếu
    chèn dòng tổng hợp mang giá trị không tồn tại trong nền, bộ phân biệt
    tách được ngay với AUC 1,000 — đo thực tế trên PaySim.
    """
    background_set = {str(c) for c in pd.unique(pd.Series(background_channels).dropna())}
    return not background_set >= {str(c) for c in set(spec_channels)}


def _spec_rng(spec: ScenarioSpec, seed: int) -> np.random.Generator:
    """Generator numpy tất định cho một (spec, seed) — không dùng hash() vì
    hash của chuỗi bị muối lại mỗi lần chạy Python."""
    return np.random.default_rng((int(spec.spec_id, 16) ^ (seed * 0x9E3779B1)) % (2 ** 63))


def _case_start(
    rng: random.Random,
    origin: datetime,
    horizon_seconds: float,
    day_part: str,
    lead_seconds: float = 0.0,
) -> datetime:
    """Chọn thời điểm bắt đầu một ca: ngày ngẫu nhiên trong cửa sổ nền,
    giờ theo khung thời gian của đặc tả.

    QUAN TRỌNG: ca được rải ĐỀU trên toàn bộ cửa sổ thời gian của dữ liệu
    nền. Nếu dồn vào một khối thời gian riêng, time-based split sẽ đẩy
    toàn bộ nhãn dương vào một tập duy nhất — lỗi đã gặp thật trong dự án
    này, xem ghi chú trong src/data/synthetic_generator.py.

    `lead_seconds` là khoảng thời gian cần CHỪA TRƯỚC thời điểm bắt đầu,
    dành cho các kịch bản phát sinh lịch sử ở quá khứ (vd. chiếm quyền tài
    khoản cần vài giao dịch nhỏ trước đó). Nếu không chừa, các dòng lịch sử
    sẽ rơi ra ngoài cửa sổ nền và trở thành những dòng sớm nhất của cả tập
    — đúng loại dấu vết nhân tạo mà kiểm định phân biệt sẽ bắt được.
    """
    usable = max(horizon_seconds - 86400, 0.0)
    lead = min(max(lead_seconds, 0.0), usable * 0.5)
    day_offset = (lead + rng.uniform(0, max(usable - lead, 0.0))) // 86400
    hour_lo, hour_hi = DAY_PART_HOURS[day_part]
    # Phân phối tam giác, đỉnh ở giữa khung giờ, thay vì phân phối đều.
    # Phân phối đều để lại dấu vết nhân tạo: giờ của giao dịch tổng hợp trải
    # phẳng và cắt vuông ở hai biên khung, khác hẳn phân phối thật.
    hour_hi = max(hour_hi - 1e-6, hour_lo)
    hour = rng.triangular(hour_lo, hour_hi, 0.5 * (hour_lo + hour_hi))
    base = origin + timedelta(days=float(day_offset))
    return base.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(hours=hour)


def _row(
    account_id: str,
    counterparty_id: str | None,
    timestamp: datetime,
    amount: float,
    spec: ScenarioSpec,
    label: bool,
    device_id: str,
    ip_country: str,
    source: str,
) -> dict:
    return {
        "account_id": account_id,
        "counterparty_id": counterparty_id,
        "timestamp": timestamp,
        "amount": float(amount),
        "channel": spec.channel,
        "device_id": device_id,
        "ip_country": ip_country,
        "label": bool(label),
        "source": source,
        "fraud_scenario_type": spec.archetype,
        "scenario_code": spec.scenario_code,
        "spec_id": spec.spec_id,
    }


def _device(spec: ScenarioSpec, account: str, tag: str) -> str:
    return f"dev_new_{tag}" if spec.new_device else f"dev_{account}"


def _ip(spec: ScenarioSpec, rng: random.Random) -> str:
    return rng.choice(FOREIGN_IP_CODES) if spec.foreign_ip else "VN"


#: Số mẫu thật tối thiểu TRONG KHOẢNG của đặc tả để dùng phân phối thực
#: nghiệm. Dưới mức này, mẫu quá ít nên quay về phân phối đều.
MIN_POOL_SAMPLES = 5

#: Số giá trị KHÁC NHAU tối thiểu của cả bể mẫu. Bể quá nhỏ (vd. 28 giá
#: trị cho 675 dòng cần sinh) thì mỗi giá trị bị dùng lại hàng chục lần,
#: và chất lượng của cách bốc mẫu không kiểm chứng được bằng kiểm định
#: phân biệt (xem quality.MIN_CLASS_FOR_ADVERSARIAL) — khi đó dùng phân
#: phối đều cho minh bạch thay vì giả vờ khớp phân phối thật.
MIN_POOL_DISTINCT = 30


def _draw_amount(
    rng: random.Random, spec: ScenarioSpec, pool: np.ndarray | None
) -> float:
    """Lấy một số tiền trong khoảng của đặc tả.

    Ưu tiên bốc từ PHÂN PHỐI THỰC NGHIỆM của số tiền gian lận thật, chỉ
    giới hạn trong khoảng mà đặc tả cho phép. Phân phối đều để lại dấu vết
    nhân tạo rõ rệt: đo thực tế trên synthetic_vn, bộ sinh biến thể dùng
    phân phối đều cho AUC kiểm định phân biệt 0,891 (ngưỡng đạt là ≤ 0,60).
    """
    lo, hi = spec.amount_vnd
    if pool is not None and pool.size:
        in_range = pool[(pool >= lo) & (pool <= hi)]
        if in_range.size >= MIN_POOL_SAMPLES:
            # Bốc một mẫu thật rồi nhiễu nhẹ ±5% để không lặp lại nguyên giá trị
            base = float(in_range[rng.randrange(in_range.size)])
            return float(min(max(base * rng.uniform(0.95, 1.05), lo), hi))
    return rng.uniform(lo, hi)


def simulate_spec(
    spec: ScenarioSpec,
    background_accounts: Sequence[str],
    time_origin: datetime,
    horizon_seconds: float,
    seed: int,
    source: str = DataSource.SYNTHETIC_VN.value,
    amount_pool: Sequence[float] | None = None,
    amount_decimals: int | None = None,
    channel_pool: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Sinh toàn bộ các ca của MỘT đặc tả.

    Args:
        spec: đặc tả đã hợp lệ (đã qua validate_spec).
        background_accounts: mã tài khoản có thật trong dữ liệu nền, dùng
            làm tài khoản nguồn/nạn nhân để cạnh tổng hợp nối vào mạng thật.
        time_origin: mốc đầu cửa sổ thời gian của dữ liệu nền.
        horizon_seconds: độ dài cửa sổ thời gian của dữ liệu nền.
        seed: hạt giống — cùng (spec, seed) luôn cho cùng kết quả.
        source: giá trị cột `source` ghi vào từng dòng.
        amount_pool: số tiền của các giao dịch gian lận THẬT, dùng làm
            phân phối thực nghiệm để bốc số tiền (xem _draw_amount). Đây là
            dữ liệu tổng hợp ở mức phân phối, KHÔNG gửi cho nhà cung cấp LLM.
        amount_decimals: làm tròn số tiền về bấy nhiêu chữ số thập phân để
            khớp độ lượng tử hóa của nền — xem infer_amount_decimals().
        channel_pool: giá trị cột `channel` của các giao dịch gian lận THẬT.
            Nếu nền dùng từ vựng kênh khác với đặc tả (xem
            channel_override_needed), kênh của dòng tổng hợp được bốc theo
            phân phối thực nghiệm này và kênh trong đặc tả bị GHI ĐÈ —
            build.py ghi việc ghi đè đó vào manifest.

    Raises:
        ValueError: nếu không có tài khoản nền nào để tái sử dụng.
    """
    if not background_accounts:
        raise ValueError(
            "background_accounts rỗng — simulator cần mã tài khoản thật để tái sử dụng "
            "(xem docstring module về việc nối cạnh tổng hợp vào mạng lưới thật)."
        )
    pool = None
    if amount_pool is not None:
        candidate = np.sort(np.asarray(amount_pool, dtype=float))  # sort để tất định
        if np.unique(candidate).size >= MIN_POOL_DISTINCT:
            pool = candidate

    # Hạt giống phụ thuộc cả spec_id: hai đặc tả khác nhau không bao giờ
    # sinh ra cùng một dãy số, kể cả khi dùng cùng seed gốc.
    rng = random.Random(f"{seed}:{spec.spec_id}")
    rows: list[dict] = []

    # Chỉ kịch bản chiếm quyền tài khoản sinh dòng ở QUÁ KHỨ của thời điểm
    # đột biến (lịch sử nhỏ, đều đặn) — xem _account_takeover.
    lead_seconds = (
        max(1, spec.burst_count - 1) * 4 * 86400.0
        if spec.archetype == "account_takeover" else 0.0
    )
    # _case_start tự kẹp lead về tối đa nửa cửa sổ; tính lại đúng giá trị đã
    # được kẹp để phần lịch sử rải trong khoảng thật sự có chỗ.
    effective_lead = min(lead_seconds, max(horizon_seconds - 86400, 0.0) * 0.5)

    for case in range(spec.n_cases):
        start = _case_start(rng, time_origin, horizon_seconds, spec.day_part, lead_seconds)
        amount = _draw_amount(rng, spec, pool)
        victim = rng.choice(list(background_accounts))
        tag = f"{spec.spec_id}_{case}"

        if spec.archetype == "money_mule_chain":
            rows.extend(_mule_chain(spec, rng, victim, tag, start, amount, source))
        elif spec.archetype == "smurfing_structuring":
            rows.extend(_smurfing(spec, rng, victim, tag, start, amount, source))
        elif spec.archetype == "account_takeover":
            rows.extend(_account_takeover(
                spec, rng, victim, tag, start, amount, source, effective_lead))
        elif spec.archetype == "investment_romance_scam":
            rows.extend(_investment_scam(spec, rng, victim, tag, start, amount, source))
        else:  # pragma: no cover — validate_spec đã chặn từ vựng ngoài danh sách
            raise ValueError(f"Chưa cài đặt simulator cho archetype '{spec.archetype}'")

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Làm tròn tập trung ở một chỗ, để mọi kiểu hình đều khớp độ lượng tử
    # hóa của nền — kể cả số tiền sinh ra sau các phép nhân suy giảm.
    if amount_decimals is not None:
        df["amount"] = df["amount"].round(amount_decimals)
    # Khớp MARGINAL đã biết của nền (kênh) để kiểm định phân biệt dành cho
    # việc phát hiện dấu vết KHÔNG CHỦ Ý, chứ không phải phát hiện lại một
    # khác biệt mà ta đã biết là do từ vựng cột không thống nhất.
    if channel_pool is not None and channel_override_needed([spec.channel], channel_pool):
        values = pd.Series(channel_pool).dropna().to_numpy()
        if values.size:
            df["channel"] = _spec_rng(spec, seed).choice(values, size=len(df))
    return df


def _mule_chain(spec, rng, victim, tag, start, amount, source) -> list[dict]:
    """Tiền chảy qua chuỗi tài khoản trung gian rồi ra ngoài, số tiền gần
    như giữ nguyên, mỗi chặng cách nhau vài phút tới vài giờ."""
    chain = [victim] + [f"{SYNTH_PREFIX}_MULE_{tag}_{i}" for i in range(1, spec.chain_length)]
    rows = []
    t = start
    for hop in range(len(chain) - 1):
        hop_amount = amount * (spec.amount_decay ** hop)
        rows.append(_row(
            chain[hop], chain[hop + 1], t, hop_amount, spec, True,
            _device(spec, chain[hop], f"{tag}_{hop}"), _ip(spec, rng), source))
        t = t + timedelta(minutes=rng.uniform(*spec.hop_gap_minutes))
    return rows


def _smurfing(spec, rng, victim, tag, start, amount, source) -> list[dict]:
    """Chia nhỏ một khoản lớn thành nhiều khoản dưới ngưỡng, gửi tới nhiều
    tài khoản khác nhau trong thời gian ngắn, sau đó gom về một đầu mối."""
    n_split = max(2, spec.burst_count)
    collectors = [f"{SYNTH_PREFIX}_SMURF_{tag}_{i}" for i in range(n_split)]
    hub = f"{SYNTH_PREFIX}_HUB_{tag}"
    piece = amount / n_split
    rows = []
    t = start
    for i, collector in enumerate(collectors):
        jitter = piece * rng.uniform(0.85, 1.15)
        rows.append(_row(
            victim, collector, t, jitter, spec, True,
            _device(spec, victim, f"{tag}_{i}"), _ip(spec, rng), source))
        t = t + timedelta(minutes=rng.uniform(*spec.hop_gap_minutes))
    # Gom về đầu mối — chặng thứ hai của cấu trúc
    for i, collector in enumerate(collectors):
        rows.append(_row(
            collector, hub, t + timedelta(minutes=rng.uniform(*spec.hop_gap_minutes)),
            piece * spec.amount_decay, spec, True,
            _device(spec, collector, f"{tag}_h{i}"), _ip(spec, rng), source))
    return rows


def _account_takeover(spec, rng, victim, tag, start, amount, source, lead_seconds) -> list[dict]:
    """Tài khoản có lịch sử nhỏ, đều đặn, rồi đột ngột có giao dịch giá
    trị lớn từ thiết bị/IP lần đầu xuất hiện.

    Các dòng lịch sử nhỏ mang label=False (chúng KHÔNG phải gian lận) —
    đây là phần làm cho Behavior Memory có cái để so sánh. Chỉ dòng đột
    biến mang label=True.

    Lịch sử được rải trong khoảng `lead_seconds` đã được chừa trước thời
    điểm đột biến, nhân thêm 0,9 để không vượt ra ngoài cửa sổ nền khi
    _case_start làm tròn xuống theo ngày.
    """
    beneficiary = f"{SYNTH_PREFIX}_BEN_{tag}"
    small = amount / rng.uniform(8.0, 20.0)
    rows = []
    # Lịch sử bình thường trước thời điểm bị chiếm quyền
    n_history = max(1, spec.burst_count - 1)
    span = max(lead_seconds, 0.0) * 0.9
    for i in range(n_history):
        frac = (n_history - i) / n_history
        t = start - timedelta(seconds=span * frac * rng.uniform(0.7, 1.0))
        rows.append(_row(
            victim, f"{SYNTH_PREFIX}_SHOP_{tag}_{i % 3}", t,
            small * rng.uniform(0.6, 1.4), spec, False,
            f"dev_{victim}", "VN", source))
    # Giao dịch đột biến
    rows.append(_row(
        victim, beneficiary, start, amount, spec, True,
        _device(spec, victim, tag), _ip(spec, rng), source))
    return rows


def _investment_scam(spec, rng, victim, tag, start, amount, source) -> list[dict]:
    """Nạn nhân tự chuyển tiền nhiều lần cho cùng một đầu mối, số tiền
    tăng dần theo thời gian — không có dấu hiệu thiết bị lạ, nên đây là
    kiểu hình mà quy tắc "thiết bị mới" bỏ sót."""
    platform = f"{SYNTH_PREFIX}_PLAT_{tag}"
    rows = []
    t = start
    step_amount = amount / max(spec.burst_count, 1)
    for i in range(max(spec.burst_count, 1)):
        rows.append(_row(
            victim, platform, t, step_amount * (1.0 + 0.6 * i), spec, True,
            f"dev_{victim}", _ip(spec, rng), source))
        t = t + timedelta(minutes=rng.uniform(*spec.hop_gap_minutes) * rng.uniform(10, 60))
    return rows


def simulate_all(
    specs: Sequence[ScenarioSpec],
    background_accounts: Sequence[str],
    time_origin: datetime,
    horizon_seconds: float,
    seed: int,
    source: str = DataSource.SYNTHETIC_VN.value,
    amount_pool: Sequence[float] | None = None,
    amount_decimals: int | None = None,
    channel_pool: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Sinh dữ liệu cho nhiều đặc tả và ghép lại, sắp theo thời gian."""
    frames = [
        simulate_spec(spec, background_accounts, time_origin, horizon_seconds, seed, source,
                      amount_pool=amount_pool, amount_decimals=amount_decimals,
                      channel_pool=channel_pool)
        for spec in specs
    ]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=[
            "account_id", "counterparty_id", "timestamp", "amount", "channel",
            "device_id", "ip_country", "label", "source", "fraud_scenario_type",
            "scenario_code", "spec_id",
        ])
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values("timestamp").reset_index(drop=True)
