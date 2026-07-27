"""Synthetic VN Data Generator (nguồn `synthetic_vn`, Mục 3.3).

Sinh dữ liệu tổng hợp mô phỏng bối cảnh giao dịch số Việt Nam, dùng CHỈ
để kiểm thử concept drift (Mục 11 — đo suy giảm Recall khi xuất hiện
fraud pattern mới không có trong tập huấn luyện). KHÔNG dùng nguồn này
để huấn luyện chính (xem data/README.md và README.md §9 Giới hạn).

=== TÍNH TRUNG THỰC DỮ LIỆU — ĐỌC TRƯỚC KHI DÙNG ===

Phân phối số tiền giao dịch được neo theo số liệu CÔNG KHAI của NHNN/Napas
(có dẫn nguồn cụ thể bên dưới), để quy mô giao dịch có tính hiện thực thay
vì bịa hoàn toàn. Tuy nhiên:

  - avg_transaction_vnd (mặc định ~6,1 triệu VND) được suy ra từ số liệu
    Napas Quý I/2025: 2.453.426.745 giao dịch, tổng giá trị 14.972.645 tỷ
    đồng (Nguồn: NHNN, "Công bố Chương trình Ngày không tiền mặt năm 2025",
    sbv.gov.vn/en/w/sbv637581). Đây là giao dịch trung bình của TOÀN BỘ hệ
    thống chuyển mạch (bao gồm cả giao dịch giá trị lớn giữa doanh nghiệp),
    không phải giao dịch bán lẻ cá nhân thuần túy — dùng làm điểm neo tham
    khảo, KHÔNG phải con số chính xác cho phân khúc khách hàng cá nhân.

  - adult_banked_ratio = 0.889 lấy từ số liệu NHNN cuối năm 2025 (gần 89%
    người trưởng thành có tài khoản ngân hàng).

  - fraud_rate_assumed: KHÔNG có nguồn công khai nào công bố tỷ lệ gian lận
    trên tổng số giao dịch ở Việt Nam tại mức chi tiết cần cho mô phỏng.
    Hệ thống SIMO của NHNN (tính đến 14/01/2026) ghi nhận quy mô TUYỆT ĐỐI
    của vấn đề (592.000 tài khoản/ví bị gắn cờ nghi ngờ từ 122/147 tổ chức
    báo cáo, ~2,6 triệu lượt cảnh báo, ~831.000 giao dịch bị tạm dừng/hủy
    trị giá ~3,06 nghìn tỷ đồng) — nhưng đây KHÔNG suy ra được tỷ lệ
    gian lận/giao dịch vì không có mẫu số (tổng số giao dịch được rà soát).
    Do đó fraud_rate_assumed là THAM SỐ GIẢ ĐỊNH do người dùng tự đặt.
    **Khi báo cáo/thuyết trình, PHẢI nêu rõ đây là giả định, không phải số
    đo thực nghiệm hay số liệu NHNN công bố.**

Nếu bạn có quyền truy cập báo cáo nội bộ hoặc số liệu ngành cụ thể hơn,
thay giá trị mặc định trong VNCalibrationConfig bằng số liệu đó và ghi rõ
nguồn mới trong docstring này.
"""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from src.data.schema import Channel, DataSource


@dataclass(frozen=True)
class VNCalibrationConfig:
    """Tham số hiệu chỉnh — xem ghi chú nguồn ở đầu file trước khi chỉnh sửa."""

    # Neo theo Napas Q1/2025 (xem docstring) — đơn vị VND
    avg_transaction_vnd: float = 6_100_000.0
    # Độ lệch chuẩn (log-scale) cho phân phối log-normal của amount
    transaction_amount_sigma: float = 1.2
    # NHNN, cuối 2025 — tỷ lệ người trưởng thành có tài khoản ngân hàng
    adult_banked_ratio: float = 0.889

    # ⚠ THAM SỐ GIẢ ĐỊNH — không có nguồn công khai, PHẢI nêu rõ khi báo cáo
    fraud_rate_assumed: float = 0.01

    num_accounts: int = 500
    num_normal_transactions: int = 5000
    time_horizon_days: int = 90

    # Tỷ trọng các kịch bản gian lận trong tổng số giao dịch gian lận sinh ra
    money_mule_weight: float = 0.4
    account_takeover_weight: float = 0.35
    investment_scam_weight: float = 0.25

    random_seed: int = 42


def _account_id(rng: random.Random) -> str:
    return f"VN{rng.randint(100000, 999999)}"


def generate_normal_transactions(config: VNCalibrationConfig) -> pd.DataFrame:
    """Sinh giao dịch bình thường — nền tảng cho behavior pattern hợp lệ.

    Số tiền theo phân phối log-normal, neo trung vị quanh avg_transaction_vnd
    (xem cảnh báo nguồn ở đầu file — đây là điểm neo tham khảo, không phải
    số đo chính xác cho phân khúc bán lẻ cá nhân).
    """
    rng = random.Random(config.random_seed)
    np_rng = np.random.default_rng(config.random_seed)

    accounts = [_account_id(rng) for _ in range(config.num_accounts)]
    start_time = datetime(2026, 1, 1)

    mu = np.log(config.avg_transaction_vnd) - (config.transaction_amount_sigma ** 2) / 2
    amounts = np_rng.lognormal(mean=mu, sigma=config.transaction_amount_sigma,
                                size=config.num_normal_transactions)

    rows = []
    for i in range(config.num_normal_transactions):
        sender = rng.choice(accounts)
        receiver = rng.choice([a for a in accounts if a != sender])
        offset_seconds = rng.uniform(0, config.time_horizon_days * 86400)
        rows.append({
            "account_id": sender,
            "counterparty_id": receiver,
            "timestamp": start_time + timedelta(seconds=offset_seconds),
            "amount": float(amounts[i]),
            "channel": rng.choice([Channel.DOMESTIC.value, Channel.E_WALLET.value]),
            "device_id": f"dev_{sender}",         # thiết bị ổn định = hành vi bình thường
            "ip_country": "VN",
            "label": False,
            "source": DataSource.SYNTHETIC_VN.value,
            "fraud_scenario_type": None,
        })

    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    return df


def generate_money_mule_scenario(
    config: VNCalibrationConfig, n_chains: int, chain_length: int = 4
) -> pd.DataFrame:
    """Money Mule: tiền chảy qua một chuỗi tài khoản trung gian rồi rút ra,
    thường trong thời gian ngắn (vài phút-giờ) — mẫu hình quan trọng để
    kiểm thử Graph-based Relationship Encoder (Mục 4.3).

    Mỗi chain: acc_0 -> acc_1 -> acc_2 -> ... -> acc_{chain_length-1}
    Tài khoản trung gian (acc_1..acc_{n-2}) thường là tài khoản MỚI, ít
    lịch sử — kiểm thử luôn tình huống cold-start (Mục 11).
    """
    rng = random.Random(config.random_seed + 1)
    # QUAN TRỌNG: dùng CÙNG khung thời gian với normal transactions
    # (start=2026-01-01, time_horizon_days ngày) — nếu đặt một khối thời
    # gian tách biệt, time-based split (preprocessing.py) sẽ dồn toàn bộ
    # mẫu gian lận vào tập test, khiến train/val có 0% nhãn dương và mô
    # hình không học được gì. Đây là lỗi đã phát hiện qua chạy thử thực tế.
    start_time = datetime(2026, 1, 1)
    rows = []

    for chain_idx in range(n_chains):
        chain_accounts = [f"MULE_{chain_idx}_{i}" for i in range(chain_length)]
        base_amount = rng.uniform(20_000_000, 200_000_000)  # giá trị lớn bất thường
        chain_start = start_time + timedelta(days=rng.uniform(0, config.time_horizon_days))

        for hop in range(chain_length - 1):
            # Mỗi hop giữ lại ~95-99% số tiền (phí/chiết khấu của mule)
            amount = base_amount * (0.95 ** hop)
            rows.append({
                "account_id": chain_accounts[hop],
                "counterparty_id": chain_accounts[hop + 1],
                "timestamp": chain_start + timedelta(minutes=hop * rng.uniform(3, 15)),
                "amount": amount,
                "channel": Channel.DOMESTIC.value,
                "device_id": f"dev_mule_{chain_idx}_{hop}",
                "ip_country": "VN",
                "label": True,
                "source": DataSource.SYNTHETIC_VN.value,
                "fraud_scenario_type": "money_mule_chain",
            })

    return pd.DataFrame(rows)


def generate_account_takeover_scenario(config: VNCalibrationConfig, n_cases: int) -> pd.DataFrame:
    """Chiếm quyền tài khoản Mobile Banking (SIM swap / OTP phishing):
    một tài khoản có lịch sử giao dịch nhỏ, đều đặn, đột ngột xuất hiện
    một giao dịch giá trị lớn TỪ THIẾT BỊ/IP KHÁC LẦN ĐẦU XUẤT HIỆN.

    Đây chính là kịch bản mà Context-aware Attention (Mục 4.2) cần phát
    hiện qua delta_t lớn — giao dịch không khớp bất kỳ slot bộ nhớ nào.
    """
    rng = random.Random(config.random_seed + 2)
    # Cùng khung thời gian với normal transactions — xem ghi chú trong
    # generate_money_mule_scenario ở trên (tránh dồn nhãn dương vào test).
    start_time = datetime(2026, 1, 1)
    rows = []

    for case_idx in range(n_cases):
        account = f"ATO_{case_idx}"
        normal_device = f"dev_{account}_home"
        attacker_device = f"dev_unknown_{uuid.uuid4().hex[:8]}"
        # Chừa đủ khoảng đầu cho n_history giao dịch lịch sử đứng trước nó
        case_start = start_time + timedelta(
            days=rng.uniform(10, max(10, config.time_horizon_days))
        )

        # 5-10 giao dịch nhỏ, bình thường để xây lịch sử hành vi trước khi bị chiếm quyền
        n_history = rng.randint(5, 10)
        typical_amount = rng.uniform(200_000, 3_000_000)
        for h in range(n_history):
            rows.append({
                "account_id": account,
                "counterparty_id": _account_id(rng),
                "timestamp": case_start - timedelta(days=(n_history - h) * rng.uniform(1, 5)),
                "amount": typical_amount * rng.uniform(0.7, 1.3),
                "channel": Channel.E_WALLET.value,
                "device_id": normal_device,
                "ip_country": "VN",
                "label": False,
                "source": DataSource.SYNTHETIC_VN.value,
                "fraud_scenario_type": None,
            })

        # Giao dịch chiếm quyền: giá trị lớn gấp nhiều lần lịch sử, thiết bị/IP lạ
        rows.append({
            "account_id": account,
            "counterparty_id": f"NEWBEN_{uuid.uuid4().hex[:8]}",  # người nhận mới, chưa từng giao dịch
            "timestamp": case_start,
            "amount": typical_amount * rng.uniform(15, 40),
            "channel": Channel.E_WALLET.value,
            "device_id": attacker_device,
            "ip_country": rng.choice(["KH", "PH", "unknown"]),  # IP nước ngoài bất thường
            "label": True,
            "source": DataSource.SYNTHETIC_VN.value,
            "fraud_scenario_type": "account_takeover",
        })

    return pd.DataFrame(rows)


def generate_investment_scam_scenario(config: VNCalibrationConfig, n_cases: int) -> pd.DataFrame:
    """Lừa đảo đầu tư/tình cảm: nạn nhân TỰ chuyển một khoản lớn tới một
    tài khoản thụ hưởng hoàn toàn mới (chưa từng xuất hiện trong đồ thị
    giao dịch trước đó) — không có dấu hiệu bất thường về thiết bị/IP vì
    chính chủ tài khoản thực hiện giao dịch.

    Đây là kịch bản KHÓ nhất cho hệ thống vì không có tín hiệu thiết bị/IP
    bất thường — chỉ có thể phát hiện qua Graph Encoder (tài khoản thụ
    hưởng mới, không có lịch sử) kết hợp với biên độ giao dịch bất thường
    so với Behavior Memory của chính khách hàng.
    """
    rng = random.Random(config.random_seed + 3)
    # Cùng khung thời gian với normal transactions — xem ghi chú trong
    # generate_money_mule_scenario ở trên.
    start_time = datetime(2026, 1, 1)
    rows = []

    for case_idx in range(n_cases):
        account = f"SCAM_VICTIM_{case_idx}"
        case_start = start_time + timedelta(
            days=rng.uniform(10, max(10, config.time_horizon_days))
        )
        typical_amount = rng.uniform(500_000, 5_000_000)

        n_history = rng.randint(5, 10)
        for h in range(n_history):
            rows.append({
                "account_id": account,
                "counterparty_id": _account_id(rng),
                "timestamp": case_start - timedelta(days=(n_history - h) * rng.uniform(1, 7)),
                "amount": typical_amount * rng.uniform(0.7, 1.3),
                "channel": Channel.DOMESTIC.value,
                "device_id": f"dev_{account}",
                "ip_country": "VN",
                "label": False,
                "source": DataSource.SYNTHETIC_VN.value,
                "fraud_scenario_type": None,
            })

        rows.append({
            "account_id": account,
            "counterparty_id": f"NEWBEN_SCAM_{uuid.uuid4().hex[:8]}",
            "timestamp": case_start,
            "amount": typical_amount * rng.uniform(10, 50),
            "channel": Channel.DOMESTIC.value,
            "device_id": f"dev_{account}",  # CÙNG thiết bị — không có tín hiệu device bất thường
            "ip_country": "VN",
            "label": True,
            "source": DataSource.SYNTHETIC_VN.value,
            "fraud_scenario_type": "investment_romance_scam",
        })

    return pd.DataFrame(rows)


FRAUD_PATTERN_NAMES = ("money_mule_chain", "account_takeover", "investment_romance_scam")


def generate_synthetic_vn_dataset(
    config: VNCalibrationConfig | None = None,
    exclude_patterns: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Lắp ráp bộ dữ liệu synthetic VN đầy đủ: giao dịch bình thường +
    ba kịch bản gian lận, theo đúng tỷ trọng cấu hình.

    Args:
        config: tham số hiệu chỉnh.
        exclude_patterns: các fraud_scenario_type KHÔNG sinh, vd.
            ("account_takeover",). Dùng để tạo tập "known" cho kiểm thử
            concept drift (Mục 11 KPI) — huấn luyện trên tập không có
            pattern X, rồi đánh giá Recall trên tập có đầy đủ pattern X
            (xem generate_concept_drift_split bên dưới).
            Giá trị hợp lệ: xem FRAUD_PATTERN_NAMES.

    Trả về DataFrame theo Unified Schema (+ cột phụ fraud_scenario_type
    để phục vụ phân tích concept drift theo từng loại hình).
    """
    config = config or VNCalibrationConfig()
    invalid = set(exclude_patterns) - set(FRAUD_PATTERN_NAMES)
    if invalid:
        raise ValueError(
            f"exclude_patterns chứa giá trị không hợp lệ: {invalid}. "
            f"Giá trị hợp lệ: {FRAUD_PATTERN_NAMES}"
        )

    n_fraud_total = max(1, int(config.num_normal_transactions * config.fraud_rate_assumed
                                / (1 - config.fraud_rate_assumed)))
    n_mule_chains = max(1, int(n_fraud_total * config.money_mule_weight / 3))  # ~3 giao dịch/chain
    n_ato_cases = max(1, int(n_fraud_total * config.account_takeover_weight))
    n_scam_cases = max(1, int(n_fraud_total * config.investment_scam_weight))

    frames = [generate_normal_transactions(config)]
    if "money_mule_chain" not in exclude_patterns:
        frames.append(generate_money_mule_scenario(config, n_chains=n_mule_chains))
    if "account_takeover" not in exclude_patterns:
        frames.append(generate_account_takeover_scenario(config, n_cases=n_ato_cases))
    if "investment_romance_scam" not in exclude_patterns:
        frames.append(generate_investment_scam_scenario(config, n_cases=n_scam_cases))

    full_df = pd.concat(frames, ignore_index=True)
    full_df = full_df.sort_values("timestamp").reset_index(drop=True)
    full_df["is_synthetic_field"] = [("device_id", "ip_country")] * len(full_df)
    return full_df


def generate_concept_drift_split(
    config: VNCalibrationConfig | None = None,
    held_out_pattern: str = "account_takeover",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sinh cặp dữ liệu để kiểm thử concept drift trực tiếp với
    src.evaluation.metrics.compute_concept_drift_recall_drop.

    - df_known: KHÔNG chứa held_out_pattern — mô phỏng dữ liệu tại thời
      điểm huấn luyện, trước khi typology này xuất hiện.
    - df_drift: CÓ đầy đủ mọi pattern kể cả held_out_pattern — mô phỏng
      dữ liệu sau khi kẻ gian đổi chiến thuật hoặc mở rộng thủ đoạn.

    Cả hai tập dùng CHUNG seed/config nên các giao dịch bình thường và
    các pattern không bị loại trừ giống hệt nhau giữa hai tập — chênh
    lệch duy nhất là sự xuất hiện của held_out_pattern, giúp phép đo
    Recall drop phản ánh đúng tác động của riêng pattern đó, không lẫn
    nhiễu từ sự khác biệt ngẫu nhiên khác giữa hai lần sinh dữ liệu.

    Ví dụ dùng:
        df_known, df_drift = generate_concept_drift_split(held_out_pattern="account_takeover")
        # ... huấn luyện/đánh giá model trên df_known ...
        # ... áp dụng CÙNG model (không huấn luyện lại) lên df_drift ...
        drop = compute_concept_drift_recall_drop(
            y_true_before=..., y_score_before=...,   # từ df_known
            y_true_after=...,  y_score_after=...,     # từ df_drift
        )
    """
    if held_out_pattern not in FRAUD_PATTERN_NAMES:
        raise ValueError(
            f"held_out_pattern phải thuộc {FRAUD_PATTERN_NAMES}, "
            f"nhận được '{held_out_pattern}'"
        )
    config = config or VNCalibrationConfig()
    df_known = generate_synthetic_vn_dataset(config, exclude_patterns=(held_out_pattern,))
    df_drift = generate_synthetic_vn_dataset(config, exclude_patterns=())
    return df_known, df_drift


def main() -> None:
    """CLI: sinh dữ liệu và ghi ra data/raw/synthetic_vn.csv.

    Ví dụ:
        python -m src.data.synthetic_generator --output data/raw/synthetic_vn.csv
        python -m src.data.synthetic_generator --output-dir data/raw \\
            --concept-drift-pattern account_takeover
    """
    import argparse

    parser = argparse.ArgumentParser(description="Sinh dữ liệu synthetic VN cho FinGuard AI")
    parser.add_argument("--output", default="data/raw/synthetic_vn.csv",
                         help="Dùng khi KHÔNG sinh cặp concept drift (mặc định)")
    parser.add_argument("--num-normal", type=int, default=5000)
    parser.add_argument("--fraud-rate", type=float, default=0.01,
                         help="THAM SỐ GIẢ ĐỊNH — xem cảnh báo nguồn trong docstring module này")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--concept-drift-pattern", default=None, choices=list(FRAUD_PATTERN_NAMES),
        help="Nếu đặt, sinh CẶP file *_known.csv / *_drift.csv thay vì 1 file duy nhất "
             "(xem generate_concept_drift_split)",
    )
    parser.add_argument("--output-dir", default="data/raw",
                         help="Thư mục ghi khi dùng --concept-drift-pattern")
    args = parser.parse_args()

    config = VNCalibrationConfig(
        num_normal_transactions=args.num_normal,
        fraud_rate_assumed=args.fraud_rate,
        random_seed=args.seed,
    )

    from pathlib import Path

    if args.concept_drift_pattern:
        df_known, df_drift = generate_concept_drift_split(
            config, held_out_pattern=args.concept_drift_pattern
        )
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        known_path = out_dir / f"synthetic_vn_known_excl_{args.concept_drift_pattern}.csv"
        drift_path = out_dir / f"synthetic_vn_drift_incl_{args.concept_drift_pattern}.csv"
        df_known.to_csv(known_path, index=False)
        df_drift.to_csv(drift_path, index=False)

        print(f"Đã sinh cặp concept drift (held_out_pattern='{args.concept_drift_pattern}'):")
        print(f"  known: {len(df_known)} giao dịch -> {known_path}")
        print(f"  drift: {len(df_drift)} giao dịch -> {drift_path}")
        return

    df = generate_synthetic_vn_dataset(config)

    from pathlib import Path
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    print(f"Đã sinh {len(df)} giao dịch -> {args.output}")
    print(f"Tỷ lệ nhãn dương thực tế: {df['label'].mean():.4f} "
          f"(tham số giả định fraud_rate_assumed={args.fraud_rate} — xem cảnh báo trong docstring)")
    print(df["fraud_scenario_type"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
