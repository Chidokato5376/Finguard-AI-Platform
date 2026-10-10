"""CLI dựng một phiên bản dataset tổng hợp (Mục 6).

Luồng: nạp nền → hiệu chỉnh → sinh đặc tả → validator → simulator →
injector → kiểm soát chất lượng → ghi phiên bản + manifest.

Ví dụ — đường cơ sở theo luật, cấu hình (ii), không cần internet:

    python -m src.augmentation.build --background synthetic_vn --origin rule

Ví dụ — đặc tả do LLM đề xuất, cấu hình (iii) (cần tự đặt khóa API):

    python -m src.augmentation.build --background synthetic_vn --origin llm \\
        --provider deepseek --model deepseek-chat --n-specs 12

Ví dụ — chạy đường LLM nhưng không có mạng (đặc tả sẽ được gán
origin="rule", KHÔNG dùng được cho cấu hình (iii)):

    python -m src.augmentation.build --background synthetic_vn --origin llm --offline
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from src.augmentation.calibration import calibrate_from_real
from src.augmentation.dataset import write_versioned
from src.augmentation.llm_adapter import LLMConfig, SpecGenerator, make_transport
from src.augmentation.mining import MiningConfig, apply_focus, mine_missed_specs
from src.augmentation.quality import (
    assess_quality,
    background_window,
    check_background_compatibility,
    inject,
)
from src.augmentation.rule_specs import build_rule_specs
from src.augmentation.simulator import (
    channel_override_needed,
    infer_amount_decimals,
    simulate_all,
)
from src.augmentation.spec_schema import ARCHETYPES

logger = logging.getLogger(__name__)

#: Đơn vị số tiền theo từng nguồn nền — ghi thẳng vào manifest để không ai
#: hiểu nhầm biên hiệu chỉnh là VND khi nền không phải VND.
UNIT_BY_SOURCE = {
    "synthetic_vn": "VND (dữ liệu mô phỏng bối cảnh Việt Nam)",
    "paysim": "đơn vị PaySim (không phải VND)",
    "ieee_cis": "USD xấp xỉ (IEEE-CIS TransactionAmt)",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dựng dataset tổng hợp có phiên bản")
    parser.add_argument("--background", default="synthetic_vn",
                        choices=["synthetic_vn", "paysim", "ieee_cis"],
                        help="Nguồn dữ liệu nền để hiệu chỉnh và chèn vào")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"],
                        help="Lát nền. Mặc định train — KHÔNG hiệu chỉnh trên test")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--origin", default="rule", choices=["rule", "llm"],
                        help="rule = cấu hình (ii); llm = cấu hình (iii)")
    parser.add_argument("--n-specs", type=int, default=12,
                        help="Số đặc tả yêu cầu LLM đề xuất (chỉ dùng với --origin llm)")
    parser.add_argument("--n-cases-per-spec", type=int, default=None,
                        help="Số ca mỗi đặc tả (chỉ dùng với --origin rule)")
    parser.add_argument("--archetypes", default=",".join(ARCHETYPES),
                        help="Danh sách kiểu hình, phân tách bằng dấu phẩy")
    parser.add_argument("--provider", default="deepseek")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key-env", default=None,
                        help="Tên biến môi trường chứa khóa. KHÔNG truyền khóa trực tiếp")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--offline", action="store_true",
                        help="Không gọi mạng; đặc tả sẽ bị gán origin='rule'")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-background-accounts", type=int, default=5000,
                        help="Số tài khoản nền lấy mẫu để tái sử dụng làm nguồn/nạn nhân")
    parser.add_argument("--out-root", default="data/synthetic")
    parser.add_argument("--write-injected", action="store_true",
                        help="Ghi thêm tập đã chèn (tiện kiểm tra, không bắt buộc)")
    parser.add_argument("--mine-from", default=None,
                        help="Parquet đã chấm điểm (risk_score,label,spec_id) để khai thác "
                             "đối kháng. BẬT MINING — không dùng cho cấu hình đối chứng")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    path = Path(args.processed_dir) / f"{args.background}_{args.split}.parquet"
    if not path.exists():
        parser.error(
            f"Không tìm thấy {path}. Chạy trước: python -m src.data.preprocessing "
            f"--source {args.background} --raw-path data/raw/<file>.csv --output {args.processed_dir}"
        )
    background = pd.read_parquet(path)
    logger.info("Nền: %s (%s) — %d dòng, %d nhãn dương", args.background, args.split,
                len(background), int(background["label"].astype(bool).sum()))

    # --- Calibration ---
    calibration = calibrate_from_real(background, unit=UNIT_BY_SOURCE[args.background],
                                      target_cases_per_spec=(3, 40))
    for param, src in calibration.sources.items():
        logger.info("  biên %-18s = %s  [%s]", param,
                    tuple(round(v, 2) for v in calibration.bounds[param]), src)

    # --- Khai thác đối kháng (nếu bật) ---
    mining_report = None
    if args.mine_from:
        scored = pd.read_parquet(args.mine_from)
        mining_report = mine_missed_specs(scored, [], MiningConfig(enabled=True))
        calibration = apply_focus(calibration, mining_report, MiningConfig(enabled=True))
        logger.warning("MINING ĐANG BẬT — dataset này phụ thuộc mô hình, không dùng làm đối chứng")

    archetypes = tuple(a.strip() for a in args.archetypes.split(",") if a.strip())
    invalid = set(archetypes) - set(ARCHETYPES)
    if invalid:
        parser.error(f"Kiểu hình không hợp lệ: {sorted(invalid)}. Hợp lệ: {list(ARCHETYPES)}")

    # --- Spec generator + validator ---
    llm_config = LLMConfig(
        provider=args.provider, model=args.model, base_url=args.base_url,
        api_key_env=args.api_key_env, temperature=args.temperature,
    )
    if args.origin == "rule":
        specs, rejections = build_rule_specs(
            calibration, archetypes=archetypes, n_cases_per_spec=args.n_cases_per_spec)
        llm_info = {"provider": "rule", "note": "cấu hình (ii) — không gọi LLM"}
        logger.info("Đặc tả theo luật: %d hợp lệ, %d bị loại", len(specs), len(rejections))
    else:
        transport = make_transport(llm_config, offline=args.offline, seed=args.seed)
        generator = SpecGenerator(calibration, transport=transport, config=llm_config)
        result = generator.propose(args.n_specs, background, archetypes=archetypes)
        specs, rejections = result.specs, result.rejections
        llm_info = {
            **(llm_config.to_json() if result.origin == "llm" else {"provider": "offline"}),
            "transport": result.transport_name,
            "origin_assigned": result.origin,
            "n_requested": result.n_requested,
            "rejection_rate": result.rejection_rate,
            "rejection_histogram": result.rejection_histogram(),
        }
        logger.info("Đặc tả từ %s: %d hợp lệ, %d bị loại (tỷ lệ loại %.0f%%), origin=%s",
                    result.transport_name, len(specs), len(rejections),
                    100 * result.rejection_rate, result.origin)
        for rej in rejections[:5]:
            logger.info("  bị loại [%s]: %s", rej.reason_code, rej.detail)

    if not specs:
        logger.error("Không có đặc tả hợp lệ nào — dừng. Xem lý do bị loại ở trên.")
        return 1

    # --- Tương thích nền × kiểu hình ---
    check_background_compatibility(background, specs)

    # --- Simulator ---
    accounts = background["account_id"].dropna().unique()
    if len(accounts) > args.max_background_accounts:
        accounts = pd.Series(accounts).sample(
            args.max_background_accounts, random_state=args.seed).to_numpy()
    origin_time, horizon = background_window(background)
    # Phân phối thực nghiệm của số tiền gian lận THẬT — dùng để simulator bốc
    # số tiền thay vì phân phối đều (xem simulator._draw_amount). Đây là dữ
    # liệu nội bộ của simulator, không liên quan tới đường gọi LLM.
    amount_pool = background.loc[
        background["label"].astype(bool), "amount"].astype(float).to_numpy()
    decimals = infer_amount_decimals(background["amount"].astype(float).to_numpy())
    logger.info("Độ lượng tử hóa số tiền của nền: %s",
                f"{decimals} chữ số thập phân" if decimals is not None else "độ chính xác đầy đủ")

    channel_pool = background.loc[background["label"].astype(bool), "channel"]
    channel_notes: list[str] = []
    if channel_override_needed([s.channel for s in specs], channel_pool):
        channel_notes.append(
            f"Nền '{args.background}' dùng từ vựng cột channel khác với từ vựng đóng của đặc tả "
            f"(nền có: {sorted(set(channel_pool.dropna()))[:6]}). Kênh trong đặc tả đã bị GHI ĐÈ "
            "bằng cách bốc theo phân phối thực nghiệm của gian lận thật, để kiểm định phân biệt "
            "đo được dấu vết không chủ ý thay vì đo lại sự khác từ vựng đã biết. Việc cột channel "
            "chưa thống nhất giữa các nguồn là một vấn đề schema cần xử lý riêng."
        )
        logger.warning("  %s", channel_notes[-1])

    rows = simulate_all(specs, list(accounts), origin_time, horizon, seed=args.seed,
                        source=args.background, amount_pool=amount_pool,
                        amount_decimals=decimals, channel_pool=channel_pool)
    logger.info("Simulator: %d dòng tổng hợp (%d nhãn dương) từ %d đặc tả",
                len(rows), int(rows["label"].sum()), len(specs))

    # --- Injector + kiểm soát chất lượng ---
    injected = inject(background, rows)
    quality = assess_quality(injected, seed=args.seed)
    status_label = {"dat": "ĐẠT", "khong_dat": "KHÔNG ĐẠT",
                    "chua_ket_luan": "CHƯA KẾT LUẬN (thiếu mẫu)"}[quality.status]
    logger.info("Kiểm định phân biệt (tổng hợp vs gian lận thật) AUC = %.3f — %s",
                quality.adversarial_auc_vs_real_fraud, status_label)
    for warning in quality.warnings:
        logger.warning("  %s", warning)

    # --- Ghi phiên bản ---
    notes = list(calibration.notes) + channel_notes
    if mining_report is not None:
        notes.extend(mining_report.notes)
    if args.origin == "llm" and args.offline:
        notes.append(
            "Chạy với --offline: đặc tả KHÔNG do LLM sinh, đã gán origin='rule'. "
            "Không dùng phiên bản này cho cấu hình (iii)."
        )
    out_dir = write_versioned(
        rows=rows, specs=specs, rejections=rejections, calibration=calibration,
        quality=quality, llm_info=llm_info, seed=args.seed,
        background_source=f"{args.background}_{args.split}", root=args.out_root, notes=notes,
    )
    if args.write_injected:
        injected.to_parquet(out_dir / "injected.parquet", index=False)
    logger.info("Đã ghi phiên bản: %s", out_dir)
    print(out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
