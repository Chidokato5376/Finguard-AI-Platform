"""Tiền xử lý: chuẩn hóa, chia tập theo thời gian (time-based split).

QUAN TRỌNG: KHÔNG dùng random split cho bài toán chuỗi hành vi — sẽ gây
temporal leakage (mô hình "nhìn thấy tương lai" của chính khách hàng
đang được dự đoán). Xem docs/architecture.md và Mục 3.4 đặc tả gốc.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SplitConfig:
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    def __post_init__(self) -> None:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Tỉ lệ split phải tổng bằng 1.0, hiện tại = {total}")


def time_based_split(
    df: pd.DataFrame, time_col: str = "step", config: SplitConfig | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chia train/val/test theo mốc thời gian, không xáo trộn ngẫu nhiên.

    Args:
        df: DataFrame đã ở Unified Schema, có cột thời gian.
        time_col: Tên cột thời gian dùng để sắp xếp (vd. "step" cho PaySim).
        config: Tỉ lệ chia — mặc định 70/15/15.

    Returns:
        (train, val, test) — mỗi tập chứa các giao dịch với timestamp không
        chồng lấn, đảm bảo test luôn ở tương lai so với train.
    """
    config = config or SplitConfig()
    if time_col not in df.columns:
        raise KeyError(f"Cột thời gian '{time_col}' không tồn tại trong DataFrame")

    df_sorted = df.sort_values(time_col, kind="stable").reset_index(drop=True)
    n = len(df_sorted)
    train_end = int(n * config.train_ratio)
    val_end = train_end + int(n * config.val_ratio)

    train = df_sorted.iloc[:train_end]
    val = df_sorted.iloc[train_end:val_end]
    test = df_sorted.iloc[val_end:]
    return train, val, test


def normalize_amount(df: pd.DataFrame, method: str = "log1p") -> pd.DataFrame:
    """Chuẩn hóa cột amount — mặc định log1p do phân phối lệch mạnh."""
    df = df.copy()
    if method == "log1p":
        import numpy as np

        df["amount_norm"] = np.log1p(df["amount"].astype(float))
    else:
        raise ValueError(f"Phương pháp chuẩn hóa không hỗ trợ: {method}")
    return df


def encode_categoricals(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    """One-hot / label-encode các cột phân loại (channel) trước khi vào GRU/Graph Encoder.

    Trả về cả mapping để có thể áp dụng lại y hệt cho tập val/test (tránh
    encoding lệch giữa các tập do chỉ fit trên train).
    """
    df = df.copy()
    mappings: dict[str, dict[str, int]] = {}
    if "channel" in df.columns:
        categories = sorted(df["channel"].dropna().unique().tolist())
        mapping = {cat: i for i, cat in enumerate(categories)}
        df["channel_encoded"] = df["channel"].map(mapping).fillna(-1).astype(int)
        mappings["channel"] = mapping
    return df, mappings


def apply_categorical_mapping(df: pd.DataFrame, mappings: dict[str, dict[str, int]]) -> pd.DataFrame:
    """Áp dụng lại mapping đã fit trên train cho val/test — KHÔNG fit lại,
    để tránh leakage thông tin phân phối của val/test vào bước chuẩn hóa.
    """
    df = df.copy()
    for col, mapping in mappings.items():
        df[f"{col}_encoded"] = df[col].map(mapping).fillna(-1).astype(int)
    return df


def build_account_id_index(df: pd.DataFrame) -> dict[str, int]:
    """Đánh số nguyên liên tục cho account_id (bao gồm cả counterparty_id)
    — cần thiết để dựng edge_index cho Graph Encoder (torch_geometric).
    """
    accounts = pd.unique(
        pd.concat([df["account_id"], df["counterparty_id"]], ignore_index=True)
    )
    return {acc: i for i, acc in enumerate(accounts)}


# Giờ trong ngày được giữ lại khi chấm điểm PaySim. Đo trên lát 3 triệu dòng
# (2026-10-17): giờ 4,5,6,7,8 có tỷ lệ gian lận đúng bằng 100% (mọi giao dịch
# trong các giờ đó đều là gian lận), giờ 1-8 chỉ chiếm 0,6% lưu lượng nhưng
# chứa 31,3% nhãn dương. Hệ quả: một BẢNG TRA 24 DÒNG theo giờ đạt AUPRC
# 0,2949 trên tập test, gấp đôi mô hình DRRE tốt nhất (0,1513). Sau khi bỏ
# giờ 0-8, baseline theo giờ rơi xuống 0,0026 (x1,8 ngẫu nhiên) — lỗi đã hết.
PAYSIM_CLEAN_HOURS = (9, 23)

# Gian lận PaySim CHỈ xuất hiện ở hai loại giao dịch này; cash_in, payment và
# debit có đúng 0 nhãn dương trên toàn bộ 6,36 triệu dòng. Chấm điểm trên cả
# năm loại tức là cho mô hình một phần đáp án miễn phí.
PAYSIM_SCORED_CHANNELS = ("transfer", "cash_out")

# Hai hình thức thanh toán của AMLWorld có ĐÚNG 0 nhãn dương trên toàn bộ
# 5.078.345 dòng HI-Small (đo 2026-10-19): `reinvestment` 481.056 dòng và
# `wire` 171.855 dòng — cộng lại 652.911 dòng, tức 12,9% dữ liệu. Đây chính
# xác là cùng lỗi với PaySim (cash_in/payment/debit cũng có đúng 0 nhãn
# dương). Chấm điểm trên cả 7 hình thức tức là cho mô hình một phần đáp án
# miễn phí: chỉ cần loại hai hạng mục này là đã bỏ được 12,9% số âm.
#
# Phân bố nhãn dương còn lại: ach 4.483 (86,6% nhãn dương trong 11,8% số
# dòng), cheque 324, credit card 206, cash 108, bitcoin 56. `ach` được GIỮ
# trong vũ trụ — ngoài thực tế ACH đúng là kênh rửa tiền phổ biến, và thu hẹp
# thêm nữa là uốn giao thức theo bộ dữ liệu. Đổi lại, `scripts/evaluate.py`
# bắt buộc đo mốc "bảng tra theo hình thức thanh toán" để biết riêng cột này
# mua được bao nhiêu.
AMLWORLD_UNSCORED_CHANNELS = ("wire", "reinvestment")


def add_eval_universe(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Thêm cột `eval_universe` — các dòng ĐƯỢC chấm điểm và tính chỉ số.

    QUAN TRỌNG: đây là mặt nạ, KHÔNG phải phép lọc. Mọi dòng vẫn được giữ
    trong tập dữ liệu, vì:
      - lịch sử chi tiêu của một tài khoản gồm cả giao dịch payment/cash_in;
      - các cạnh của đồ thị tài khoản cũng nằm ở những giao dịch đó.
    Xóa chúng đi là vừa chữa được lỗi dữ liệu vừa phá luôn chính thứ mà
    Behavior Memory và Graph Encoder cần. Vì vậy: giữ dòng, mặt nạ điểm.

    Mỗi nguồn một luật riêng, và luật nào cũng phải suy ra được từ một phép
    đo trên dữ liệu, không phải từ cảm tính:
      - paysim: giờ 9-23 ∧ loại ∈ {transfer, cash_out};
      - amlworld / transxion: hình thức thanh toán ∉ {wire, reinvestment};
      - nguồn khác (ieee_cis, synthetic_vn): True cho mọi dòng, để code phía
        sau không phải phân biệt nguồn.
    """
    df = df.copy()

    if source == "paysim":
        from src.data.loaders import PAYSIM_HOURS_PER_DAY

        hour = (df["step"] % PAYSIM_HOURS_PER_DAY).astype(int)
        lo, hi = PAYSIM_CLEAN_HOURS
        df["eval_universe"] = (
            hour.between(lo, hi) & df["channel"].isin(PAYSIM_SCORED_CHANNELS)
        )
        return df

    if source in ("amlworld", "transxion"):
        df["eval_universe"] = ~df["channel"].isin(AMLWORLD_UNSCORED_CHANNELS)
        return df

    df["eval_universe"] = True
    return df


def run_pipeline(
    source: str,
    raw_path: str,
    output_dir: str,
    split_config: SplitConfig | None = None,
    identity_path: str | None = None,
    min_daily_volume: int = 0,
    patterns_path: str | None = None,
) -> None:
    """Pipeline đầy đủ: load -> chuẩn hóa -> encode -> time-based split -> ghi ra parquet.

    Args:
        source: "paysim" | "ieee_cis" | "synthetic_vn".
        raw_path: đường dẫn file CSV gốc trong data/raw/.
        output_dir: thư mục ghi kết quả (thường là data/processed/).
        split_config: tỉ lệ train/val/test — mặc định 70/15/15.
        identity_path: CHỈ dùng khi source="ieee_cis" — đường dẫn tùy chọn
            tới train_identity.csv để join lấy device_id thật (xem
            src/data/loaders.py::load_ieee_cis).
        min_daily_volume: CHỈ dùng khi source="paysim" — loại các "ngày hỏng"
            có lưu lượng dưới ngưỡng (xem loaders.PAYSIM_MIN_DAILY_VOLUME).
            Các ngày bị loại KHÔNG bị xóa mất: chúng được ghi riêng ra
            `{source}_drift.parquet` để dùng làm tập kiểm thử dịch chuyển
            phân phối. 0 = không lọc.
        patterns_path: CHỈ dùng khi source="amlworld"/"transxion" — đường dẫn
            tới `{bộ}_Patterns.txt` để gắn thêm cột `laundering_pattern`
            (8 kiểu hình đồ thị). Cột này là NHÃN, chỉ dùng để đánh giá theo
            từng kiểu hình, KHÔNG được đưa vào đặc trưng đầu vào.
    """
    import logging
    from pathlib import Path

    from src.data.features import add_causal_features
    from src.data.graph_features import add_graph_features
    from src.data.loaders import (
        attach_laundering_pattern, healthy_day_mask, load_amlworld, load_ieee_cis,
        load_paysim, load_synthetic_vn, log_day_filter, parse_amlworld_patterns,
    )

    logger = logging.getLogger(__name__)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    df_drift: pd.DataFrame | None = None

    logger.info("Đang đọc dữ liệu nguồn '%s' từ %s", source, raw_path)
    if source == "paysim":
        df = load_paysim(raw_path)
        time_col = "step"
    elif source in ("amlworld", "transxion"):
        df = load_amlworld(raw_path, source_name=source)
        time_col = "timestamp"
        if patterns_path:
            df = attach_laundering_pattern(df, parse_amlworld_patterns(patterns_path))
    elif source == "ieee_cis":
        df = load_ieee_cis(raw_path, identity_path=identity_path)
        time_col = "step"
        logger.warning(
            "Nguồn 'ieee_cis' không có counterparty_id thật — account_index sinh "
            "ra sẽ KHÔNG dùng được cho Graph Encoder (Mục 4.3). Chỉ dùng nguồn "
            "này cho Behavior Memory và fine-tune bán giám sát (L_BCE)."
        )
    elif source == "synthetic_vn":
        df = load_synthetic_vn(raw_path)
        time_col = "timestamp"  # synthetic_vn dùng datetime thật, không có cột "step"
    else:
        raise ValueError(f"Nguồn không hỗ trợ: {source}")

    logger.info("Đã tải %d dòng. Chuẩn hóa amount và tính đặc trưng nhân quả...", len(df))
    df = normalize_amount(df, method="log1p")

    # Đặc trưng nhân quả tính trên TOÀN BỘ dòng dữ liệu, TRƯỚC khi chia tập và
    # trước khi tách ngày hỏng. Hai lý do:
    #   - một giao dịch ở tập test được phép biết lịch sử của chính tài khoản
    #     đó trong giai đoạn train (đúng như hệ thống thật); vì chỉ nhìn về
    #     quá khứ nên không rò rỉ;
    #   - nếu tính riêng từng tập thì lịch sử bị cắt vụn ở ranh giới tập và
    #     mọi giao dịch đầu tập đều bị coi như của tài khoản mới.
    df = add_causal_features(df, source)
    # Đặc trưng dòng tiền hai chiều — PHẢI chạy sau add_causal_features vì nó
    # cần khung ĐÃ sắp theo thời gian (chỉ số dòng chính là trục thời gian),
    # và trước khi chia tập, vì lịch sử phải liên tục trên cả tuyến.
    df = add_graph_features(df, source)
    df = add_eval_universe(df, source)

    # Lọc "ngày hỏng" — áp dụng cho mọi nguồn có mốc thời gian, vì cả PaySim
    # và AMLWorld đều có hiện tượng máy sinh gian lận vẫn chạy khi lưu lượng
    # hợp lệ đã tắt. Các ngày bị loại được ghi riêng ra {source}_drift.parquet
    # thay vì bị xóa, để còn dùng làm tập kiểm thử dịch chuyển phân phối.
    if min_daily_volume > 0:
        keep = healthy_day_mask(df, min_daily_volume)
        log_day_filter(df, keep, min_daily_volume, source)
        df_drift = df[~keep].reset_index(drop=True)
        df = df[keep].reset_index(drop=True)

    logger.info("Chia train/val/test theo thời gian (time-based split)...")
    train, val, test = time_based_split(df, time_col=time_col, config=split_config)

    # Fit encoding CHỈ trên train, áp dụng lại cho val/test — tránh leakage
    train, mappings = encode_categoricals(train)
    val = apply_categorical_mapping(val, mappings)
    test = apply_categorical_mapping(test, mappings)
    if df_drift is not None and not df_drift.empty:
        df_drift = apply_categorical_mapping(df_drift, mappings)

    account_index = build_account_id_index(df)
    logger.info("Số tài khoản duy nhất (node cho Graph Encoder): %d", len(account_index))

    outputs = [("train", train), ("val", val), ("test", test)]
    if df_drift is not None and not df_drift.empty:
        outputs.append(("drift", df_drift))

    clean_rates: dict[str, float] = {}
    for split_name, split_df in outputs:
        out_file = output_path / f"{source}_{split_name}.parquet"
        split_df.to_parquet(out_file, index=False)
        has_label = "label" in split_df.columns
        fraud_rate = split_df["label"].mean() if has_label else float("nan")
        universe = split_df["eval_universe"].to_numpy()
        n_clean = int(universe.sum())
        pos_clean = int(split_df.loc[universe, "label"].sum()) if has_label else 0
        rate_clean = pos_clean / n_clean if n_clean else float("nan")
        if split_name != "drift":
            clean_rates[split_name] = rate_clean
        logger.info(
            "Ghi %-5s: %8d dòng (nhãn dương %5d, %.4f%%) | vũ trụ chấm điểm %8d dòng "
            "(nhãn dương %5d, tỷ lệ nền %.5f) -> %s",
            split_name, len(split_df), int(split_df["label"].sum()) if has_label else 0,
            fraud_rate * 100, n_clean, pos_clean, rate_clean, out_file,
        )

    # Tỷ lệ nền của ba tập PHẢI gần nhau, nếu không thì AUPRC giữa các tập
    # không so sánh được với nhau (AUPRC phụ thuộc trực tiếp vào tỷ lệ nền).
    rates = [r for r in clean_rates.values() if r == r and r > 0]
    if len(rates) == 3 and max(rates) / min(rates) > 2.0:
        logger.warning(
            "Tỷ lệ nền của train/val/test lệch nhau %.1f lần (%s) — AUPRC giữa các tập "
            "KHÔNG so sánh trực tiếp được. Kiểm tra lại việc lọc ngày hoặc khoảng thời "
            "gian của dữ liệu trước khi báo cáo bất kỳ con số nào.",
            max(rates) / min(rates),
            ", ".join(f"{k}={v:.5f}" for k, v in clean_rates.items()),
        )

    import json

    with open(output_path / f"{source}_account_index.json", "w", encoding="utf-8") as f:
        json.dump(account_index, f, ensure_ascii=False)
    with open(output_path / f"{source}_categorical_mappings.json", "w", encoding="utf-8") as f:
        json.dump(mappings, f, ensure_ascii=False)

    logger.info("Hoàn tất tiền xử lý cho nguồn '%s'.", source)


def main() -> None:
    """CLI entry point.

    Ví dụ:
        python -m src.data.preprocessing --source paysim \\
            --raw-path data/raw/paysim.csv --output data/processed
    """
    import argparse
    import logging

    from src.data.loaders import PAYSIM_MIN_DAILY_VOLUME

    parser = argparse.ArgumentParser(description="Tiền xử lý dữ liệu FinGuard AI")
    parser.add_argument(
        "--source", required=True,
        choices=["paysim", "ieee_cis", "synthetic_vn", "amlworld", "transxion"])
    parser.add_argument(
        "--raw-path", required=True,
        help="Đường dẫn file CSV gốc, vd. data/raw/paysim.csv",
    )
    parser.add_argument(
        "--identity-path", default=None,
        help="CHỈ dùng với --source ieee_cis: đường dẫn tùy chọn tới train_identity.csv",
    )
    parser.add_argument("--output", default="data/processed", help="Thư mục ghi kết quả")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--min-daily-volume", type=int, default=None,
        help="CHỈ áp dụng cho --source paysim: loại các ngày có lưu lượng dưới ngưỡng "
             f"này và ghi riêng ra {{source}}_drift.parquet. Mặc định "
             f"{PAYSIM_MIN_DAILY_VOLUME} cho paysim, 0 (không lọc) cho nguồn khác. "
             "Truyền 0 để tắt hẳn và lấy lại hành vi cũ.",
    )
    parser.add_argument(
        "--patterns-path", default=None,
        help="CHỈ dùng với --source amlworld/transxion: đường dẫn tới "
             "{bộ}_Patterns.txt. Gắn thêm cột `laundering_pattern` (8 kiểu hình "
             "đồ thị) để đánh giá recall theo từng kiểu hình. Là NHÃN, không "
             "bao giờ được dùng làm đặc trưng đầu vào.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    min_daily_volume = args.min_daily_volume
    if min_daily_volume is None:
        min_daily_volume = PAYSIM_MIN_DAILY_VOLUME if args.source == "paysim" else 0
    run_pipeline(
        source=args.source, raw_path=args.raw_path, output_dir=args.output,
        identity_path=args.identity_path, min_daily_volume=min_daily_volume,
        patterns_path=args.patterns_path,
    )


if __name__ == "__main__":
    main()
