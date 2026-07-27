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

    df_sorted = df.sort_values(time_col).reset_index(drop=True)
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


def run_pipeline(
    source: str,
    raw_path: str,
    output_dir: str,
    split_config: SplitConfig | None = None,
    identity_path: str | None = None,
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
    """
    import logging
    from pathlib import Path

    from src.data.loaders import load_ieee_cis, load_paysim, load_synthetic_vn

    logger = logging.getLogger(__name__)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Đang đọc dữ liệu nguồn '%s' từ %s", source, raw_path)
    if source == "paysim":
        df = load_paysim(raw_path)
        time_col = "step"
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

    logger.info("Đã tải %d dòng. Chuẩn hóa amount và encode categorical...", len(df))
    df = normalize_amount(df, method="log1p")

    logger.info("Chia train/val/test theo thời gian (time-based split)...")
    train, val, test = time_based_split(df, time_col=time_col, config=split_config)

    # Fit encoding CHỈ trên train, áp dụng lại cho val/test — tránh leakage
    train, mappings = encode_categoricals(train)
    val = apply_categorical_mapping(val, mappings)
    test = apply_categorical_mapping(test, mappings)

    account_index = build_account_id_index(df)
    logger.info("Số tài khoản duy nhất (node cho Graph Encoder): %d", len(account_index))

    for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
        out_file = output_path / f"{source}_{split_name}.parquet"
        split_df.to_parquet(out_file, index=False)
        fraud_rate = split_df["label"].mean() if "label" in split_df.columns else float("nan")
        logger.info(
            "Ghi %s: %d dòng -> %s (tỷ lệ nhãn dương: %.4f%%)",
            split_name, len(split_df), out_file, fraud_rate * 100,
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

    parser = argparse.ArgumentParser(description="Tiền xử lý dữ liệu FinGuard AI")
    parser.add_argument("--source", required=True, choices=["paysim", "ieee_cis", "synthetic_vn"])
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
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    run_pipeline(
        source=args.source, raw_path=args.raw_path, output_dir=args.output,
        identity_path=args.identity_path,
    )


if __name__ == "__main__":
    main()
