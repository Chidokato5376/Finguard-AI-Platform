"""Loader cho ba nguồn dữ liệu: PaySim, IEEE-CIS, Synthetic VN.

Mỗi loader trả về pandas.DataFrame đã ánh xạ sang Unified Schema
(src/data/schema.py), kèm cờ is_synthetic_field cho các trường không
có trong dữ liệu gốc.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Cột gốc thực tế của PaySim (Kaggle) — KHÔNG bao gồm device_id/ip_country.
PAYSIM_RAW_COLUMNS = [
    "step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
    "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud",
]


def load_paysim(path: str | Path, synthesize_device_ip: bool = False) -> pd.DataFrame:
    """Đọc PaySim và ánh xạ sang Unified Schema.

    Args:
        path: Đường dẫn tới file CSV PaySim gốc.
        synthesize_device_ip: Nếu True, sinh bổ sung device_id/ip_country
            (dùng cho demo/prototype). Bản ghi kết quả sẽ được đánh dấu
            is_synthetic_field=("device_id", "ip_country") — KHÔNG được
            trình bày như dữ liệu thật.

    Returns:
        DataFrame theo Unified Schema, cột `source` = "paysim".

    Raises:
        FileNotFoundError: nếu path không tồn tại.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {path}. Tải PaySim từ Kaggle và đặt vào data/raw/."
        )
    raw = pd.read_csv(path)
    missing = set(PAYSIM_RAW_COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f"PaySim file thiếu cột: {missing}")

    df = pd.DataFrame(
        {
            "account_id": raw["nameOrig"],
            "counterparty_id": raw["nameDest"],
            "step": raw["step"],
            "amount": raw["amount"],
            "channel": raw["type"].str.lower(),
            "label": raw["isFraud"].astype(bool),
            "source": "paysim",
        }
    )
    if synthesize_device_ip:
        logger.warning(
            "synthesize_device_ip=True: device_id/ip_country sẽ được SINH, "
            "không phải dữ liệu gốc. Xem data/README.md."
        )
        df["device_id"] = None  # TODO: gắn generator từ src/data/synthetic_generator.py
        df["ip_country"] = None
        df["is_synthetic_field"] = [("device_id", "ip_country")] * len(df)
    else:
        df["device_id"] = None
        df["ip_country"] = None
        df["is_synthetic_field"] = [()] * len(df)
    return df


IEEE_CIS_TRANSACTION_REQUIRED_COLUMNS = [
    "TransactionID", "isFraud", "TransactionDT", "TransactionAmt", "ProductCD",
    "card1", "card2", "card3", "card5", "addr1", "P_emaildomain",
]


def _build_pseudo_client_id(df: pd.DataFrame) -> pd.Series:
    """Tái tạo 'khách hàng' XẤP XỈ từ IEEE-CIS — bộ dữ liệu KHÔNG có
    customer ID thật, chỉ có thông tin thẻ/địa chỉ đã ẩn danh hóa một
    phần. Đây là heuristic phổ biến trong cộng đồng Kaggle (kết hợp
    card1/card2/card3/card5/addr1/P_emaildomain) để nhóm các giao dịch
    NHIỀU KHẢ NĂNG cùng một khách hàng — KHÔNG phải định danh chính xác.

    GIỚI HẠN ĐÃ BIẾT (nêu rõ, không giấu):
      - Có thể GỘP nhầm hai khách hàng khác nhau nếu trùng ngẫu nhiên tổ
        hợp card+addr+email (hiếm nhưng không loại trừ).
      - Có thể TÁCH nhầm một khách hàng thật thành nhiều "account_id" nếu
        họ dùng nhiều thẻ hoặc đổi địa chỉ/email giữa các giao dịch.
      - Behavior Memory huấn luyện trên nguồn này vì vậy học một xấp xỉ
        NHIỄU của hành vi cá nhân — chấp nhận được cho vai trò đã định vị
        của nguồn này (huấn luyện phụ + đầu dò bán giám sát L_BCE, xem
        Mục 3.3), KHÔNG chấp nhận được nếu dùng làm nguồn chính cho Graph
        Encoder.
    """
    key_cols = ["card1", "card2", "card3", "card5", "addr1", "P_emaildomain"]
    parts = [df[c].astype(str).fillna("NA") for c in key_cols]
    combined = parts[0]
    for p in parts[1:]:
        combined = combined.str.cat(p, sep="_")
    return "IEEECIS_" + combined


def load_ieee_cis(
    transaction_path: str | Path, identity_path: str | Path | None = None
) -> pd.DataFrame:
    """Đọc IEEE-CIS Fraud Detection và ánh xạ sang Unified Schema.

    Args:
        transaction_path: đường dẫn train_transaction.csv (bắt buộc).
        identity_path: đường dẫn train_identity.csv (tùy chọn) — nếu có,
            join theo TransactionID để lấy DeviceInfo làm device_id thật
            (chỉ phủ một phần giao dịch — identity table không đầy đủ).

    ⚠ GIỚI HẠN QUAN TRỌNG (xem thêm data/README.md, docs/architecture.md):
      - KHÔNG có counterparty_id thật — cột này luôn trả về None. TUYỆT
        ĐỐI không dùng nguồn này làm input cho Graph Encoder (Mục 4.3);
        preprocessing.py vẫn chạy được nhưng đồ thị sinh ra sẽ vô nghĩa.
      - account_id là PSEUDO-ID suy luận (xem _build_pseudo_client_id),
        không phải customer ID xác thực.
      - ip_country LUÔN là None. Các cột id_19/id_20 trong bảng identity
        đôi khi được cộng đồng Kaggle suy đoán liên quan tới định vị,
        nhưng Vesta (chủ dữ liệu) CHƯA BAO GIỜ công bố ý nghĩa chính thức
        của các cột id_*/V* đã ẩn danh hóa. Loader này KHÔNG tự gán ý
        nghĩa địa lý cho các cột đó — làm vậy là suy diễn không có căn cứ.
      - step = TransactionDT là timedelta tính bằng giây từ một mốc tham
        chiếu ẩn danh, KHÔNG phải timestamp thật — chỉ dùng để SẮP XẾP
        thứ tự thời gian (time-based split), không suy ra được ngày/giờ
        thực của giao dịch.

    Returns:
        DataFrame theo Unified Schema, cột `source` = "ieee_cis".

    Raises:
        FileNotFoundError: nếu transaction_path không tồn tại.
        ValueError: nếu thiếu cột bắt buộc trong file transaction.
    """
    transaction_path = Path(transaction_path)
    if not transaction_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {transaction_path}. Tải IEEE-CIS từ Kaggle competition "
            "(cần Accept Rules trước khi tải — xem README)."
        )
    raw = pd.read_csv(transaction_path)
    missing = set(IEEE_CIS_TRANSACTION_REQUIRED_COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f"IEEE-CIS train_transaction.csv thiếu cột: {missing}")

    df = pd.DataFrame(
        {
            "account_id": _build_pseudo_client_id(raw),
            "counterparty_id": None,  # KHÔNG có trong IEEE-CIS — xem cảnh báo docstring
            "step": raw["TransactionDT"],
            "amount": raw["TransactionAmt"],
            "channel": raw["ProductCD"],
            "label": raw["isFraud"].astype(bool),
            "source": "ieee_cis",
        }
    )
    df["device_id"] = None
    df["ip_country"] = None
    df["is_synthetic_field"] = [()] * len(df)

    if identity_path is not None:
        identity_path = Path(identity_path)
        if not identity_path.exists():
            logger.warning(
                "identity_path=%s không tồn tại — bỏ qua, device_id giữ None "
                "cho toàn bộ bản ghi.", identity_path,
            )
        else:
            identity = pd.read_csv(identity_path)
            if "TransactionID" not in identity.columns or "DeviceInfo" not in identity.columns:
                logger.warning(
                    "train_identity.csv thiếu TransactionID/DeviceInfo — bỏ qua device_id."
                )
            else:
                device_map = identity.drop_duplicates("TransactionID").set_index("TransactionID")["DeviceInfo"]
                df["device_id"] = raw["TransactionID"].map(device_map)
                n_matched = int(df["device_id"].notna().sum())
                logger.info(
                    "Đã join device_id (thật, không tổng hợp) từ train_identity.csv: "
                    "%d / %d giao dịch có DeviceInfo (%.1f%%). Phần còn lại giữ None — "
                    "identity table vốn không phủ hết mọi giao dịch.",
                    n_matched, len(df), 100 * n_matched / max(len(df), 1),
                )

    logger.warning(
        "load_ieee_cis: counterparty_id = None cho TOÀN BỘ bản ghi (nguồn này "
        "không có thông tin người nhận). KHÔNG dùng làm input cho Graph Encoder."
    )
    return df


def load_synthetic_vn(path: str | Path) -> pd.DataFrame:
    """Đọc dữ liệu synthetic tự sinh bởi src/data/synthetic_generator.py.

    Dùng cho kiểm thử concept drift — KHÔNG dùng để huấn luyện chính
    (xem cảnh báo về fraud_rate_assumed trong synthetic_generator.py).

    Args:
        path: đường dẫn tới CSV đã sinh (vd. data/raw/synthetic_vn.csv).
            Sinh file này bằng:
            `python -m src.data.synthetic_generator --output data/raw/synthetic_vn.csv`
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {path}. Sinh file bằng: "
            f"python -m src.data.synthetic_generator --output {path}"
        )
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["source"] = "synthetic_vn"
    return df
