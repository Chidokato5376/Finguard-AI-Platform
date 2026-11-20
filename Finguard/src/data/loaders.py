"""Loader cho ba nguồn dữ liệu: PaySim, IEEE-CIS, Synthetic VN.

Mỗi loader trả về pandas.DataFrame đã ánh xạ sang Unified Schema
(src/data/schema.py), kèm cờ is_synthetic_field cho các trường không
có trong dữ liệu gốc.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Cột gốc thực tế của PaySim (Kaggle) — KHÔNG bao gồm device_id/ip_country.
PAYSIM_RAW_COLUMNS = [
    "step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
    "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud",
]

PAYSIM_HOURS_PER_DAY = 24

# Máy sinh gian lận của PaySim phát ra một số vụ gần như KHÔNG ĐỔI mỗi ngày
# (220-316 vụ, trung bình 265 trên cả 31 ngày) trong khi lưu lượng hợp lệ sụp
# từ ~420.000 giao dịch/ngày xuống còn 282 giao dịch ở ngày cuối. Đo trên toàn
# bộ 6.362.620 dòng (2026-10-17):
#
#   - 17 trong 31 ngày chỉ chứa 425.107 dòng (6,7% dữ liệu) nhưng nắm 4.456
#     giao dịch gian lận (54% toàn bộ nhãn dương);
#   - ngày 30 có 282 giao dịch và CẢ 282 đều là gian lận.
#
# Giữ các ngày đó lại thì một phép chia 70/15/15 theo thời gian cho tập test
# tỷ lệ nền 0,601% so với train 0,123% (gấp 5 lần). Vì AUPRC phụ thuộc trực
# tiếp vào tỷ lệ nền, baseline "chỉ xếp theo số tiền" nhảy từ 0,0138 lên
# 0,0881 — cao gấp 6,4 lần mà không có cải thiện mô hình nào.
#
# Ngưỡng dưới đây lọc theo LƯU LƯỢNG, không dùng nhãn (nên không phải chọn
# mẫu theo nhãn). Khoảng cách giữa ngày lành (>= 350.000 dòng) và ngày hỏng
# (<= 58.000 dòng) rất lớn: mọi ngưỡng trong khoảng 60.000-350.000 đều chọn
# ra ĐÚNG cùng 14 ngày [0, 1, 5..16], nên đây không phải ngưỡng được tinh
# chỉnh để lấy kết quả đẹp.
PAYSIM_MIN_DAILY_VOLUME = 100_000


def healthy_day_mask(df: pd.DataFrame, min_daily_volume: int) -> pd.Series:
    """Mặt nạ True cho các giao dịch thuộc "ngày lành" (đủ lưu lượng).

    Dùng được cho mọi nguồn: PaySim tính ngày từ `step` (giờ), các nguồn có
    mốc thời gian thật tính từ `timestamp` — xem _day_index.

    Args:
        df: DataFrame đã ở Unified Schema.
        min_daily_volume: số giao dịch tối thiểu của một ngày để ngày đó
            được giữ. <= 0 nghĩa là KHÔNG lọc (giữ nguyên hành vi cũ).

    Returns:
        pandas.Series kiểu bool, cùng index với df.
    """
    if min_daily_volume <= 0:
        return pd.Series(True, index=df.index)
    day = _day_index(df)
    return day.map(day.value_counts()) >= min_daily_volume


def paysim_healthy_day_mask(
    df: pd.DataFrame, min_daily_volume: int = PAYSIM_MIN_DAILY_VOLUME
) -> pd.Series:
    """Như healthy_day_mask nhưng mặc định dùng ngưỡng đã đo của PaySim."""
    return healthy_day_mask(df, min_daily_volume)


def log_day_filter(
    df: pd.DataFrame, keep: pd.Series, min_daily_volume: int, source: str = "?"
) -> None:
    """Ghi log những ngày bị loại — KHÔNG được lọc âm thầm, vì đây là quyết
    định ảnh hưởng trực tiếp tới mọi con số báo cáo về sau."""
    day = _day_index(df)
    dropped_days = sorted(day[~keep].unique().tolist())
    if not dropped_days:
        logger.info("Lọc ngày lành '%s' (ngưỡng %d dòng/ngày): không ngày nào bị loại.",
                    source, min_daily_volume)
        return
    n_drop = int((~keep).sum())
    label = df["label"] if "label" in df.columns else None
    pos_drop = int(label[~keep].sum()) if label is not None else 0
    pos_all = int(label.sum()) if label is not None else 0
    logger.warning(
        "Lọc ngày lành '%s' (ngưỡng %d dòng/ngày): LOẠI %d ngày %s — %d dòng (%.2f%%) "
        "chứa %d nhãn dương (%.1f%% toàn bộ nhãn dương). Giữ %d ngày %s. Lý do: lưu lượng "
        "hợp lệ sụp đổ ở các ngày này trong khi số gian lận/ngày gần như không đổi, "
        "xem ghi chú tại PAYSIM_MIN_DAILY_VOLUME.",
        source, min_daily_volume, len(dropped_days), dropped_days, n_drop,
        100 * n_drop / max(len(df), 1),
        pos_drop, 100 * pos_drop / max(pos_all, 1),
        len(sorted(day[keep].unique())), sorted(day[keep].unique().tolist()),
    )


def load_paysim(
    path: str | Path, synthesize_device_ip: bool = False, min_daily_volume: int = 0
) -> pd.DataFrame:
    """Đọc PaySim và ánh xạ sang Unified Schema.

    Args:
        path: Đường dẫn tới file CSV PaySim gốc.
        synthesize_device_ip: Nếu True, sinh bổ sung device_id/ip_country
            (dùng cho demo/prototype). Bản ghi kết quả sẽ được đánh dấu
            is_synthetic_field=("device_id", "ip_country") — KHÔNG được
            trình bày như dữ liệu thật.
        min_daily_volume: lọc bỏ các "ngày hỏng" có lưu lượng thấp hơn
            ngưỡng này (xem PAYSIM_MIN_DAILY_VOLUME). Mặc định 0 = KHÔNG
            lọc, để hàm này giữ nguyên hành vi cũ cho mọi lời gọi hiện có;
            pipeline truyền PAYSIM_MIN_DAILY_VOLUME một cách tường minh.

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
    # Kiểm tra header trước rồi chỉ nạp các cột thực sự dùng. File đầy đủ có
    # 6,36 triệu dòng; đọc cả 11 cột (trong đó 4 cột số dư KHÔNG dùng — xem
    # ghi chú rò rỉ số dư trong data/README.md) tốn gần gấp đôi bộ nhớ và làm
    # tiến trình chết trên máy RAM 7,4 GB.
    header = pd.read_csv(path, nrows=0).columns
    missing = set(PAYSIM_RAW_COLUMNS) - set(header)
    if missing:
        raise ValueError(f"PaySim file thiếu cột: {missing}")
    raw = pd.read_csv(
        path,
        usecols=["step", "type", "amount", "nameOrig", "nameDest", "isFraud"],
        dtype={"step": "int32", "amount": "float64", "isFraud": "int8"},
    )

    df = pd.DataFrame(
        {
            # PaySim: người GỬI gần như không lặp lại (99.85% chỉ có đúng 1
            # giao dịch trong toàn bộ dữ liệu) nên Behavior Memory, L_pred và
            # L_contrast đều không có dữ liệu. Người NHẬN thì có: 17.9% xuất
            # hiện >= 2 lần, và 73.7% giao dịch gian lận có người nhận đã
            # từng xuất hiện. Vì vậy với PaySim, "tài khoản" được mô hình hóa
            # là BÊN NHẬN — cũng đúng với bài toán Money Mule (tài khoản
            # trung gian nhận tiền từ nhiều nguồn).
            "account_id": raw["nameDest"],
            "counterparty_id": raw["nameOrig"],
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

    if min_daily_volume > 0:
        keep = paysim_healthy_day_mask(df, min_daily_volume)
        log_day_filter(df, keep, min_daily_volume, "paysim")
        df = df[keep].reset_index(drop=True)
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
    # Chỉ đọc các cột thực sự dùng: file gốc có 394 cột (~2–3 GB RAM nếu
    # đọc hết) — quá nặng với máy RAM 8 GB, trong khi 11 cột này chỉ vài
    # chục MB. Kiểm tra header trước để giữ nguyên thông báo lỗi thiếu cột.
    header = pd.read_csv(transaction_path, nrows=0).columns
    missing = set(IEEE_CIS_TRANSACTION_REQUIRED_COLUMNS) - set(header)
    if missing:
        raise ValueError(f"IEEE-CIS train_transaction.csv thiếu cột: {missing}")
    raw = pd.read_csv(transaction_path, usecols=IEEE_CIS_TRANSACTION_REQUIRED_COLUMNS)

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


# Header gốc của IBM AMLWorld ("IBM Transactions for Anti Money Laundering",
# Kaggle: ealtman2019/ibm-transactions-for-anti-money-laundering-aml, giấy phép
# CDLA-Sharing-1.0). CHÚ Ý: file có HAI cột cùng tên "Account" (bên gửi và bên
# nhận), nên không đọc bằng tên cột được — phải đọc theo VỊ TRÍ.
#
# Repo chaos-max/TransXion dùng đúng schema này, nên cùng một loader đọc được
# cả hai, cho phép train trên AMLWorld rồi kiểm định chéo trên TransXion.
AMLWORLD_RAW_HEADER = [
    "timestamp", "from bank", "account", "to bank", "account",
    "amount received", "receiving currency", "amount paid", "payment currency",
    "payment format", "is laundering",
]

AMLWORLD_COLUMN_NAMES = [
    "timestamp", "from_bank", "from_account", "to_bank", "to_account",
    "amount_received", "receiving_currency", "amount_paid", "payment_currency",
    "payment_format", "is_laundering",
]


def _day_index(df: pd.DataFrame) -> pd.Series:
    """Số thứ tự NGÀY của mỗi giao dịch, dùng chung cho mọi nguồn."""
    if "step" in df.columns:  # PaySim: step là giờ kể từ mốc 0
        return (df["step"] // PAYSIM_HOURS_PER_DAY).astype(int)
    return pd.to_datetime(df["timestamp"]).dt.normalize().rank(method="dense").astype(int) - 1


def log_daily_volume_table(df: pd.DataFrame, source: str) -> None:
    """In bảng lưu lượng + tỷ lệ nhãn dương theo từng ngày, và cảnh báo những
    ngày có tỷ lệ gian lận lệch hẳn khỏi phần còn lại.

    Đây là bước kiểm định BẮT BUỘC trước khi tin bất kỳ con số nào từ một bộ
    dữ liệu mô phỏng: cả PaySim và AMLWorld đều có hiện tượng "máy sinh gian
    lận vẫn chạy khi lưu lượng hợp lệ đã tắt". Với PaySim, 17/31 ngày bị như
    vậy (xem PAYSIM_MIN_DAILY_VOLUME). Tác giả AMLWorld cũng đã xác nhận trong
    phần thảo luận trên Kaggle rằng các giao dịch nằm SAU khoảng thời gian
    chính của mỗi file ĐỀU là giao dịch rửa tiền.
    """
    day = _day_index(df)
    table = df.assign(_day=day).groupby("_day").agg(
        rows=("label", "size"), pos=("label", "sum"))
    table["rate"] = table["pos"] / table["rows"]
    overall = float(df["label"].mean()) if len(df) else 0.0
    logger.info("Lưu lượng theo ngày của '%s' (%d ngày, tỷ lệ nhãn dương chung %.5f):\n%s",
                source, len(table), overall, table.to_string())
    suspicious = table.index[table["rate"] > 10 * max(overall, 1e-12)].tolist()
    if suspicious:
        logger.warning(
            "Ngày %s có tỷ lệ nhãn dương cao hơn 10 lần mức chung — dấu hiệu lưu lượng hợp "
            "lệ đã tắt trong khi máy sinh gian lận vẫn chạy. Cân nhắc đặt min_daily_volume "
            "để loại các ngày này, nếu không tỷ lệ nền của các tập sẽ lệch nhau và mọi "
            "AUPRC sẽ phình lên mà không phản ánh chất lượng mô hình.", suspicious)


def _combine_bank_account(
    bank: pd.Series, account: pd.Series, prefix: str = "B"
) -> pd.Categorical:
    """Ghép "<prefix><bank>_<account>" thành định danh, trả về Categorical.

    Vì sao không ghép thẳng bằng toán tử `+`: trên HI-Small (5.078.345 dòng)
    phép đó tạo 5 triệu đối tượng str Python cho MỖI cột định danh (~0,5 GB
    mỗi cột), và tổng bộ nhớ của cả bước tiền xử lý đo được là ~9,7 GB — vượt
    RAM 7,4 GB của máy. Nhưng bộ này chỉ có ~515.000 tài khoản, nên chỉ cần
    tạo chuỗi cho từng CẶP DUY NHẤT rồi dùng mã nguyên: nhẹ hơn khoảng 10 lần.

    Lưu ý đã biết: nếu mã ngân hàng hoặc số tài khoản tự chứa dấu "_" thì hai
    cặp khác nhau có thể cho cùng một nhãn. Với AMLWorld cả hai đều là số nên
    không xảy ra; loader khác dùng lại hàm này cần tự kiểm.
    """
    b = bank.astype("category")
    a = account.astype("category")
    # Mã của Categorical dùng -1 cho giá trị thiếu. Dịch mọi mã lên 1 và chèn
    # nhãn "NA" ở đầu để giá trị thiếu thành một nhóm bình thường, thay vì gọi
    # fillna (không đặt được giá trị mới trên Categorical).
    b_cats = np.concatenate([["NA"], b.cat.categories.to_numpy().astype(str)])
    a_cats = np.concatenate([["NA"], a.cat.categories.to_numpy().astype(str)])
    n_a = len(a_cats)
    key = ((b.cat.codes.to_numpy().astype("int64") + 1) * n_a
           + a.cat.codes.to_numpy().astype("int64") + 1)
    codes, unique_keys = pd.factorize(key, sort=False)
    labels = pd.Index(
        [f"{prefix}{b_cats[k // n_a]}_{a_cats[k % n_a]}" for k in unique_keys], dtype=object)
    return pd.Categorical.from_codes(codes, categories=labels)


def load_amlworld(path: str | Path, source_name: str = "amlworld") -> pd.DataFrame:
    """Đọc IBM AMLWorld (hoặc TransXion, cùng schema) sang Unified Schema.

    Khác PaySim ở một điểm quan trọng: bộ này có ĐỊNH DANH THẬT CHO CẢ HAI
    ĐẦU giao dịch và cả hai đều lặp lại (≈10 giao dịch/tài khoản ở bản
    HI-Small: 5 triệu giao dịch trên 515.000 tài khoản). Vì vậy `account_id`
    lấy BÊN GỬI — đúng theo định nghĩa Unified Schema — thay vì bên nhận như
    PaySim (nơi 99,85% người gửi chỉ có đúng 1 giao dịch).

    Số tài khoản bị TRÙNG giữa các ngân hàng, nên định danh phải ghép mã ngân
    hàng: "B<bank>_<account>". Bỏ bước này là gộp nhầm hàng nghìn tài khoản
    khác nhau thành một.

    Args:
        path: CSV gốc, vd. data/raw/HI-Small_Trans.csv.
        source_name: giá trị cho cột `source` ("amlworld" hoặc "transxion").

    Returns:
        DataFrame theo Unified Schema, thêm hai cột `currency` và `is_fx`
        (giao dịch đổi tiền — một dấu hiệu của bước layering).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {path}. Tải bằng: kaggle datasets download -d "
            "ealtman2019/ibm-transactions-for-anti-money-laundering-aml -f HI-Small_Trans.csv"
        )
    # pandas tự đổi tên cột trùng: cột "Account" thứ hai thành "Account.1".
    # Bỏ hậu tố ".<số>" để so với header gốc của file.
    header = [
        re.sub(r"\.\d+$", "", c.strip().lower())
        for c in pd.read_csv(path, nrows=0).columns
    ]
    if header != AMLWORLD_RAW_HEADER:
        raise ValueError(
            f"Header không đúng schema AMLWorld.\nMong đợi: {AMLWORLD_RAW_HEADER}\n"
            f"Nhận được: {header}"
        )
    # Đọc các cột định danh/phân loại dưới dạng category: HI-Small có 5 triệu
    # dòng nhưng chỉ ~515.000 tài khoản, 7 hình thức thanh toán và ~11 loại
    # tiền, nên lưu bằng mã nguyên thay vì chuỗi tiết kiệm phần lớn bộ nhớ.
    raw = pd.read_csv(
        path, skiprows=1, names=AMLWORLD_COLUMN_NAMES,
        dtype={"from_bank": "category", "from_account": "category",
               "to_bank": "category", "to_account": "category",
               "receiving_currency": "category", "payment_currency": "category",
               "payment_format": "category"},
        parse_dates=["timestamp"],
    )

    df = pd.DataFrame(
        {
            "account_id": _combine_bank_account(raw["from_bank"], raw["from_account"]),
            "counterparty_id": _combine_bank_account(raw["to_bank"], raw["to_account"]),
            "timestamp": raw["timestamp"],
            "amount": raw["amount_paid"].astype(float),
            # channel/currency để dạng chuỗi thường (chỉ 7 và ~11 giá trị nên
            # Categorical chỉ tiết kiệm vài chục MB), vì encode_categoricals
            # và apply_categorical_mapping dùng fillna(-1) — không đặt được
            # giá trị mới trên Categorical. Chỉ hai cột ĐỊNH DANH giữ
            # Categorical, đó là chỗ tiết kiệm thật (515k thay vì 5 triệu chuỗi).
            "channel": raw["payment_format"].astype(str).str.strip().str.lower(),
            "label": raw["is_laundering"].astype(int).astype(bool),
            "source": source_name,
        }
    )
    pay = raw["payment_currency"].astype(str).str.strip()
    recv = raw["receiving_currency"].astype(str).str.strip()
    df["currency"] = pay
    df["is_fx"] = (pay != recv).to_numpy()
    df["device_id"] = None
    df["ip_country"] = None
    df["is_synthetic_field"] = [()] * len(df)

    logger.info(
        "AMLWorld '%s': %d giao dịch, %d rửa tiền (%.5f), %d tài khoản gửi, "
        "%d loại hình thanh toán, %.1f%% giao dịch đổi tiền",
        source_name, len(df), int(df["label"].sum()), df["label"].mean(),
        df["account_id"].nunique(), df["channel"].nunique(), 100 * df["is_fx"].mean(),
    )
    log_daily_volume_table(df, source_name)
    return df


# --------------------------------------------------------------------------
# Nhãn kiểu hình rửa tiền — {bộ}_Patterns.txt
# --------------------------------------------------------------------------

#: Tám kiểu hình rửa tiền AMLWorld gán nhãn. Tất cả đều là kiểu hình ĐỒ THỊ —
#: định nghĩa bằng hình dạng của mạng lưới giao dịch, không phải bằng thuộc
#: tính của một giao dịch đơn lẻ. Vì vậy recall theo từng kiểu hình là thước
#: đo trực tiếp nhất cho câu hỏi "nhánh đồ thị có đóng góp gì không".
AMLWORLD_TYPOLOGIES = (
    "FAN-OUT", "FAN-IN", "CYCLE", "GATHER-SCATTER",
    "SCATTER-GATHER", "BIPARTITE", "STACK", "RANDOM",
)

#: Nhãn cho giao dịch CÓ nhãn dương nhưng không thuộc mẫu nào trong file
#: Patterns. Trên HI-Small có 5.177 nhãn dương trong khi file Patterns chỉ
#: liệt kê 3.579 dòng, nên phần dư là thật và phải gọi tên, không được lặng
#: lẽ gán vào một kiểu hình nào đó.
PATTERN_UNSPECIFIED = "UNSPECIFIED"

_PATTERN_BEGIN_RE = re.compile(
    r"^BEGIN LAUNDERING ATTEMPT\s*-\s*([A-Z][A-Z\-]*)\s*:?\s*(.*?)\s*$"
)


def parse_amlworld_patterns(path: str | Path) -> pd.DataFrame:
    """Đọc `{bộ}_Patterns.txt` thành bảng các giao dịch thuộc mẫu rửa tiền.

    Định dạng file (đã kiểm trên HI-Small ngày 2026-10-19):

        BEGIN LAUNDERING ATTEMPT - FAN-OUT:  Max 16-degree Fan-Out
        2022/09/01 00:06,021174,800737690,012,80011F990,2848.96,Euro,...,ACH,1
        ...  (các giao dịch của mẫu, đúng 11 cột như Trans.csv)
        END LAUNDERING ATTEMPT - FAN-OUT

    Ba kiểu hình không có phần mô tả sau dấu hai chấm (BIPARTITE,
    SCATTER-GATHER, STACK) nên dấu hai chấm là tùy chọn.

    Returns:
        DataFrame gồm `account_id`, `counterparty_id`, `timestamp`, `amount`,
        `laundering_pattern`, `pattern_detail`, `pattern_id`. Định danh ghép
        theo ĐÚNG quy tắc của `_combine_bank_account` ("B<bank>_<account>")
        để ghép được với bảng giao dịch.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {path}. File Patterns đi kèm bộ AMLWorld, nằm cùng "
            "thư mục với file Trans.csv tương ứng."
        )

    rows: list[tuple[str, str, str, float, str, str, int]] = []
    current_type: str | None = None
    current_detail = ""
    pattern_id = -1
    unknown_types: set[str] = set()
    non_laundering = 0

    with open(path, "r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("BEGIN LAUNDERING ATTEMPT"):
                match = _PATTERN_BEGIN_RE.match(line)
                if match is None:
                    raise ValueError(f"Dòng {line_no}: không đọc được đầu mẫu {line!r}")
                current_type, current_detail = match.group(1), match.group(2)
                if current_type not in AMLWORLD_TYPOLOGIES:
                    unknown_types.add(current_type)
                pattern_id += 1
                continue
            if line.startswith("END LAUNDERING ATTEMPT"):
                current_type = None
                continue
            if current_type is None:
                continue

            parts = line.split(",")
            if len(parts) != len(AMLWORLD_COLUMN_NAMES):
                raise ValueError(
                    f"Dòng {line_no}: mong đợi {len(AMLWORLD_COLUMN_NAMES)} cột như "
                    f"Trans.csv, nhận được {len(parts)}: {line!r}"
                )
            # parts: timestamp, from_bank, from_account, to_bank, to_account,
            #        amount_received, receiving_currency, amount_paid,
            #        payment_currency, payment_format, is_laundering
            if parts[10].strip() != "1":
                non_laundering += 1
            rows.append((
                f"B{parts[1].strip()}_{parts[2].strip()}",
                f"B{parts[3].strip()}_{parts[4].strip()}",
                parts[0].strip(),
                round(float(parts[7]), 2),
                current_type,
                current_detail,
                pattern_id,
            ))

    if unknown_types:
        logger.warning(
            "File Patterns chứa kiểu hình ngoài danh sách đã biết: %s. Vẫn giữ "
            "nguyên tên, nhưng hãy cập nhật AMLWORLD_TYPOLOGIES.", sorted(unknown_types),
        )
    if non_laundering:
        # Giả định "mọi dòng trong file Patterns đều là rửa tiền" là nền tảng
        # của phép ghép ở attach_laundering_pattern (chỉ dựng khóa trên dòng có
        # nhãn dương). Giả định sai thì phép ghép bỏ sót mà không ai biết.
        raise ValueError(
            f"{non_laundering} dòng trong {path.name} có is_laundering != 1. "
            "attach_laundering_pattern giả định mọi dòng ở đây đều là rửa tiền — "
            "phải xem lại trước khi dùng."
        )

    out = pd.DataFrame(rows, columns=[
        "account_id", "counterparty_id", "timestamp", "amount",
        "laundering_pattern", "pattern_detail", "pattern_id",
    ])
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    logger.info(
        "Đã đọc %d giao dịch thuộc %d mẫu rửa tiền từ %s; số mẫu theo kiểu hình: %s",
        len(out), out["pattern_id"].nunique(), path.name,
        out.groupby("laundering_pattern", observed=True)["pattern_id"].nunique().to_dict(),
    )
    return out


def attach_laundering_pattern(df: pd.DataFrame, patterns: pd.DataFrame) -> pd.DataFrame:
    """Gắn cột `laundering_pattern` vào bảng giao dịch.

    CHỈ dựng khóa ghép trên các dòng CÓ nhãn dương (HI-Small: 5.177 dòng trên
    5.078.345). Hai lý do: mọi dòng trong file Patterns đều là rửa tiền (đã
    kiểm trong parse_amlworld_patterns), và dựng khóa chuỗi cho cả 5 triệu
    dòng chính là phép toán đã làm tràn RAM 7,4 GB ở lần tiền xử lý đầu tiên.

    CẢNH BÁO VỀ CÁCH DÙNG: cột này là NHÃN, suy ra từ đáp án. Nó chỉ được dùng
    để ĐÁNH GIÁ (recall theo từng kiểu hình) và để kiểm định bộ truy xuất tri
    thức. Tuyệt đối KHÔNG được đưa vào đặc trưng đầu vào của mô hình.

    Giá trị: một trong `AMLWORLD_TYPOLOGIES`; `PATTERN_UNSPECIFIED` cho giao
    dịch có nhãn dương nhưng không thuộc mẫu nào; NaN cho giao dịch không phải
    rửa tiền.
    """
    out = df.copy()
    categories = list(AMLWORLD_TYPOLOGIES) + [PATTERN_UNSPECIFIED]
    code_of = {name: i for i, name in enumerate(categories)}
    unspecified = code_of[PATTERN_UNSPECIFIED]

    label = out["label"].fillna(False).astype(bool).to_numpy()
    values = np.full(len(out), -1, dtype="int8")
    n_pos = int(label.sum())
    n_matched = 0

    if n_pos:
        lookup: dict[tuple[str, str, pd.Timestamp, float], str] = {}
        for acc, cpy, ts, amt, kind in zip(
            patterns["account_id"], patterns["counterparty_id"],
            patterns["timestamp"], patterns["amount"], patterns["laundering_pattern"],
        ):
            lookup.setdefault((acc, cpy, ts, amt), kind)

        pos = out.loc[label, ["account_id", "counterparty_id", "timestamp", "amount"]]
        matched = [
            code_of.get(lookup.get((str(a), str(c), t, round(float(m), 2))), unspecified)
            for a, c, t, m in zip(pos["account_id"], pos["counterparty_id"],
                                  pos["timestamp"], pos["amount"])
        ]
        values[label] = np.asarray(matched, dtype="int8")
        n_matched = int(((values >= 0) & (values != unspecified)).sum())

    out["laundering_pattern"] = pd.Categorical.from_codes(values, categories=categories)
    logger.info(
        "Gắn kiểu hình rửa tiền: %d/%d nhãn dương ghép được mẫu (%.1f%%), %d dòng "
        "rơi vào %s. Phân bố: %s",
        n_matched, n_pos, 100 * n_matched / n_pos if n_pos else 0.0,
        n_pos - n_matched, PATTERN_UNSPECIFIED,
        out["laundering_pattern"].value_counts().to_dict(),
    )
    return out
