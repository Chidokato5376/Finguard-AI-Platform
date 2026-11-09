"""Đặc trưng hành vi tính NHÂN QUẢ — chỉ dùng các giao dịch TRƯỚC đó.

=== VÌ SAO CÓ FILE NÀY ===

Trước 2026-10-17, DRRE chỉ nhận 2 con số mỗi giao dịch (`amount_norm`,
`channel_encoded`) trong khi mô hình có 3 nhánh và 56.201 tham số — ba nhánh
không có gì để hợp nhất. Bằng chứng cho thấy đây là chỗ nghẽn thật: trên
IEEE-CIS, GBDT với ĐÚNG hai đặc trưng đó đạt AUPRC 0,111; thêm đặc trưng lịch
sử tự tính thì lên 0,121-0,124. Thông tin nằm sẵn trong dữ liệu, chỉ là chưa
được đưa vào mô hình.

=== HAI NGUYÊN TẮC BẮT BUỘC ===

1. NHÂN QUẢ: mọi đặc trưng chỉ được dùng các dòng có thời điểm TRƯỚC dòng
   đang tính. Cách làm: `cumsum - giá_trị_hiện_tại`, `cumcount`, `shift(1)`.
   TUYỆT ĐỐI không dùng thống kê trên cả cột (mean/std/count toàn tập) — đó
   là rò rỉ tương lai, và nó làm AUPRC đẹp lên mà không có giá trị thật.
   tests/test_features.py kiểm chứng điều này bằng cách so với một bản tính
   thủ công theo từng dòng.

2. TÍNH TRÊN CẢ DÒNG DỮ LIỆU, TRƯỚC KHI CHIA TẬP: một giao dịch trong tập
   test được phép biết lịch sử của chính tài khoản đó ở giai đoạn train —
   đúng như hệ thống thật. Vì chỉ nhìn về quá khứ nên đây KHÔNG phải rò rỉ;
   ngược lại, tính riêng trong từng tập mới là sai (lịch sử bị cắt vụn ở
   ranh giới tập).

=== VÌ SAO MỖI NGUỒN MỘT BỘ ĐẶC TRƯNG KHÁC NHAU ===

- PaySim KHÔNG được dùng nhóm đồng hồ (`hour_*`, `dow`): giờ trong ngày là
  một ĐẶC TRƯNG RÒ RỈ của bộ này — giờ 4..8 có tỷ lệ gian lận đúng 100%, và
  một bảng tra 24 dòng theo giờ đạt AUPRC 0,2949, gấp đôi cả mô hình.
- IEEE-CIS KHÔNG có `counterparty_id` (None cho toàn bộ bản ghi) nên cả nhóm
  đặc trưng đối tác đều vô nghĩa với nguồn này.
- `is_fx` (giao dịch đổi tiền) chỉ AMLWorld/TransXion có, nhưng ĐÃ BỊ LOẠI
  khỏi bộ đặc trưng — xem cảnh báo ngay dưới đây.

=== CẢNH BÁO: `is_fx` LÀ ĐẶC TRƯNG RÒ RỈ NGƯỢC, KHÔNG ĐƯỢC ĐƯA VÀO MÔ HÌNH ===

Đo trên toàn bộ 5.078.345 dòng HI-Small ngày 2026-10-19: 72.170 giao dịch
đổi tiền (1,42% số dòng) có ĐÚNG 0 nhãn dương, trong khi giao dịch cùng loại
tiền có tỷ lệ 0,1034%. Máy sinh AMLWorld không bao giờ gán rửa tiền cho giao
dịch đổi tiền, nên `is_fx = 1` là một dấu hiệu LOẠI TRỪ hoàn hảo.

Nguy hiểm hơn việc AUPRC phình lên: ngoài thực tế, đổi tiền LÀ dấu hiệu của
bước layering — chiều quan hệ thật ngược hẳn. Mô hình học trên đặc trưng này
sẽ học đúng chiều ngược, tức khi triển khai thật sẽ GIẢM điểm rủi ro cho đúng
những giao dịch xuyên tiền tệ đáng nghi nhất.

Cột `is_fx` vẫn được loader sinh ra và vẫn nằm trong parquet, để còn báo cáo
được lỗi này; nó chỉ không nằm trong FEATURES_BY_SOURCE.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Số giây của một đơn vị cột `step`, theo từng nguồn. PaySim đếm step bằng
#: GIỜ, còn IEEE-CIS đếm bằng GIÂY (TransactionDT) — nhầm chỗ này thì mọi
#: đặc trưng khoảng cách thời gian sai 3600 lần.
SECONDS_PER_STEP = {"paysim": 3600.0, "ieee_cis": 1.0}

#: Hai đặc trưng gốc do preprocessing.py sinh ra.
BASE_FEATURES = ["amount_norm", "channel_encoded"]

#: Lịch sử của CHÍNH tài khoản đó.
ACCOUNT_HISTORY_FEATURES = ["log_n_prior", "log_gap", "z_amount", "log_amount_ratio"]

#: Quan hệ với đối tác — cần `counterparty_id` thật.
COUNTERPARTY_FEATURES = [
    "new_counterparty", "log_n_distinct_cp", "log_n_prior_as_cp", "log_n_distinct_senders",
]

#: Hình dạng số tiền (dấu hiệu structuring: chia nhỏ cho chẵn để lách ngưỡng).
AMOUNT_SHAPE_FEATURES = ["is_round_amount"]

#: Nhóm đồng hồ — xem cảnh báo về PaySim ở đầu file.
CLOCK_FEATURES = ["hour_sin", "hour_cos", "dow"]

#: Đổi tiền.
FX_FEATURES = ["is_fx"]

ALL_FEATURES = (
    BASE_FEATURES + ACCOUNT_HISTORY_FEATURES + COUNTERPARTY_FEATURES
    + AMOUNT_SHAPE_FEATURES + CLOCK_FEATURES + FX_FEATURES
)

#: Bộ đặc trưng dùng cho từng nguồn — nguồn nào thiếu thông tin thì KHÔNG
#: đưa đặc trưng tương ứng vào, thay vì gán 0 rồi để mô hình hiểu nhầm 0 là
#: một giá trị có nghĩa.
FEATURES_BY_SOURCE: dict[str, list[str]] = {
    # 11 đặc trưng — bỏ nhóm đồng hồ (rò rỉ) và is_fx (không có)
    "paysim": BASE_FEATURES + ACCOUNT_HISTORY_FEATURES + COUNTERPARTY_FEATURES
    + AMOUNT_SHAPE_FEATURES,
    # 10 đặc trưng — bỏ nhóm đối tác (counterparty_id = None)
    "ieee_cis": BASE_FEATURES + ACCOUNT_HISTORY_FEATURES + AMOUNT_SHAPE_FEATURES
    + CLOCK_FEATURES,
    # 14 đặc trưng
    "synthetic_vn": BASE_FEATURES + ACCOUNT_HISTORY_FEATURES + COUNTERPARTY_FEATURES
    + AMOUNT_SHAPE_FEATURES + CLOCK_FEATURES,
    # 14 đặc trưng — ALL_FEATURES TRỪ `is_fx` (rò rỉ ngược, xem đầu file)
    "amlworld": [f for f in ALL_FEATURES if f not in FX_FEATURES],
    "transxion": [f for f in ALL_FEATURES if f not in FX_FEATURES],
}


def features_for_source(source: str, include_graph: bool = True) -> list[str]:
    """Danh sách đặc trưng của một nguồn; nguồn lạ thì chỉ dùng BASE_FEATURES.

    Args:
        include_graph: kèm nhóm đặc trưng dòng tiền của
            `src.data.graph_features` (mặc định có). Đặt False khi gọi từ
            bên trong `add_causal_features`, vì hàm đó KHÔNG sinh ra nhóm
            kia — nhóm kia do `add_graph_features` sinh ở bước ngay sau.
    """
    cols = list(FEATURES_BY_SOURCE.get(source, BASE_FEATURES))
    if include_graph:
        from src.data.graph_features import graph_features_for_source

        cols += graph_features_for_source(source)
    return cols


def time_seconds(df: pd.DataFrame, source: str) -> pd.Series:
    """Trục thời gian quy về GIÂY, dùng chung cho mọi nguồn."""
    if "timestamp" in df.columns:
        return pd.to_datetime(df["timestamp"]).astype("int64") / 1e9
    per_step = SECONDS_PER_STEP.get(source)
    if per_step is None:
        raise KeyError(
            f"Không biết một đơn vị 'step' của nguồn '{source}' bằng bao nhiêu giây. "
            f"Thêm vào SECONDS_PER_STEP (hiện có: {sorted(SECONDS_PER_STEP)})."
        )
    return df["step"].astype(float) * per_step


def _hour_and_dow(df: pd.DataFrame, source: str) -> tuple[pd.Series, pd.Series]:
    """Giờ trong ngày (0-23) và ngày trong tuần (0-6).

    Với IEEE-CIS, mốc gốc của TransactionDT không được công bố nên "giờ" ở
    đây lệch một hằng số chưa biết so với giờ thật — vẫn là đặc trưng CHU KỲ
    hợp lệ (mô hình học được nhịp ngày/đêm), chỉ không đọc được theo nghĩa
    "3 giờ sáng". Ghi rõ để không ai diễn giải sai khi trình bày.
    """
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"])
        return ts.dt.hour.astype(float), ts.dt.dayofweek.astype(float)
    secs = time_seconds(df, source)
    return (secs // 3600) % 24, (secs // 86400) % 7


def add_causal_features(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Thêm toàn bộ đặc trưng nhân quả tính được cho `source`.

    Yêu cầu: df đã ở Unified Schema và đã có `amount_norm` (từ
    preprocessing.normalize_amount). Hàm tự sắp theo thời gian trước khi tính.

    Trả về DataFrame mới, đã sắp theo thời gian, có thêm các cột đặc trưng.
    """
    # include_graph=False: nhóm dòng tiền do add_graph_features sinh ở bước
    # sau, hàm này không chịu trách nhiệm cho chúng.
    wanted = set(features_for_source(source, include_graph=False))
    time_col = "step" if "step" in df.columns else "timestamp"
    out = df.sort_values(time_col, kind="stable").reset_index(drop=True).copy()

    acc = out["account_id"]
    log_amt = np.log1p(out["amount"].astype(float))

    # ---------------- Lịch sử của chính tài khoản ----------------
    # n_prior, tổng và tổng bình phương của các giao dịch TRƯỚC đó. Dùng
    # cumsum trừ đi giá trị hiện tại thay vì expanding().apply() — cùng kết
    # quả, nhưng chạy ở tầng C nên xử lý được 6 triệu dòng trong vài giây.
    n_prior = out.groupby(acc, sort=False, observed=True).cumcount()
    sum_prior = log_amt.groupby(acc, observed=True).cumsum() - log_amt
    sumsq_prior = (log_amt ** 2).groupby(acc, observed=True).cumsum() - log_amt ** 2
    n = n_prior.where(n_prior > 0)
    mean_prior = sum_prior / n
    # Phương sai của mẫu trước đó (hiệu chỉnh Bessel) — chỉ có nghĩa từ 2 giao dịch
    var_prior = ((sumsq_prior - sum_prior ** 2 / n) / (n - 1)).where(n_prior > 1)
    std_prior = np.sqrt(var_prior.clip(lower=0))

    out["log_n_prior"] = np.log1p(n_prior)
    out["log_amount_ratio"] = (log_amt - mean_prior).fillna(0.0).clip(-5, 5)
    out["z_amount"] = (
        ((log_amt - mean_prior) / std_prior.replace(0, np.nan)).fillna(0.0).clip(-5, 5)
    )

    secs = time_seconds(out, source)
    gap = secs - secs.groupby(acc, observed=True).shift(1)
    out["log_gap"] = np.log1p(gap.clip(lower=0)).fillna(0.0)

    # ---------------- Quan hệ với đối tác ----------------
    if wanted & set(COUNTERPARTY_FEATURES):
        cp = out["counterparty_id"]
        # Cặp (gửi, nhận) xuất hiện lần đầu. duplicated() giữ lần đầu = False.
        pair_new = ~out.duplicated(subset=["account_id", "counterparty_id"])
        pair_new_int = pair_new.astype(int)
        out["new_counterparty"] = pair_new.astype(float)
        # Số đối tác KHÁC NHAU mà tài khoản này đã giao dịch TRƯỚC đó
        out["log_n_distinct_cp"] = np.log1p(
            pair_new_int.groupby(acc, observed=True).cumsum() - pair_new_int)
        # Đối tác này đã nhận tiền bao nhiêu lần trước đó
        out["log_n_prior_as_cp"] = np.log1p(out.groupby(cp, sort=False, observed=True).cumcount())
        # Bao nhiêu NGƯỜI GỬI KHÁC NHAU đã chuyển cho đối tác này trước đó —
        # đây là dấu hiệu fan-in của tài khoản trung gian (money mule).
        out["log_n_distinct_senders"] = np.log1p(
            pair_new_int.groupby(cp, observed=True).cumsum() - pair_new_int)

    # ---------------- Hình dạng số tiền ----------------
    amount = out["amount"].astype(float)
    out["is_round_amount"] = (np.isclose(amount % 100, 0) & (amount > 0)).astype(float)

    # ---------------- Đồng hồ ----------------
    if wanted & set(CLOCK_FEATURES):
        hour, dow = _hour_and_dow(out, source)
        out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        out["dow"] = dow / 6.0

    # ---------------- Đổi tiền ----------------
    # `is_fx` KHÔNG còn nằm trong FEATURES_BY_SOURCE (rò rỉ ngược — xem đầu
    # file), nhưng vẫn được chuẩn hóa về float khi loader có sinh ra, để báo
    # cáo và kiểm định dùng chung một kiểu dữ liệu.
    if "is_fx" in out.columns:
        out["is_fx"] = out["is_fx"].astype(float)
    elif "is_fx" in wanted:
        raise KeyError(
            f"Nguồn '{source}' cần cột 'is_fx' từ loader nhưng không thấy. "
            "Xem src/data/loaders.py::load_amlworld."
        )

    # Chỉ kiểm những đặc trưng mà CHÍNH hàm này chịu trách nhiệm sinh ra.
    # BASE_FEATURES do bước khác tạo: `amount_norm` từ normalize_amount, còn
    # `channel_encoded` từ encode_categoricals — bước đó chạy SAU khi chia tập
    # (mapping chỉ được fit trên train để tránh rò rỉ), nên ở đây nó chưa tồn
    # tại. Việc kiểm đủ cột đặc trưng so với dữ liệu thật thuộc về
    # src/drre/train.py::train, nơi đọc parquet đã hoàn chỉnh.
    missing = wanted - set(BASE_FEATURES) - set(out.columns)
    if missing:
        raise RuntimeError(
            f"Thiếu đặc trưng {sorted(missing)} cho nguồn '{source}' sau khi tính — "
            "FEATURES_BY_SOURCE và add_causal_features không khớp nhau."
        )
    logger.info(
        "Đã thêm đặc trưng nhân quả cho '%s': %d đặc trưng %s | "
        "giao dịch có lịch sử: %.1f%%",
        source, len(wanted), sorted(wanted), 100 * float((n_prior > 0).mean()),
    )
    return out
