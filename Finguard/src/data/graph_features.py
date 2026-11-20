"""Đặc trưng ĐỒ THỊ tính nhân quả — dòng tiền vào/ra của từng tài khoản.

=== VÌ SAO CÓ FILE NÀY ===

Đo ngày 2026-10-23: trên AMLWorld, nhánh đồ thị của DRRE mang **92,4% tín
hiệu** cho các giao dịch bị gắn cờ (trọng số gate beta_graph trung bình trên
nhãn dương), trong khi ta mới cấp cho nó **2 trong 5** đặc trưng đồ thị cơ
bản: số đối tác gửi đi (`log_n_distinct_cp`) và số người gửi tới
(`log_n_distinct_senders`). Cả hai đều là phép ĐẾM.

Thiếu hẳn nhóm **giá trị dòng tiền**: một tài khoản nhận 10 khoản 1.000 đồng
và một tài khoản nhận 10 khoản 1 tỷ đồng hiện đang có đặc trưng giống hệt
nhau.

=== VÌ SAO KHÔNG DÙNG PageRank / BETWEENNESS CỦA networkx ===

Hai đại lượng đó là TOÀN CỤC: muốn tính nhân quả phải dựng lại đồ thị của
quá khứ tại từng mốc thời gian (ảnh chụp theo ngày). Betweenness chính xác
là O(V·E); với ~500.000 nút và ~3 triệu cạnh thì networkx thuần Python không
kham nổi, còn bản lấy mẫu thì vừa chậm vừa nhiễu.

Nhưng Ý NGHĨA của betweenness — "tài khoản trung gian chuyển tiếp", tức
money mule — tính được CHÍNH XÁC và RẺ: một tài khoản trung chuyển là tài
khoản vừa nhận nhiều vừa chuyển đi gần hết. Đó chính là `passthrough_ratio`
và `flow_imbalance` dưới đây, tính bằng đúng kỹ thuật `cumsum - giá trị hiện
tại` đã dùng trong features.py.

=== ĐIỂM MỚI QUAN TRỌNG: CHIỀU NHẬN ===

`account_id` của Unified Schema là BÊN GỬI, nên tới nay mọi đặc trưng lịch
sử đều chỉ nhìn chiều gửi. Đó là lý do đã đo được cho điểm yếu FAN-IN (DRRE
66% so với GBDT 75–80%): hiện tượng "nhiều người gửi tới" xảy ra ở phía
NHẬN, mà bên gửi của giao dịch hiện tại không có đặc trưng nào mô tả lịch sử
NHẬN của chính nó.

File này sửa điều đó mà KHÔNG phải nhân đôi số dòng: mỗi giao dịch được tách
thành hai sự kiện (bên gửi: một lần chi; bên nhận: một lần thu), cộng dồn
theo từng thực thể, rồi ghép ngược về dòng gốc. Nhờ vậy bên gửi của mỗi giao
dịch có cả lịch sử CHI và lịch sử THU của chính nó.

=== BẤT BIẾN NHÂN QUẢ ===

Mọi đặc trưng ở đây chỉ dùng các sự kiện TRƯỚC sự kiện đang xét, kể cả sự
kiện đối ứng của chính giao dịch đó (bên gửi không nhìn thấy khoản thu của
bên nhận trong cùng giao dịch). tests/test_graph_features.py kiểm chứng bằng
cách thêm dòng tương lai và đòi các giá trị cũ không đổi.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Dòng tiền tích lũy của chính tài khoản đang gửi.
FLOW_FEATURES = [
    "log_amt_out_prior",    # tổng tiền tài khoản này ĐÃ CHUYỂN ĐI trước đó
    "log_amt_in_prior",     # tổng tiền tài khoản này ĐÃ NHẬN trước đó  <- mới
    "log_n_in_prior",       # số lần đã nhận trước đó                   <- mới
]

#: Dấu hiệu trung chuyển (thay cho betweenness).
MULE_FEATURES = [
    "passthrough_ratio",    # min(thu, chi) / max(thu, chi) — 1 = trung chuyển hoàn hảo
    "flow_imbalance",       # (thu - chi) / (thu + chi) — gần 0 = nhận bao nhiêu chuyển đi bấy nhiêu
]

#: Phía đối tác.
COUNTERPARTY_FLOW_FEATURES = [
    "log_cp_amt_in_prior",  # tổng tiền ĐỐI TÁC đã nhận trước đó
]

ALL_GRAPH_FEATURES = FLOW_FEATURES + MULE_FEATURES + COUNTERPARTY_FLOW_FEATURES

#: Nguồn nào có đủ định danh hai đầu để tính được nhóm này.
GRAPH_FEATURE_SOURCES = ("amlworld", "transxion", "paysim", "synthetic_vn")


def graph_features_for_source(source: str) -> list[str]:
    """Danh sách đặc trưng đồ thị dùng được cho một nguồn.

    `ieee_cis` không có `counterparty_id` thật nên không có nhóm nào.
    """
    return list(ALL_GRAPH_FEATURES) if source in GRAPH_FEATURE_SOURCES else []


def _entity_flow_state(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Trạng thái dòng tiền TRƯỚC mỗi sự kiện, cho bên gửi và bên nhận.

    Mỗi giao dịch sinh HAI sự kiện trên cùng một trục thời gian:
    bên gửi chi một khoản, bên nhận thu một khoản. Cộng dồn theo từng thực
    thể rồi trừ đi chính sự kiện hiện tại cho ra trạng thái ngay TRƯỚC nó.

    Returns:
        (trạng thái của bên gửi, trạng thái của bên nhận) — cả hai đã được
        đánh chỉ số theo đúng thứ tự dòng của `df`.
    """
    n = len(df)
    order = np.arange(n, dtype=np.int64)
    amount = df["amount"].astype(float).to_numpy()

    long = pd.DataFrame({
        "entity": pd.concat(
            [df["account_id"].astype(str), df["counterparty_id"].astype(str)],
            ignore_index=True,
        ).astype("category"),
        "row": np.concatenate([order, order]),
        "is_out": np.concatenate([np.ones(n, bool), np.zeros(n, bool)]),
        "amount": np.concatenate([amount, amount]),
    })

    # Sắp theo (thực thể, thứ tự thời gian). `df` đã được sắp theo thời gian
    # từ add_causal_features nên chỉ số dòng CHÍNH LÀ trục thời gian.
    long = long.sort_values(["entity", "row"], kind="stable")

    long["_out_amt"] = np.where(long["is_out"], long["amount"], 0.0)
    long["_in_amt"] = np.where(long["is_out"], 0.0, long["amount"])
    long["_in_cnt"] = (~long["is_out"]).astype(float)

    grouped = long.groupby("entity", sort=False, observed=True)
    for col in ("_out_amt", "_in_amt", "_in_cnt"):
        long["prior" + col] = grouped[col].cumsum() - long[col]

    keep = ["row", "prior_out_amt", "prior_in_amt", "prior_in_cnt"]
    sender = long.loc[long["is_out"], keep].set_index("row").reindex(order)
    receiver = long.loc[~long["is_out"], keep].set_index("row").reindex(order)
    return sender, receiver


def add_graph_features(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Thêm các đặc trưng dòng tiền nhân quả.

    Yêu cầu `df` ĐÃ được sắp theo thời gian (add_causal_features làm việc đó
    và trả về khung đã sắp) và có `account_id`, `counterparty_id`, `amount`.

    Nguồn không có đối tác thật (ieee_cis) được trả về nguyên vẹn.
    """
    wanted = graph_features_for_source(source)
    if not wanted:
        logger.info("Nguồn '%s' không có counterparty thật — bỏ qua đặc trưng đồ thị.", source)
        return df
    if df["counterparty_id"].isna().any():
        raise ValueError(
            f"Nguồn '{source}' có counterparty_id rỗng — không tính được dòng tiền hai chiều. "
            "Lọc bỏ các dòng đó trước, hoặc loại nguồn này khỏi GRAPH_FEATURE_SOURCES."
        )

    out = df.copy()
    sender, receiver = _entity_flow_state(out)

    amt_out = sender["prior_out_amt"].to_numpy()
    amt_in = sender["prior_in_amt"].to_numpy()
    cnt_in = sender["prior_in_cnt"].to_numpy()

    out["log_amt_out_prior"] = np.log1p(np.clip(amt_out, 0, None))
    out["log_amt_in_prior"] = np.log1p(np.clip(amt_in, 0, None))
    out["log_n_in_prior"] = np.log1p(np.clip(cnt_in, 0, None))

    # Trung chuyển: thu và chi gần bằng nhau. Tài khoản chưa có cả thu lẫn
    # chi thì KHÔNG phải trung chuyển, nên để 0 chứ không để NaN.
    hi = np.maximum(amt_in, amt_out)
    lo = np.minimum(amt_in, amt_out)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["passthrough_ratio"] = np.where(hi > 0, lo / hi, 0.0)
        total = amt_in + amt_out
        out["flow_imbalance"] = np.where(total > 0, (amt_in - amt_out) / total, 0.0)

    out["log_cp_amt_in_prior"] = np.log1p(
        np.clip(receiver["prior_in_amt"].to_numpy(), 0, None))

    for col in wanted:
        values = out[col].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise RuntimeError(f"Đặc trưng '{col}' có giá trị không hữu hạn — xem lại phép chia.")

    logger.info(
        "Đã thêm %d đặc trưng đồ thị cho '%s' | giao dịch mà bên gửi ĐÃ TỪNG NHẬN tiền: "
        "%.1f%% | trung chuyển (passthrough > 0,5): %.1f%%",
        len(wanted), source, 100 * float((cnt_in > 0).mean()),
        100 * float((out["passthrough_ratio"] > 0.5).mean()),
    )
    return out
