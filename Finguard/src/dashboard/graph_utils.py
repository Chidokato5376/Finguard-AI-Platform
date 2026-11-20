"""Dựng đồ thị giao dịch (node_features, edge_index) từ một batch dữ liệu
đã tiền xử lý — dùng để chạy Graph Encoder (Mục 4.3) theo thời gian thực
trên Dashboard.

Đây là bản dựng đồ thị ĐƠN GIẢN cho mục đích demo/inference tại chỗ: mỗi
lần gọi build_account_graph() dựng lại toàn bộ đồ thị từ batch hiện tại
(không lưu trạng thái đồ thị giữa các batch). Với hệ thống sản xuất thật,
cần một graph store bền vững (không nằm trong phạm vi Dashboard scaffold
này) — xem cảnh báo latency trong docs/architecture.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


def _to_node_index(
    series: pd.Series, mapping: dict[str, int], column: str
) -> np.ndarray:
    """Ánh xạ một cột định danh sang mảng chỉ số node kiểu int64.

    VÌ SAO PHẢI CÓ HÀM RIÊNG THAY VÌ GỌI THẲNG `.map()`:

    Loader AMLWorld lưu `account_id` và `counterparty_id` dưới dạng
    Categorical để tiết kiệm bộ nhớ (5,08 triệu dòng nhưng chỉ ~497.000 tài
    khoản — ghép chuỗi thẳng tốn ~0,5 GB mỗi cột và từng làm tràn RAM 7,4 GB
    của máy nhóm). Nhưng `Series.map()` trên một Categorical ánh xạ các
    CATEGORY rồi trả về một Categorical MỚI có categories là số nguyên, nên
    `.to_numpy()` cho mảng dtype=object. numpy từ chối dùng mảng object làm
    chỉ số, với thông báo không hề nhắc tới Categorical:

        IndexError: arrays used as indices must be of integer (or boolean) type

    Lỗi này nằm im cho tới lần đầu chạy DRRE trên AMLWorld (2026-10-20), vì
    PaySim và dữ liệu tổng hợp đều để định danh dưới dạng chuỗi thường.

    Raises:
        KeyError: có giá trị không nằm trong `mapping` — nghĩa là đồ thị và
            khung dữ liệu được dựng từ hai tập dòng khác nhau.
    """
    mapped = series.map(mapping)
    if isinstance(mapped.dtype, pd.CategoricalDtype):
        mapped = mapped.astype("object")
    if mapped.isna().any():
        thieu = series[mapped.isna()].astype(str).unique()[:5].tolist()
        raise KeyError(
            f"{int(mapped.isna().sum())}/{len(series)} giá trị của cột {column!r} "
            f"không có trong account_to_idx (vd. {thieu}). Đồ thị và khung dữ liệu "
            "phải được dựng từ CÙNG một tập dòng."
        )
    return mapped.to_numpy(dtype=np.int64)


def build_account_graph(
    df: pd.DataFrame, feature_cols: list[str], use_graph: bool = True
) -> tuple["torch.Tensor", "torch.Tensor", dict[str, int]]:
    """Dựng (node_features, edge_index, account_to_idx) từ một batch giao dịch.

    node_features cho mỗi tài khoản = trung bình các đặc trưng của các
    giao dịch mà tài khoản đó đóng vai trò account_id (sender) trong batch.
    Tài khoản chỉ xuất hiện với vai trò counterparty (chưa từng là sender
    trong batch hiện tại) được gán vector 0 — tương đương one giả lập
    trường hợp cold-start tại chỗ.

    Args:
        df: batch giao dịch, cần có account_id, counterparty_id, và các
            cột trong feature_cols.
        feature_cols: tên cột đặc trưng dùng làm node feature (phải khớp
            input_dim của DRRE).

    Returns:
        node_features: (num_nodes, len(feature_cols))
        edge_index: (2, num_edges) COO format — hàng 0 = sender, hàng 1 = receiver.
        account_to_idx: ánh xạ account_id (string) -> chỉ số node.
    """
    if torch is None:
        raise ImportError(
            "torch chưa được cài — cần cho Graph Encoder. "
            "Chạy: pip install torch torch-geometric"
        )
    missing = set(feature_cols) - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame thiếu cột đặc trưng cho node: {missing}")

    if not use_graph:
        # Chế độ tắt đồ thị (train.py --no-graph): dùng cho nguồn không có
        # người nhận (IEEE-CIS) hoặc để làm ablation. Mọi nút có đặc trưng 0
        # và không có cạnh -> Graph Encoder chỉ cho ra một vector hằng; nhánh
        # đồ thị đồng thời bị tắt hẳn ở gate (xem AdaptiveRiskFusion).
        accounts = pd.unique(df["account_id"])
        return (
            torch.zeros((len(accounts), len(feature_cols)), dtype=torch.float32),
            torch.empty((2, 0), dtype=torch.long),
            {acc: i for i, acc in enumerate(accounts)},
        )

    # Chặn cứng: một số nguồn dữ liệu (vd. IEEE-CIS, xem
    # src/data/loaders.py::load_ieee_cis) không có counterparty_id thật và
    # để giá trị None cho TOÀN BỘ bản ghi. Nếu không chặn ở đây, mọi hàng
    # sẽ map về CÙNG một "node None" — tạo ra một hub giả kết nối với mọi
    # tài khoản, Graph Encoder học trên cấu trúc đồ thị vô nghĩa mà KHÔNG
    # có bất kỳ lỗi/cảnh báo runtime nào (đã xác nhận qua kiểm thử thực
    # tế — đây là lỗi âm thầm nguy hiểm hơn nhiều so với việc crash).
    n_null_counterparty = df["counterparty_id"].isna().sum()
    if n_null_counterparty > 0:
        raise ValueError(
            f"{n_null_counterparty}/{len(df)} dòng có counterparty_id = None. "
            "Graph Encoder không thể dùng nguồn dữ liệu không có counterparty "
            "thật (vd. IEEE-CIS — xem cảnh báo trong "
            "src/data/loaders.py::load_ieee_cis). Lọc bỏ các dòng này trước, "
            "hoặc không dùng nguồn này cho DRRE/Dashboard."
        )

    accounts = pd.unique(
        pd.concat([df["account_id"], df["counterparty_id"]], ignore_index=True)
    )
    account_to_idx = {acc: i for i, acc in enumerate(accounts)}

    feature_dim = len(feature_cols)
    node_features = np.zeros((len(accounts), feature_dim), dtype=np.float32)
    counts = np.zeros(len(accounts), dtype=np.float32)

    sender_idx = _to_node_index(df["account_id"], account_to_idx, "account_id")
    feature_values = df[feature_cols].to_numpy(dtype=np.float32)
    np.add.at(node_features, sender_idx, feature_values)
    np.add.at(counts, sender_idx, 1.0)

    nonzero = counts > 0
    node_features[nonzero] /= counts[nonzero, None]

    edge_index_np = np.stack(
        [
            sender_idx,
            _to_node_index(df["counterparty_id"], account_to_idx, "counterparty_id"),
        ]
    )

    return (
        torch.tensor(node_features, dtype=torch.float32),
        torch.tensor(edge_index_np, dtype=torch.long),
        account_to_idx,
    )
