"""Data Service cho Dashboard — tách riêng khỏi app.py để có thể unit-test
mà không cần chạy Streamlit (Streamlit chỉ chạy được trong browser session).
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import pandas as pd


def load_processed_split(
    source: str, split: str, processed_dir: str | Path = "data/processed"
) -> pd.DataFrame:
    """Đọc một tập đã tiền xử lý (train/val/test) cho một nguồn dữ liệu.

    Args:
        source: "paysim" | "ieee_cis" | "synthetic_vn".
        split: "train" | "val" | "test".
        processed_dir: thư mục chứa output của src/data/preprocessing.py.

    Raises:
        FileNotFoundError: nếu chưa chạy preprocessing cho nguồn/split này.
    """
    path = Path(processed_dir) / f"{source}_{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {path}. Chạy trước: "
            f"python -m src.data.preprocessing --source {source} "
            f"--raw-path data/raw/{source}.csv --output {processed_dir}"
        )
    return pd.read_parquet(path)


def list_available_sources(processed_dir: str | Path = "data/processed") -> list[str]:
    """Quét data/processed/ để tìm các nguồn đã sẵn sàng — dùng cho dropdown UI."""
    processed_dir = Path(processed_dir)
    if not processed_dir.exists():
        return []
    sources = set()
    for f in processed_dir.glob("*_train.parquet"):
        sources.add(f.name.removesuffix("_train.parquet"))
    return sorted(sources)


def get_recent_transactions(df: pd.DataFrame, n: int = 200, time_col: str = "step") -> pd.DataFrame:
    """Lấy N giao dịch gần nhất theo thời gian — mô phỏng 'hàng đợi live'.

    Với dữ liệu thật (streaming), hàm này sẽ được thay bằng đọc từ hàng đợi
    tin nhắn (Kafka/queue) thay vì cắt lát tĩnh từ tập đã lưu.
    """
    col = time_col if time_col in df.columns else "timestamp"
    if col not in df.columns:
        raise KeyError(f"Không tìm thấy cột thời gian ('{time_col}' hoặc 'timestamp') trong DataFrame")
    return df.sort_values(col, ascending=False).head(n).reset_index(drop=True)


def build_account_graph(df: pd.DataFrame, max_edges: int = 2000) -> nx.DiGraph:
    """Dựng đồ thị quan hệ tài khoản từ giao dịch — dùng cho panel mạng lưới.

    Trọng số cạnh = số lần giao dịch giữa hai tài khoản (không phải tổng
    giá trị) — đơn giản hóa cho mục đích trực quan hóa trên dashboard, khác
    với đồ thị đầy đủ dùng để huấn luyện GraphRelationshipEncoder.
    """
    df_sample = df.tail(max_edges)  # giới hạn để render không bị chậm
    graph = nx.DiGraph()
    for _, row in df_sample.iterrows():
        u, v = row["account_id"], row["counterparty_id"]
        if graph.has_edge(u, v):
            graph[u][v]["weight"] += 1
            graph[u][v]["total_amount"] += float(row["amount"])
        else:
            graph.add_edge(u, v, weight=1, total_amount=float(row["amount"]))
    return graph


def get_account_neighborhood(
    graph: nx.DiGraph, account_id: str, hops: int = 2
) -> nx.DiGraph:
    """Trích xuất subgraph quanh một tài khoản trong bán kính `hops` —
    dùng để hiển thị 'các tài khoản liên quan' cho một cảnh báo cụ thể.
    """
    if account_id not in graph:
        return nx.DiGraph()
    undirected = graph.to_undirected()
    nodes = {account_id}
    frontier = {account_id}
    for _ in range(hops):
        next_frontier = set()
        for node in frontier:
            next_frontier |= set(undirected.neighbors(node))
        nodes |= next_frontier
        frontier = next_frontier
    return graph.subgraph(nodes).copy()
