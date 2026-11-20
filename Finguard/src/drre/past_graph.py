"""Đồ thị CHỈ GỒM CẠNH QUÁ KHỨ cho nhánh mạng lưới (Giai đoạn 2B1, lỗi 5).

=== VẤN ĐỀ CỦA ĐỒ THỊ CỬA SỔ ===

`src/dashboard/graph_utils.py::build_account_graph` dựng MỘT đồ thị cho cả cửa
sổ 32.768 giao dịch (≈ 1,5 giờ dữ liệu HI-Small). Mọi giao dịch trong cửa sổ
thấy cùng đồ thị đó, nên một giao dịch lúc 10:00 thấy cả các cạnh lúc 11:20, và
đặc trưng nút là trung bình trên mọi giao dịch của tài khoản trong cửa sổ — kể
cả chính nó và các giao dịch SAU nó. Tức là nhánh đồ thị nhìn trước tương lai.

=== CÁCH LÀM Ở ĐÂY ===

Cửa sổ được chia thành các khối `block` dòng liên tiếp (mặc định 2.048, ≈ 5 phút).
Mọi dòng trong khối có thời điểm bắt đầu τ dùng chung một đồ thị gồm đúng các
giao dịch có thời điểm trong [τ − lookback, τ) — chặt chẽ TRƯỚC τ, kể cả các dòng
của khối trước trong cùng cửa sổ và của các cửa sổ trước (bộ đệm mang qua cửa sổ).

  - Đặc trưng nút = trung bình đặc trưng THÔ của các giao dịch tài khoản đó đã GỬI
    trong khoảng đó (giống build_account_graph, nhưng chỉ quá khứ).
  - Cạnh: người gửi → người nhận, như đồ thị cũ (hai chiều là việc của 2B2).
  - Chỉ dựng vùng lân cận L bước NGƯỢC chiều cạnh quanh các tài khoản gửi của khối
    (L = số lớp của Graph Encoder). Với GraphSAGE gộp trung bình, embedding của các
    nút đích trên đồ thị con này TRÙNG KHÍT với embedding trên toàn đồ thị khoảng đó
    (tests/test_past_graph.py), nhưng rẻ hơn nhiều.

Thêm giao dịch tương lai không làm đổi điểm của giao dịch cũ — đây là phép thử
của lộ trình (tests/test_past_graph.py::test_future_rows_do_not_change_past_scores).

Bộ đệm chỉ là dữ liệu (không có trọng số), nên mạng online và mạng EMA dùng chung
cùng các đồ thị khối.

=== CHẾ ĐỘ "edge" (Giai đoạn 2B2–2B5, R5) ===

`arch="edge"` dựng cho bộ mã hóa hai chiều có đặc trưng cạnh (model_v2.BiEdgeGraphEncoder):
  - Nút đích gồm CẢ tài khoản gửi lẫn tài khoản nhận của từng dòng (đọc theo cạnh).
  - Vùng lân cận mở rộng theo CẢ HAI chiều cạnh, L bước.
  - Mỗi nút có hồ sơ phía gửi (trung bình đặc trưng các giao dịch đã gửi) và phía nhận
    (trung bình các giao dịch đã nhận) cùng số lượng mỗi phía — nút chỉ-nhận không
    còn là vector 0.
  - Cạnh mang đặc trưng thô của giao dịch và độ cũ (τ − t)/lookback ∈ (0, 1].
  - Giới hạn bậc `fanout`: một cạnh chỉ được truyền tin nếu nó nằm trong `fanout` cạnh
    RA gần nhất của người gửi VÀ trong `fanout` cạnh VÀO gần nhất của người nhận (tính
    tới τ). Tập cạnh này xác định chỉ bởi τ (không phụ thuộc nút đích), nên đồ thị con
    vẫn trùng khít đồ thị đầy đủ của tập cạnh đó, và tài khoản trung chuyển lớn (hub)
    không kéo cả đồ thị vào mỗi khối. Hồ sơ nút thì vẫn tính trên MỌI cạnh quá khứ.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

DEFAULT_LOOKBACK_HOURS = 24.0
DEFAULT_BLOCK = 2048
DEFAULT_FANOUT = 10
GRAPH_MODES = ("window", "past")
GRAPH_ARCHES = ("sage", "edge")


def time_column(df: pd.DataFrame) -> str:
    return "step" if "step" in df.columns else "timestamp"


def time_values(df: pd.DataFrame, time_col: str) -> np.ndarray:
    """Thời điểm dạng int64: nano giây cho datetime, nguyên giá trị cho `step` (giờ)."""
    col = df[time_col]
    if pd.api.types.is_datetime64_any_dtype(col):
        return col.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    return col.to_numpy(dtype=np.int64)


def lookback_units(hours: float, time_col: str) -> int:
    """Độ dài khoảng nhìn lại theo đơn vị của `time_values`."""
    return int(round(hours)) if time_col == "step" else int(round(hours * 3600 * 1e9))


@dataclass
class BlockGraph:
    """Đồ thị quá khứ của một khối dòng [start, stop) trong cửa sổ."""

    start: int
    stop: int
    node_features: torch.Tensor  # (n_nodes, n_features) đặc trưng thô
    edge_index: torch.Tensor  # (2, n_edges) chỉ số cục bộ
    row_node: torch.Tensor  # (stop - start,) nút của tài khoản gửi từng dòng
    # --- chỉ có ở arch="edge" ---
    row_cp_node: torch.Tensor | None = None  # nút của tài khoản nhận từng dòng
    in_features: torch.Tensor | None = None  # (n_nodes, n_features) trung bình phía nhận
    counts: torch.Tensor | None = None  # (n_nodes, 2) số giao dịch đã gửi, đã nhận
    edge_x: torch.Tensor | None = None  # (n_edges, n_features) đặc trưng thô của cạnh
    edge_age: torch.Tensor | None = None  # (n_edges,) (τ − t)/lookback


def _group_seq(keys: np.ndarray) -> np.ndarray:
    """Số thứ tự (0, 1, ...) của mỗi phần tử trong nhóm cùng khóa, theo thứ tự xuất hiện."""
    if not len(keys):
        return np.empty(0, dtype=np.int64)
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    starts = np.r_[True, sorted_keys[1:] != sorted_keys[:-1]]
    group_start = np.maximum.accumulate(np.where(starts, np.arange(len(keys)), 0))
    seq = np.empty(len(keys), dtype=np.int64)
    seq[order] = np.arange(len(keys)) - group_start
    return seq


class PastGraphBuffer:
    """Giữ các giao dịch trong `lookback` gần nhất, mang qua các cửa sổ liên tiếp.

    Dùng: `graphs = buf.window_graphs(window)` rồi `buf.append(window)` sau khi đã
    chấm xong cửa sổ đó. Các cửa sổ PHẢI đến theo đúng thứ tự thời gian.
    """

    def __init__(self, feature_columns, lookback_hours: float = DEFAULT_LOOKBACK_HOURS,
                 block: int = DEFAULT_BLOCK, num_hops: int = 2, arch: str = "sage",
                 fanout: int = DEFAULT_FANOUT) -> None:
        if block < 1 or num_hops < 1 or lookback_hours <= 0 or fanout < 1:
            raise ValueError("block, num_hops, fanout và lookback_hours phải dương")
        if arch not in GRAPH_ARCHES:
            raise ValueError(f"arch phải thuộc {GRAPH_ARCHES}, nhận {arch!r}")
        self.feature_columns = list(feature_columns)
        self.lookback_hours = float(lookback_hours)
        self.block = int(block)
        self.num_hops = int(num_hops)
        self.arch = arch
        self.fanout = int(fanout)
        self.reset()

    # ------------------------------------------------------------------ trạng thái
    def reset(self) -> None:
        self._ids: dict = {}
        self._src = np.empty(0, dtype=np.int64)
        self._dst = np.empty(0, dtype=np.int64)
        self._t = np.empty(0, dtype=np.int64)
        self._x = np.empty((0, len(self.feature_columns)), dtype=np.float32)
        self._last_t: int | None = None

    def __len__(self) -> int:
        return len(self._t)

    def state_dict(self) -> dict:
        return {
            "ids": list(self._ids), "src": self._src, "dst": self._dst, "t": self._t,
            "x": self._x, "last_t": self._last_t, "feature_columns": self.feature_columns,
            "lookback_hours": self.lookback_hours,
        }

    def load_state_dict(self, state: dict) -> None:
        if list(state["feature_columns"]) != self.feature_columns:
            raise ValueError("Bộ đệm đồ thị lưu với bộ cột khác")
        self._ids = {k: i for i, k in enumerate(state["ids"])}
        self._src, self._dst = np.asarray(state["src"]), np.asarray(state["dst"])
        self._t, self._x = np.asarray(state["t"]), np.asarray(state["x"])
        self._last_t = state["last_t"]

    # ------------------------------------------------------------------ mã tài khoản
    def _codes(self, values: pd.Series) -> np.ndarray:
        s = values.astype(object)
        codes = s.map(self._ids)
        if codes.isna().any():
            new = pd.unique(s[codes.isna()])
            base = len(self._ids)
            self._ids.update(zip(new, range(base, base + len(new))))
            codes = s.map(self._ids)
        return codes.to_numpy(dtype=np.int64)

    def _window_arrays(self, window: pd.DataFrame):
        time_col = time_column(window)
        t = time_values(window, time_col)
        if len(t) and np.any(np.diff(t) < 0):
            raise ValueError("Cửa sổ phải được sắp theo thời gian")
        if self._last_t is not None and len(t) and t[0] < self._last_t:
            raise ValueError("Các cửa sổ phải đến theo đúng thứ tự thời gian")
        src = self._codes(window["account_id"])
        dst = self._codes(window["counterparty_id"])
        x = window[self.feature_columns].to_numpy(dtype=np.float32)
        return time_col, t, src, dst, x

    # ------------------------------------------------------------------ dựng đồ thị
    def window_graphs(self, window: pd.DataFrame) -> list[BlockGraph]:
        """Đồ thị quá khứ cho từng khối của `window` (không thay đổi bộ đệm)."""
        time_col, t, src, dst, x = self._window_arrays(window)
        lookback = lookback_units(self.lookback_hours, time_col)
        all_t = np.concatenate([self._t, t])
        all_src = np.concatenate([self._src, src])
        all_dst = np.concatenate([self._dst, dst])
        all_x = np.concatenate([self._x, x])
        n_ids = len(self._ids)
        if self.arch == "edge":
            return self._edge_window_graphs(t, src, dst, all_t, all_src, all_dst, all_x,
                                            lookback, n_ids)
        graphs = []
        for start in range(0, len(window), self.block):
            stop = min(start + self.block, len(window))
            tau = t[start]
            lo = int(np.searchsorted(all_t, tau - lookback, side="left"))
            hi = int(np.searchsorted(all_t, tau, side="left"))
            graphs.append(self._block_graph(
                start, stop, src[start:stop], all_src[lo:hi], all_dst[lo:hi], all_x[lo:hi],
                n_ids))
        return graphs

    def _block_graph(self, start, stop, senders, src, dst, x, n_ids) -> BlockGraph:
        targets = np.unique(senders)
        visited = np.zeros(n_ids, dtype=bool)
        visited[targets] = True
        frontier = visited.copy()
        edge_sel = np.zeros(len(src), dtype=bool)
        for _ in range(self.num_hops):
            into = frontier[dst]
            if not into.any():
                break
            edge_sel |= into
            reached = np.unique(src[into])
            reached = reached[~visited[reached]]
            frontier = np.zeros(n_ids, dtype=bool)
            frontier[reached] = True
            visited[reached] = True
        nodes = np.flatnonzero(visited)
        local = np.full(n_ids, -1, dtype=np.int64)
        local[nodes] = np.arange(len(nodes))

        # Đặc trưng nút: trung bình các giao dịch nút đó đã gửi trong khoảng quá khứ.
        # (index_add_ thay cho np.add.at: nhanh hơn hàng chục lần với vài trăm nghìn cạnh.)
        sent = visited[src]
        owner = torch.as_tensor(local[src[sent]])
        feats = torch.zeros((len(nodes), x.shape[1]), dtype=torch.float64).index_add_(
            0, owner, torch.as_tensor(x[sent], dtype=torch.float64))
        counts = torch.zeros(len(nodes), dtype=torch.float64).index_add_(
            0, owner, torch.ones(len(owner), dtype=torch.float64))
        feats = feats / counts.clamp_min(1.0).unsqueeze(-1)

        edge_index = np.stack([local[src[edge_sel]], local[dst[edge_sel]]])
        return BlockGraph(
            start=start, stop=stop,
            node_features=feats.to(torch.float32),
            edge_index=torch.as_tensor(edge_index, dtype=torch.long),
            row_node=torch.as_tensor(local[senders], dtype=torch.long),
        )

    # ------------------------------------------------------------------ chế độ edge
    def _edge_window_graphs(self, t, src, dst, all_t, all_src, all_dst, all_x, lookback,
                            n_ids) -> list[BlockGraph]:
        # Số thứ tự của mỗi cạnh trong chuỗi cạnh ra của người gửi / cạnh vào của người
        # nhận; trừ đi từ số cạnh nút đó có trước τ thì ra thứ hạng "gần nhất" (0).
        seq_out, seq_in = _group_seq(all_src), _group_seq(all_dst)
        n_out = np.zeros(n_ids, dtype=np.int64)
        n_in = np.zeros(n_ids, dtype=np.int64)
        # Hồ sơ nút trên khoảng [lo, hi) được CỘNG DỒN: mỗi khối chỉ cộng các cạnh mới
        # và trừ các cạnh vừa quá hạn, thay vì quét lại cả 24 giờ cạnh.
        n_feat = all_x.shape[1]
        sums = {side: torch.zeros((n_ids, n_feat), dtype=torch.float64) for side in ("out", "in")}
        cnts = {side: torch.zeros(n_ids, dtype=torch.float64) for side in ("out", "in")}
        x64 = torch.as_tensor(all_x, dtype=torch.float64)
        owners = {"out": torch.as_tensor(all_src), "in": torch.as_tensor(all_dst)}

        def shift(a: int, b: int, sign: float) -> None:
            if b <= a:
                return
            vals = x64[a:b] if sign > 0 else -x64[a:b]
            ones = torch.full((b - a,), sign, dtype=torch.float64)
            for side in ("out", "in"):
                sums[side].index_add_(0, owners[side][a:b], vals)
                cnts[side].index_add_(0, owners[side][a:b], ones)

        done = cur_lo = 0
        graphs = []
        for start in range(0, len(t), self.block):
            stop = min(start + self.block, len(t))
            tau = t[start]
            lo = int(np.searchsorted(all_t, tau - lookback, side="left"))
            hi = int(np.searchsorted(all_t, tau, side="left"))
            n_out += np.bincount(all_src[done:hi], minlength=n_ids)
            n_in += np.bincount(all_dst[done:hi], minlength=n_ids)
            shift(done, hi, 1.0)
            shift(cur_lo, lo, -1.0)
            done, cur_lo = hi, lo
            sl = slice(lo, hi)
            s, d = all_src[sl], all_dst[sl]
            kept = ((n_out[s] - 1 - seq_out[sl] < self.fanout)
                    & (n_in[d] - 1 - seq_in[sl] < self.fanout))
            age = (tau - all_t[sl]).astype(np.float64) / float(lookback)
            graphs.append(self._edge_block_graph(
                start, stop, src[start:stop], dst[start:stop], s, d, all_x[sl], age, kept,
                n_ids, sums, cnts))
        return graphs

    def _edge_block_graph(self, start, stop, senders, receivers, src, dst, x, age, kept,
                          n_ids, sums, cnts) -> BlockGraph:
        targets = np.unique(np.concatenate([senders, receivers]))
        visited = np.zeros(n_ids, dtype=bool)
        visited[targets] = True
        frontier = visited.copy()
        edge_sel = np.zeros(len(src), dtype=bool)
        for _ in range(self.num_hops):
            touch = kept & (frontier[src] | frontier[dst]) & ~edge_sel
            if not touch.any():
                break
            edge_sel |= touch
            reached = np.unique(np.concatenate([src[touch], dst[touch]]))
            reached = reached[~visited[reached]]
            frontier = np.zeros(n_ids, dtype=bool)
            frontier[reached] = True
            visited[reached] = True
        nodes = np.flatnonzero(visited)
        local = np.full(n_ids, -1, dtype=np.int64)
        local[nodes] = np.arange(len(nodes))

        node_t = torch.as_tensor(nodes)

        def profile(side: str):
            # Cộng rồi trừ dồn có thể để lại sai số làm tròn cỡ 1e-12 -> làm tròn số đếm.
            count = cnts[side][node_t].round().clamp_min(0.0)
            total = sums[side][node_t]
            mean = torch.where(count.unsqueeze(-1) > 0,
                               total / count.clamp_min(1.0).unsqueeze(-1), 0.0)
            return mean.to(torch.float32), count

        out_mean, out_count = profile("out")
        in_mean, in_count = profile("in")
        edge_index = np.stack([local[src[edge_sel]], local[dst[edge_sel]]])
        return BlockGraph(
            start=start, stop=stop, node_features=out_mean,
            edge_index=torch.as_tensor(edge_index, dtype=torch.long),
            row_node=torch.as_tensor(local[senders], dtype=torch.long),
            row_cp_node=torch.as_tensor(local[receivers], dtype=torch.long),
            in_features=in_mean,
            counts=torch.stack([out_count, in_count], dim=-1).to(torch.float32),
            edge_x=torch.as_tensor(x[edge_sel], dtype=torch.float32),
            edge_age=torch.as_tensor(age[edge_sel], dtype=torch.float32),
        )

    def append(self, window: pd.DataFrame) -> None:
        """Đưa các giao dịch của `window` vào bộ đệm và bỏ phần quá cũ."""
        time_col, t, src, dst, x = self._window_arrays(window)
        if not len(t):
            return
        lookback = lookback_units(self.lookback_hours, time_col)
        self._t = np.concatenate([self._t, t])
        self._src = np.concatenate([self._src, src])
        self._dst = np.concatenate([self._dst, dst])
        self._x = np.concatenate([self._x, x])
        self._last_t = int(t[-1])
        keep = int(np.searchsorted(self._t, self._last_t - lookback, side="left"))
        if keep:
            self._t, self._src = self._t[keep:], self._src[keep:]
            self._dst, self._x = self._dst[keep:], self._x[keep:]


def encode_rows(model, graphs: list[BlockGraph], n_rows: int, device: torch.device) -> torch.Tensor:
    """Embedding đồ thị cho TỪNG dòng, theo thứ tự dòng của cửa sổ.

    Đồ thị "sage": embedding nút người gửi. Đồ thị "edge": mô hình tự đọc ra theo cạnh
    (DRREv2.encode_block — ghép embedding người gửi và người nhận).
    """
    parts = []
    for g in graphs:
        if g.row_cp_node is not None:
            parts.append(model.encode_block(g, device))
            continue
        g_all = model.encode_graph(g.node_features.to(device), g.edge_index.to(device))
        parts.append(g_all[g.row_node.to(device)])
    out = torch.cat(parts) if parts else torch.empty(0, device=device)
    if out.shape[0] != n_rows:
        raise RuntimeError(f"encode_rows: {out.shape[0]} dòng, cần {n_rows}")
    return out


def graph_mode(model) -> str:
    return getattr(model, "graph_mode", "window")


def make_buffer(model, feature_columns=None) -> PastGraphBuffer | None:
    """Bộ đệm mới cho mô hình dùng đồ thị quá khứ; None với đồ thị cửa sổ."""
    if graph_mode(model) != "past":
        return None
    return PastGraphBuffer(
        feature_columns if feature_columns is not None else model.feature_columns,
        lookback_hours=model.graph_lookback_hours,
        block=model.graph_block, num_hops=model.graph_encoder.num_layers,
        arch=getattr(model, "graph_arch", "sage"),
        fanout=getattr(model, "graph_fanout", None) or DEFAULT_FANOUT)
