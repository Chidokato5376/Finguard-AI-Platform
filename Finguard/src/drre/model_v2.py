"""DRRE v2 — tầng chấm điểm đã sửa (Giai đoạn 2A, docs/lo-trinh-train-dot-2.md).

Sửa bốn lỗi của v1 ở tầng chấm điểm, giữ nguyên Behavior Memory, Attention và
Graph Encoder (nhánh đồ thị được sửa riêng ở Giai đoạn 2B):

  Lỗi 7  Đầu vào chưa chuẩn hóa; kênh đi vào như một số 0–6.
         -> chuẩn hóa theo trung bình/độ lệch chuẩn học trên train (buffer, nằm
            trong state_dict nên đi cùng checkpoint); kênh qua embedding 8 chiều.
  Lỗi 3  Ba nhánh dùng chung đầu vào nên cổng sụp về một nhánh.
         -> mỗi nhánh nhận một NHÓM cột riêng (BRANCH_OF_* bên dưới).
  Lỗi 2  φ và đầu chấm điểm tuyến tính, hợp nhất chỉ là tổng có trọng số.
         -> φ là MLP 2 lớp; hợp nhất bằng ghép nối [φ₁‖φ₂‖φ₃‖φ₁⊙φ₃‖δ_t] rồi MLP.
            β không còn nằm trên đường tính điểm, chỉ dùng để giải thích.
  Lỗi 4  δ_t tính ra nhưng không vào điểm.
         -> δ_t và |q_t − h̃_t| đi vào nhánh thói quen, δ_t vào cả bộ hợp nhất.

Nhánh đồ thị "edge" (Giai đoạn 2B2–2B5, `graph_arch="edge"`, chỉ dùng với đồ thị quá
khứ src/drre/past_graph.py): truyền tin HAI CHIỀU với hai bộ trọng số, cạnh mang đặc
trưng (số tiền, kênh, độ cũ), nút có hồ sơ cả phía gửi lẫn phía nhận, và đọc ra THEO
CẠNH: φ₃ nhận [g_người gửi ‖ g_người nhận]; cùng x_t (φ₁), h̃_t và δ_t (φ₂) trong bộ hợp
nhất, đó chính là MLP([x_t ‖ g_gửi ‖ g_nhận ‖ h̃_gửi ‖ δ_t]) của lộ trình.

Giao diện giống hệt DRRE v1 (cùng chữ ký forward, cùng các khóa đầu ra, cùng
thuộc tính behavior_memory / attention / graph_encoder / fusion), nên vòng huấn
luyện, BehaviorStateStore và mọi đường chấm điểm dùng lại được nguyên vẹn.
Đầu vào x_t vẫn là các cột THÔ theo `feature_columns`; chuẩn hóa nằm trong mô hình.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import nn

from src.drre.attention import ContextAwareAttention
from src.drre.behavior_memory import BehaviorMemory
from src.drre.graph_encoder import GraphRelationshipEncoder

CHANNEL_COLUMN = "channel_encoded"
#: Giá trị sau chuẩn hóa bị chặn trong [-CLIP, CLIP]: một vài số tiền cực lớn
#: không được phép chi phối cả lô.
CLIP = 8.0

#: Nhánh của 20 đặc trưng gốc. "both" = mô tả chính sự kiện (số tiền), cần cho
#: cả nhánh giao dịch lẫn bộ nhớ thói quen (GRU phải biết đã xảy ra gì).
BRANCH_OF_BASE = {
    "amount_norm": "both",
    "z_amount": "transaction", "log_amount_ratio": "transaction",
    "new_counterparty": "transaction", "is_round_amount": "transaction",
    "hour_sin": "transaction", "hour_cos": "transaction", "dow": "transaction",
    "log_n_prior": "behavior", "log_gap": "behavior",
    "log_amt_out_prior": "behavior", "log_amt_in_prior": "behavior",
    "log_n_in_prior": "behavior", "passthrough_ratio": "behavior", "flow_imbalance": "behavior",
    "log_n_distinct_cp": "network", "log_n_prior_as_cp": "network",
    "log_n_distinct_senders": "network", "log_cp_amt_in_prior": "network",
}
#: Nhánh mặc định của từng mô-đun kịch bản (src/data/scenarios).
BRANCH_OF_MODULE = {
    "fan_out": "network", "fan_in": "network", "cycle": "network", "synchronized": "network",
    "hub": "network", "gather_scatter": "behavior", "mule_chain": "behavior",
    "cash_structuring": "behavior", "integration": "behavior", "temporal_windows": "behavior",
    "cross_bank_fx": "transaction",
}
#: Ngoại lệ trong một mô-đun: cột mô tả CHÍNH giao dịch này, hoặc mô tả đối tác.
BRANCH_OVERRIDES = {
    "mc_forward_ratio_24h": "transaction", "mc_amt_sim_last_in": "transaction",
    "cb_log_n_banks_7d": "network",
    "in_large_first_time": "transaction", "in_log_ratio_prior_max": "transaction",
    "fo_amt_dev_24h": "transaction", "cs_cash_ratio_ctr": "transaction",
}
BRANCHES = ("transaction", "behavior", "network")
#: Cột thô của giao dịch đi vào đặc trưng cạnh (ngoài kênh và độ cũ) ở graph_arch="edge".
EDGE_COLUMNS = ("amount_norm",)
GRAPH_ARCHES = ("sage", "edge")


def branch_of(column: str) -> str:
    """Nhánh nhận cột `column`; cột không có trong bảng thì vào nhánh giao dịch."""
    if column in BRANCH_OVERRIDES:
        return BRANCH_OVERRIDES[column]
    if column in BRANCH_OF_BASE:
        return BRANCH_OF_BASE[column]
    from src.data.scenarios.store import module_of

    module = module_of(column)
    return BRANCH_OF_MODULE.get(module, "transaction") if module else "transaction"


def branch_groups(feature_columns: Sequence[str]) -> dict[str, list[int]]:
    """Chỉ số cột (không gồm cột kênh) của từng nhánh."""
    groups: dict[str, list[int]] = {b: [] for b in BRANCHES}
    for i, column in enumerate(feature_columns):
        if column == CHANNEL_COLUMN:
            continue
        branch = branch_of(column)
        for target in (("transaction", "behavior") if branch == "both" else (branch,)):
            groups[target].append(i)
    return groups


class MLP(nn.Module):
    """Linear -> ReLU -> Linear. `out_features` giống nn.Linear để mã cũ đọc được."""

    def __init__(self, in_features: int, hidden: int, out_features: int) -> None:
        super().__init__()
        self.out_features = out_features
        self.net = nn.Sequential(nn.Linear(in_features, hidden), nn.ReLU(), nn.Linear(hidden, out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _mean_into(values: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    """Trung bình các hàng `values` theo nút đích `index` (nút không có cạnh -> 0)."""
    total = values.new_zeros((n, values.shape[-1])).index_add(0, index, values)
    count = values.new_zeros(n).index_add(0, index, values.new_ones(len(index)))
    return total / count.clamp_min(1.0).unsqueeze(-1)


class BiEdgeLayer(nn.Module):
    """Một lớp truyền tin hai chiều có đặc trưng cạnh (kiểu GINE, gộp trung bình).

    Mỗi cạnh u→v (u trả tiền cho v) gửi hai thông điệp với hai bộ trọng số riêng:
    v nhận ψ_vào([h_u ‖ e]) ("ai đã trả cho tôi"), u nhận ψ_ra([h_v ‖ e]) ("tôi đã trả cho ai").
    """

    def __init__(self, dim: int, edge_dim: int) -> None:
        super().__init__()
        self.msg_in = nn.Linear(dim + edge_dim, dim)
        self.msg_out = nn.Linear(dim + edge_dim, dim)
        self.self_lin = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        n = h.shape[0]
        agg_in = _mean_into(torch.relu(self.msg_in(torch.cat([h[src], e], dim=-1))), dst, n)
        agg_out = _mean_into(torch.relu(self.msg_out(torch.cat([h[dst], e], dim=-1))), src, n)
        return h + torch.relu(self.norm(self.self_lin(h) + agg_in + agg_out))


class BiEdgeGraphEncoder(nn.Module):
    """Chiếu nút vào `hidden` rồi `num_layers` lớp BiEdgeLayer."""

    def __init__(self, node_dim: int, edge_dim: int, hidden: int, num_layers: int) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.input = nn.Linear(node_dim, hidden)
        self.layers = nn.ModuleList(BiEdgeLayer(hidden, edge_dim) for _ in range(num_layers))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.input(x))
        for layer in self.layers:
            h = layer(h, edge_index, e)
        return h


class ConcatFusion(nn.Module):
    """e_t = MLP([φ₁(giao dịch) ‖ φ₂(thói quen) ‖ φ₃(mạng lưới) ‖ φ₁⊙φ₃ ‖ δ_t]).

    β (trả về để giải thích ở Module 6) là tỷ lệ độ lớn ‖φ_i‖ của từng nhánh. Nó
    KHÔNG tham gia tính điểm, nên không thể "sụp" như cổng softmax của v1.
    """

    BRANCHES = ("transaction", "behavior", "graph")

    def __init__(self, tx_dim: int, behavior_dim: int, g_dim: int, embedding_dim: int,
                 branch_dim: int) -> None:
        super().__init__()
        self.phi_1 = MLP(tx_dim, branch_dim, branch_dim)
        self.phi_2 = MLP(behavior_dim, branch_dim, branch_dim)
        self.phi_3 = MLP(g_dim, branch_dim, branch_dim)
        self.mixer = MLP(4 * branch_dim + 1, embedding_dim, embedding_dim)
        self.register_buffer("branch_mask", torch.ones(3), persistent=False)

    def set_disabled_branches(self, names) -> None:
        """Tắt nhánh (ablation): đầu ra φ của nhánh đó luôn bằng 0."""
        mask = torch.ones(3, device=self.branch_mask.device)
        for name in names:
            mask[self.BRANCHES.index(name)] = 0.0
        if mask.sum() == 0:
            raise ValueError("Không thể tắt cả ba nhánh.")
        self.branch_mask = mask

    def forward(self, x_tx: torch.Tensor, behavior: torch.Tensor, g_t: torch.Tensor,
                delta_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        p1 = self.phi_1(x_tx) * self.branch_mask[0]
        p2 = self.phi_2(behavior) * self.branch_mask[1]
        p3 = self.phi_3(g_t) * self.branch_mask[2]
        e_t = self.mixer(torch.cat([p1, p2, p3, p1 * p3, delta_t.unsqueeze(-1)], dim=-1))
        with torch.no_grad():
            norms = torch.stack([p.norm(dim=-1) for p in (p1, p2, p3)], dim=-1)
            beta = norms / norms.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return e_t, beta


class DRREv2(nn.Module):
    """DRRE với tầng chấm điểm đã sửa. Xem docstring của module."""

    def __init__(
        self,
        input_dim: int,
        feature_columns: Sequence[str],
        hidden_dim: int = 64,
        num_slots: int = 4,
        graph_hidden_dim: int = 64,
        graph_num_layers: int = 2,
        embedding_dim: int = 128,
        n_channels: int = 16,
        channel_dim: int = 8,
        branch_dim: int = 64,
        graph_arch: str = "sage",
    ) -> None:
        super().__init__()
        if graph_arch not in GRAPH_ARCHES:
            raise ValueError(f"graph_arch phải thuộc {GRAPH_ARCHES}, nhận {graph_arch!r}")
        feature_columns = list(feature_columns)
        self.graph_arch = graph_arch
        if len(feature_columns) != input_dim:
            raise ValueError(f"input_dim={input_dim} nhưng có {len(feature_columns)} cột.")
        self.embedding_dim = embedding_dim
        self.n_channels = n_channels
        groups = branch_groups(feature_columns)
        if not groups["network"]:
            raise ValueError("DRRE v2 cần ít nhất một cột cho nhánh mạng lưới.")
        self.channel_index = (feature_columns.index(CHANNEL_COLUMN)
                              if CHANNEL_COLUMN in feature_columns else -1)
        ch = channel_dim if self.channel_index >= 0 else 0
        for name, idx in groups.items():
            self.register_buffer(f"idx_{name}", torch.tensor(idx, dtype=torch.long), persistent=False)

        # Thống kê chuẩn hóa: học trên train bằng fit_normalizer, lưu trong state_dict.
        self.register_buffer("feature_mean", torch.zeros(input_dim))
        self.register_buffer("feature_std", torch.ones(input_dim))
        self.channel_embedding = nn.Embedding(n_channels, channel_dim) if ch else None

        tx_dim = len(groups["transaction"]) + ch
        behavior_in = len(groups["behavior"]) + ch
        self.behavior_memory = BehaviorMemory(behavior_in, hidden_dim, num_slots)
        self.attention = ContextAwareAttention(behavior_in, key_dim=hidden_dim)
        if graph_arch == "edge":
            # Hồ sơ phía nhận: cột mạng lưới + các cột mô tả giao dịch (số tiền).
            edge_idx = [feature_columns.index(c) for c in EDGE_COLUMNS if c in feature_columns]
            in_idx = list(groups["network"]) + [i for i in edge_idx if i not in groups["network"]]
            self.register_buffer("idx_in", torch.tensor(in_idx, dtype=torch.long), persistent=False)
            self.register_buffer("idx_edge", torch.tensor(edge_idx, dtype=torch.long),
                                 persistent=False)
            node_dim = len(groups["network"]) + len(in_idx) + 2
            edge_dim = len(edge_idx) + ch + 1
            self.graph_encoder = BiEdgeGraphEncoder(node_dim, edge_dim, graph_hidden_dim,
                                                    graph_num_layers)
            g_dim = 2 * graph_hidden_dim  # đọc theo cạnh: người gửi ‖ người nhận
        else:
            self.graph_encoder = GraphRelationshipEncoder(
                len(groups["network"]), graph_hidden_dim, graph_num_layers)
            g_dim = graph_hidden_dim
        # Nhánh thói quen nhận h̃_t, |q_t − h̃_t| (lệch so với thói quen, theo từng chiều) và δ_t.
        self.fusion = ConcatFusion(tx_dim, 2 * hidden_dim + 1, g_dim, embedding_dim, branch_dim)
        self.predictor = nn.Linear(embedding_dim, embedding_dim)

    @torch.no_grad()
    def fit_normalizer(self, values: np.ndarray) -> None:
        """Học trung bình/độ lệch chuẩn từ ma trận đặc trưng THÔ của tập train."""
        values = np.asarray(values, dtype=np.float64)
        mean = np.nanmean(values, axis=0)
        std = np.nanstd(values, axis=0)
        std = np.where(np.isfinite(std) & (std > 1e-6), std, 1.0)
        mean = np.where(np.isfinite(mean), mean, 0.0)
        if self.channel_index >= 0:
            mean[self.channel_index], std[self.channel_index] = 0.0, 1.0
        self.feature_mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        self.feature_std.copy_(torch.as_tensor(std, dtype=torch.float32))

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return ((x - self.feature_mean) / self.feature_std).clamp(-CLIP, CLIP)

    def _branch_inputs(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self._normalize(x)
        tx, behavior = z[:, self.idx_transaction], z[:, self.idx_behavior]
        if self.channel_embedding is not None:
            code = x[:, self.channel_index].round().long().clamp(0, self.n_channels - 1)
            emb = self.channel_embedding(code)
            tx, behavior = torch.cat([tx, emb], dim=-1), torch.cat([behavior, emb], dim=-1)
        return tx, behavior

    def encode_graph(self, node_features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Embedding nút từ các cột nhánh mạng lưới.

        node_features là trung bình đặc trưng THÔ của các giao dịch tài khoản gửi
        (graph_utils.build_account_graph); chuẩn hóa là phép affine nên chuẩn hóa
        sau khi lấy trung bình cho cùng kết quả. Nút chỉ-nhận có vector 0 thô và
        được giữ ở 0 (nhánh đồ thị được làm lại ở Giai đoạn 2B).
        """
        if self.graph_arch == "edge":
            raise RuntimeError("graph_arch='edge' chỉ chạy với đồ thị quá khứ (encode_block)")
        only_receiver = (node_features == 0).all(dim=-1, keepdim=True)
        z = self._normalize(node_features)[:, self.idx_network].masked_fill(only_receiver, 0.0)
        return self.graph_encoder(z, edge_index)

    def encode_block(self, g, device: torch.device) -> torch.Tensor:
        """[g_người gửi ‖ g_người nhận] cho từng dòng của một khối (graph_arch="edge").

        `g` là past_graph.BlockGraph dựng với arch="edge"."""
        counts = g.counts.to(device)
        has_out, has_in = counts[:, :1] > 0, counts[:, 1:] > 0
        z_out = self._normalize(g.node_features.to(device))[:, self.idx_network] * has_out
        z_in = self._normalize(g.in_features.to(device))[:, self.idx_in] * has_in
        nodes = torch.cat([z_out, z_in, torch.log1p(counts)], dim=-1)
        edge_x = g.edge_x.to(device)
        parts = [self._normalize(edge_x)[:, self.idx_edge]]
        if self.channel_embedding is not None:
            code = edge_x[:, self.channel_index].round().long().clamp(0, self.n_channels - 1)
            parts.append(self.channel_embedding(code))
        parts.append(g.edge_age.to(device).unsqueeze(-1))
        h = self.graph_encoder(nodes, g.edge_index.to(device), torch.cat(parts, dim=-1))
        return torch.cat([h[g.row_node.to(device)], h[g.row_cp_node.to(device)]], dim=-1)

    def forward(
        self,
        x_t: torch.Tensor,
        h_prev: torch.Tensor,
        m_prev: torch.Tensor,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        node_index: torch.Tensor,
        g_all: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        tx, behavior_in = self._branch_inputs(x_t)
        h_t, m_t = self.behavior_memory(behavior_in, h_prev, m_prev)
        h_tilde, alpha_k, delta_t = self.attention(behavior_in, m_t)
        deviation = (self.attention.w_q(behavior_in) - h_tilde).abs()
        if g_all is None:
            g_all = self.encode_graph(node_features, edge_index)
        g_t = g_all[node_index]
        behavior = torch.cat([h_tilde, deviation, delta_t.unsqueeze(-1)], dim=-1)
        e_t, beta = self.fusion(tx, behavior, g_t, delta_t)
        return {
            "e_t": e_t,
            "e_pred": self.predictor(e_t),
            "alpha_k": alpha_k,
            "beta": beta,
            "delta_t": delta_t,
            "h_t": h_t,
            "m_t": m_t,
        }
