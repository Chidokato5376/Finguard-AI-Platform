"""Graph-based Relationship Encoder (Mục 4.3, dùng dữ liệu từ Module 4).

g_t^c = AGGREGATE({ f_msg(g_u^(l-1)) : u in N(c) }),  l = 1,...,L

Sau L lop, g_t^c ma hoa cau truc lan can L-hop -- du de nhan dien mot
tai khoan Money Mule du ban than giao dich hien tai trong binh thuong.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn

try:
    from torch_geometric.nn import GATConv, SAGEConv
except ImportError:  # pragma: no cover — cho phép import module khi chưa cài PyG
    SAGEConv = None
    GATConv = None


class GraphRelationshipEncoder(nn.Module):
    """Wrapper quanh GraphSAGE hoặc GAT (PyTorch Geometric).

    Lưu ý vận hành (docs/architecture.md §"Ràng buộc vận hành"): L-hop
    message passing trên đồ thị giao dịch quy mô ngân hàng có thể tốn
    latency đáng kể nếu không có subgraph sampling / neighbor caching.
    Chưa implement caching trong scaffold này — cần đo trước khi claim
    "real-time" trong bản pitch.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        encoder_type: Literal["GraphSAGE", "GAT"] = "GraphSAGE",
    ) -> None:
        super().__init__()
        if SAGEConv is None:
            raise ImportError(
                "torch-geometric chưa được cài. `pip install torch-geometric` "
                "(xem requirements.txt)."
            )
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            if encoder_type == "GraphSAGE":
                self.layers.append(SAGEConv(in_dim, hidden_dim))
            elif encoder_type == "GAT":
                self.layers.append(GATConv(in_dim, hidden_dim))
            else:
                raise ValueError(f"encoder_type không hỗ trợ: {encoder_type}")

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (num_nodes, input_dim) — đặc trưng node (tài khoản).
            edge_index: (2, num_edges) — cạnh đồ thị giao dịch (COO format).

        Returns:
            g: (num_nodes, hidden_dim) — biểu diễn quan hệ mạng lưới sau L lớp.
        """
        h = x
        for layer in self.layers:
            h = torch.relu(layer(h, edge_index))
        return h
