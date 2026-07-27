"""Dynamic Risk Representation Engine — lắp ráp toàn bộ Module 3′.

Luồng: Behavior Memory -> Context-aware Attention  ┐
       Graph Encoder ────────────────────────────── ┴─> Adaptive Fusion -> e_t^c
"""

from __future__ import annotations

import torch
from torch import nn

from src.drre.attention import ContextAwareAttention
from src.drre.behavior_memory import BehaviorMemory
from src.drre.fusion import AdaptiveRiskFusion
from src.drre.graph_encoder import GraphRelationshipEncoder


class DRRE(nn.Module):
    """Dynamic Risk Representation Engine (Module 3′).

    Ghi chú: nhánh Quantum-enhanced Embedding (Mục 4.5) KHÔNG được lắp
    trong forward() mặc định — xem src/quantum/quantum_kernel.py và
    config.yaml: quantum.enabled. Chỉ bật sau khi benchmark AUPRC xác
    nhận cải thiện thực sự (docs/architecture.md).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_slots: int = 4,
        graph_hidden_dim: int = 64,
        graph_num_layers: int = 2,
        embedding_dim: int = 128,
    ) -> None:
        super().__init__()
        self.behavior_memory = BehaviorMemory(input_dim, hidden_dim, num_slots)
        self.attention = ContextAwareAttention(input_dim, key_dim=hidden_dim)
        self.graph_encoder = GraphRelationshipEncoder(
            input_dim, graph_hidden_dim, graph_num_layers
        )
        self.fusion = AdaptiveRiskFusion(
            x_dim=input_dim, h_dim=hidden_dim, g_dim=graph_hidden_dim,
            embedding_dim=embedding_dim,
        )
        # Đầu dự đoán tự giám sát cho L_pred (Mục 4.6)
        self.predictor = nn.Linear(embedding_dim, embedding_dim)

    def forward(
        self,
        x_t: torch.Tensor,
        h_prev: torch.Tensor,
        m_prev: torch.Tensor,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        node_index: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Một bước forward cho một batch giao dịch.

        Args:
            x_t: (batch, input_dim) giao dịch hiện tại.
            h_prev, m_prev: trạng thái Behavior Memory trước đó.
            node_features, edge_index: đồ thị giao dịch hiện tại (toàn cục).
            node_index: (batch,) — chỉ số node tương ứng với từng account_id trong batch,
                dùng để lấy đúng hàng embedding từ Graph Encoder.

        Returns:
            dict gồm: e_t (Risk Embedding), alpha_k, beta, delta_t, e_pred
            (dùng cho L_pred), h_t, m_t (trạng thái mới để lưu lại).
        """
        h_t, m_t = self.behavior_memory(x_t, h_prev, m_prev)
        h_tilde, alpha_k, delta_t = self.attention(x_t, m_t)

        g_all = self.graph_encoder(node_features, edge_index)  # (num_nodes, graph_hidden_dim)
        g_t = g_all[node_index]  # (batch, graph_hidden_dim)

        e_t, beta = self.fusion(x_t, h_tilde, g_t)
        e_pred = self.predictor(e_t)  # dùng cho L_pred so với target network ở bước t+1

        return {
            "e_t": e_t,
            "e_pred": e_pred,
            "alpha_k": alpha_k,
            "beta": beta,
            "delta_t": delta_t,
            "h_t": h_t,
            "m_t": m_t,
        }


def main() -> None:
    """CLI entry point — training loop thật nằm ở src/drre/train.py
    (tách riêng khỏi model.py để giữ file này thuần về kiến trúc).
    Giữ hàm này để `python -m src.drre.model` vẫn hoạt động như README
    từng mô tả, chuyển tiếp toàn bộ tham số sang src.drre.train.main().
    """
    from src.drre.train import main as train_main

    train_main()


if __name__ == "__main__":
    main()
