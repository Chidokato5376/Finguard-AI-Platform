"""Kiểm thử chế độ tắt đồ thị (--no-graph) và tắt nhánh (--disable-branch)."""

import pandas as pd
import pytest
import torch

from src.dashboard.graph_utils import build_account_graph
from src.drre.model import DRRE

FEATURES = ["amount_norm", "channel_encoded"]


def _df_without_counterparty() -> pd.DataFrame:
    return pd.DataFrame({
        "account_id": ["a", "a", "b"],
        "counterparty_id": [None, None, None],
        "amount_norm": [1.0, 2.0, 3.0],
        "channel_encoded": [0, 1, 0],
    })


def _model() -> DRRE:
    return DRRE(input_dim=2, hidden_dim=8, num_slots=2, graph_hidden_dim=8,
                graph_num_layers=2, embedding_dim=16)


def test_no_graph_mode_accepts_missing_counterparty():
    nf, ei, idx = build_account_graph(_df_without_counterparty(), FEATURES, use_graph=False)
    assert nf.shape == (2, 2) and torch.count_nonzero(nf) == 0
    assert ei.shape == (2, 0)
    assert set(idx) == {"a", "b"}


def test_graph_mode_still_rejects_missing_counterparty():
    with pytest.raises(ValueError):
        build_account_graph(_df_without_counterparty(), FEATURES)


def test_disabled_branches_get_zero_gate_and_model_runs_without_edges():
    model = _model()
    model.fusion.set_disabled_branches({"behavior", "graph"})
    nf, ei, idx = build_account_graph(_df_without_counterparty(), FEATURES, use_graph=False)
    h, m = model.behavior_memory.init_state(1, torch.device("cpu"))
    out = model(torch.randn(1, 2), h, m, nf, ei, torch.tensor([idx["a"]]))
    assert torch.allclose(out["beta"][0, 1:], torch.zeros(2))
    assert torch.isclose(out["beta"][0, 0], torch.tensor(1.0))
    out["e_t"].sum().backward()


def test_cannot_disable_all_branches():
    with pytest.raises(ValueError):
        _model().fusion.set_disabled_branches({"transaction", "behavior", "graph"})
