"""Unit tests cho src/qape/qubo.py và src/qape/solvers.py."""

import pytest

from src.qape.qubo import Alert
from src.qape.solvers import solve_greedy


def _sample_alerts() -> list[Alert]:
    return [
        Alert(alert_id="a1", risk_score=95, cost=5, urgency=1.5),  # critical, phải chọn
        Alert(alert_id="a2", risk_score=40, cost=3, urgency=1.0),
        Alert(alert_id="a3", risk_score=60, cost=4, urgency=1.2),
        Alert(alert_id="a4", risk_score=20, cost=2, urgency=0.8),
    ]


def test_greedy_always_includes_critical_alerts() -> None:
    alerts = _sample_alerts()
    result = solve_greedy(alerts, budget=10.0)
    assert "a1" in result.selected_alert_ids  # r_i >= 90 -> bắt buộc


def test_greedy_respects_budget_when_feasible() -> None:
    alerts = _sample_alerts()
    result = solve_greedy(alerts, budget=100.0)
    assert not result.constraint_violated


def test_alert_is_critical_threshold() -> None:
    alert = Alert(alert_id="x", risk_score=90, cost=1, urgency=1.0)
    assert alert.is_critical
    alert_below = Alert(alert_id="y", risk_score=89.9, cost=1, urgency=1.0)
    assert not alert_below.is_critical


def test_ilp_exact_infeasible_when_critical_cost_exceeds_budget() -> None:
    """Khi tổng chi phí các cảnh báo Critical (bắt buộc x_i=1) vượt budget,
    bài toán không có nghiệm khả thi — status phải là 'infeasible', KHÔNG
    được trả về objective/selection vô nghĩa (lỗi đã phát hiện qua chạy
    thử thực tế, xem ghi chú trong src/qape/solvers.py).
    """
    from src.qape.solvers import solve_ilp_exact

    alerts = [Alert(alert_id=str(i), risk_score=95.0, cost=10.0, urgency=1.0) for i in range(13)]
    result = solve_ilp_exact(alerts, budget=50.0)
    assert result.status == "infeasible"
    assert result.selected_alert_ids == []
    assert result.constraint_violated is True


def test_ilp_exact_optimal_when_feasible() -> None:
    from src.qape.solvers import solve_ilp_exact

    alerts = [Alert(alert_id=str(i), risk_score=95.0, cost=1.0, urgency=1.0) for i in range(3)] + [
        Alert(alert_id="low", risk_score=40.0, cost=100.0, urgency=1.0)
    ]
    result = solve_ilp_exact(alerts, budget=10.0)
    assert result.status == "optimal"
    assert result.objective_value == pytest.approx(285.0)
    assert set(result.selected_alert_ids) == {"0", "1", "2"}
