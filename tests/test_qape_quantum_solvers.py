"""Test cho src/qape/solvers.py::solve_simulated_annealing, solve_qaoa,
và cơ chế repair ràng buộc Critical (_repair_critical_constraint).

QAOA test dùng quy mô CỰC NHỎ (3 cảnh báo, p_layers=1, maxiter thấp) để
giữ thời gian chạy CI hợp lý (~2s đo thực tế) — vẫn là mạch lượng tử
THẬT chạy qua StatevectorSampler, không phải mock.
"""

from __future__ import annotations

from src.qape.qubo import Alert
from src.qape.solvers import (
    _repair_critical_constraint,
    solve_qaoa,
    solve_simulated_annealing,
)


def _critical_budget_conflict_alerts() -> list[Alert]:
    """Kịch bản đã phát hiện qua kiểm thử thực tế: với lambda_1 mặc định,
    penalty method của SA có thể BỎ SÓT cảnh báo Critical (r_i>=90) khi
    tối ưu hóa QUBO đã penalize — đây chính là bộ dữ liệu tái tạo lại
    tình huống đó, dùng làm test hồi quy cho bước repair.
    """
    return [
        Alert(alert_id=f"a{i}", risk_score=float(50 + i * 3), cost=float(2 + i % 5), urgency=1.0)
        for i in range(15)
    ]


# --------------------------------------------------------------------------
# _repair_critical_constraint — unit test trực tiếp, không cần solver
# --------------------------------------------------------------------------

def test_repair_adds_missing_critical_alert() -> None:
    alerts = [
        Alert(alert_id="critical", risk_score=95.0, cost=3.0, urgency=1.0),
        Alert(alert_id="normal", risk_score=40.0, cost=2.0, urgency=1.0),
    ]
    repaired = _repair_critical_constraint(["normal"], alerts, budget=10.0)
    assert "critical" in repaired
    assert "normal" in repaired  # đủ budget, không cần loại bớt


def test_repair_removes_low_value_alerts_to_respect_budget() -> None:
    """Khi thêm Critical vào làm vượt budget, phải loại bớt non-critical
    có tỷ lệ giá trị/chi phí THẤP NHẤT trước — không loại ngẫu nhiên.
    """
    alerts = [
        Alert(alert_id="critical", risk_score=95.0, cost=8.0, urgency=1.0),
        Alert(alert_id="good_value", risk_score=80.0, cost=1.0, urgency=1.0),   # tỷ lệ cao
        Alert(alert_id="bad_value", risk_score=20.0, cost=2.0, urgency=1.0),     # tỷ lệ thấp
    ]
    # budget=9: đủ cho critical(8) + good_value(1) = 9, không đủ cho cả bad_value(2) nữa
    repaired = _repair_critical_constraint(
        ["good_value", "bad_value"], alerts, budget=9.0
    )
    assert "critical" in repaired
    assert "good_value" in repaired
    assert "bad_value" not in repaired  # bị loại vì tỷ lệ giá trị/chi phí thấp nhất


def test_repair_keeps_only_critical_when_budget_too_small() -> None:
    alerts = [
        Alert(alert_id="critical", risk_score=95.0, cost=5.0, urgency=1.0),
        Alert(alert_id="normal", risk_score=80.0, cost=10.0, urgency=1.0),
    ]
    repaired = _repair_critical_constraint(["normal"], alerts, budget=5.0)
    assert repaired == ["critical"]


# --------------------------------------------------------------------------
# solve_simulated_annealing — chạy solver thật (dwave-neal)
# --------------------------------------------------------------------------

def test_sa_always_includes_critical_regardless_of_lambda() -> None:
    """Test hồi quy trực tiếp cho phát hiện qua kiểm thử thực tế: trước
    khi có bước repair, lambda_1=5.0 (mặc định) khiến SA bỏ sót cảnh báo
    Critical. Sau khi vá, phải LUÔN đúng bất kể lambda_1.
    """
    alerts = _critical_budget_conflict_alerts()
    critical_id = next(a.alert_id for a in alerts if a.is_critical)

    for lam in [5.0, 20.0, 50.0, 100.0]:
        result = solve_simulated_annealing(alerts, budget=25.0, lambda_1=lam, seed=42)
        assert critical_id in result.selected_alert_ids, f"Thất bại với lambda_1={lam}"
        assert not result.constraint_violated


def test_sa_respects_budget_when_no_critical_conflict() -> None:
    alerts = [
        Alert(alert_id=f"a{i}", risk_score=40.0, cost=3.0, urgency=1.0) for i in range(10)
    ]
    result = solve_simulated_annealing(alerts, budget=10.0, seed=1)
    used = sum(a.cost for a in alerts if a.alert_id in result.selected_alert_ids)
    assert used <= 10.0
    assert not result.constraint_violated
    assert result.status == "heuristic"


# --------------------------------------------------------------------------
# solve_qaoa — chạy mạch lượng tử THẬT (StatevectorSampler), quy mô nhỏ
# --------------------------------------------------------------------------

def test_qaoa_runs_and_respects_critical_constraint() -> None:
    """Quy mô nhỏ (2 cảnh báo, 4 qubit) để test nhanh trong CI.

    LƯU Ý LỊCH SỬ (bài học thực tế quan trọng, xem QAOA_MAX_QUBITS_DEFAULT
    trong src/qape/solvers.py): nếu test này bỗng dưng chạy RẤT chậm hoặc
    treo, nguyên nhân nhiều khả năng nhất là xung đột phiên bản qiskit
    (vd. `pip install qiskit-machine-learning` không ghim `<0.9` đã âm
    thầm nâng qiskit lên 2.x, phá vỡ qiskit-algorithms==0.4.0) — KHÔNG
    phải lỗi trong code này. Kiểm tra `pip list | grep qiskit` khớp đúng
    requirements.txt trước khi nghi ngờ code.
    """
    alerts = [
        Alert(alert_id="critical", risk_score=95.0, cost=1.0, urgency=1.0),
        Alert(alert_id="normal1", risk_score=40.0, cost=1.0, urgency=1.0),
    ]
    result = solve_qaoa(alerts, budget=2.0, cost_scale=1.0, p_layers=1, maxiter=10, seed=42)

    assert result.status == "heuristic"
    assert "critical" in result.selected_alert_ids
    assert not result.constraint_violated
    assert result.objective_value > 0


def test_qaoa_max_qubits_guard_raises_without_running() -> None:
    """Bảo vệ chống treo máy: vượt max_qubits phải raise NGAY, không chạy
    mô phỏng statevector (tốn bộ nhớ theo cấp số nhân theo số qubit).
    """
    import time

    alerts = [Alert(alert_id=f"a{i}", risk_score=50.0, cost=2.0, urgency=1.0) for i in range(25)]
    start = time.perf_counter()
    try:
        solve_qaoa(alerts, budget=30.0, max_qubits=20)
        assert False, "Kỳ vọng ValueError khi vượt max_qubits"
    except ValueError as e:
        assert "max_qubits" in str(e)
    elapsed = time.perf_counter() - start
    assert elapsed < 2.0, "Phải raise NGAY, không được chạy mô phỏng trước khi kiểm tra"
