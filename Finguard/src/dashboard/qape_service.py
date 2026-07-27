"""QAPE Integration cho Dashboard — chuyển batch cảnh báo đã chấm điểm
thành src.qape.qubo.Alert và chạy solver thật (Greedy + ILP, đã kiểm thử
đầy đủ trong tests/test_qape.py).
"""

from __future__ import annotations

import pandas as pd

from src.qape.qubo import Alert
from src.qape.solvers import SolverResult, solve_greedy, solve_ilp_exact


def build_alerts_from_scored_df(
    df: pd.DataFrame, top_n: int = 30, cost_col: str | None = None
) -> list[Alert]:
    """Chuyển N giao dịch có risk_score cao nhất thành danh sách Alert cho QAPE.

    Args:
        df: DataFrame đã qua scoring_service.compute_heuristic_scores
            (phải có cột risk_score).
        top_n: chỉ lấy N cảnh báo cao nhất — QAOA/ILP simulator chỉ khả
            thi tới ~20-30 alert/lô (xem docs/architecture.md).
        cost_col: cột dùng làm chi phí xử lý c_i. Nếu None, dùng chi phí
            xấp xỉ = 1 + log(amount)/10 (giao dịch giá trị lớn hơn giả
            định tốn nhiều thời gian xác minh hơn — GIẢ ĐỊNH ĐƠN GIẢN
            HÓA, không phải số đo thời gian xử lý thực tế của đội ngũ).

    Returns:
        Danh sách Alert, sắp xếp theo risk_score giảm dần trước khi cắt top_n.
    """
    if "risk_score" not in df.columns:
        raise KeyError("DataFrame cần có cột 'risk_score' — chạy scoring_service trước.")

    top = df.sort_values("risk_score", ascending=False).head(top_n).reset_index(drop=True)

    alerts = []
    for i, row in top.iterrows():
        cost = row[cost_col] if cost_col else 1.0 + float(pd.Series([row["amount"]]).apply(
            lambda x: __import__("math").log1p(x) / 10
        ).iloc[0])
        alerts.append(
            Alert(
                alert_id=f"alert_{i}_{row['account_id']}",
                risk_score=float(row["risk_score"]),
                cost=cost,
                urgency=1.0,  # đơn giản hóa: mọi cảnh báo cùng mức khẩn cấp cơ bản
            )
        )
    return alerts


def run_qape_for_dashboard(
    df: pd.DataFrame, budget: float, top_n: int = 30
) -> dict[str, SolverResult | list[Alert] | None]:
    """Chạy QAPE thật (Greedy + ILP) trên batch cảnh báo hiện tại.

    Dashboard CHỈ dùng Greedy + ILP (đủ nhanh cho tương tác thời gian
    thực). Simulated Annealing và QAOA ĐÃ implement đầy đủ và có test
    (src/qape/solvers.py::solve_simulated_annealing, solve_qaoa), nhưng
    KHÔNG gọi ở đây — QAOA đặc biệt chậm (giây tới chục giây tùy số
    qubit, xem QAOA_MAX_QUBITS_DEFAULT) nên không phù hợp cho một
    Dashboard cần phản hồi tức thời. Dùng src/qape/benchmark.py để so
    sánh cả 4 phương pháp (Greedy/ILP/SA/QAOA) ngoài luồng, không phải
    trong Dashboard.

    ⚠ ILP có thể infeasible (tổng chi phí cảnh báo Critical > budget) —
    xem SolverResult.status trong src/qape/solvers.py. Khi đó trả về
    ilp=None và caller (app.py) PHẢI hiển thị cảnh báo, không được coi
    im lặng như "0 cảnh báo được chọn".
    """
    alerts = build_alerts_from_scored_df(df, top_n=top_n)
    if not alerts:
        return {"alerts": [], "greedy": None, "ilp": None, "ilp_infeasible": False}

    greedy_result = solve_greedy(alerts, budget)
    ilp_infeasible = False
    try:
        ilp_result = solve_ilp_exact(alerts, budget)
        if ilp_result.status == "infeasible":
            ilp_infeasible = True
            ilp_result = None
    except ImportError:
        ilp_result = None  # pulp chưa cài — dashboard vẫn chạy được với Greedy

    return {
        "alerts": alerts,
        "greedy": greedy_result,
        "ilp": ilp_result,
        "ilp_infeasible": ilp_infeasible,
    }
