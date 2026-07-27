"""QAPE — phát biểu bài toán và chuyển sang QUBO (Mục 6.1, 6.2).

max  sum_i (r_i * u_i) * x_i
s.t. sum_i c_i * x_i <= C
     x_i = 1  forall i: r_i >= 90

H_C = -sum_i(r_i*u_i)*x_i + lambda_1*(sum_i c_i*x_i - C)^2
      + lambda_2*sum_{i: r_i>=90}(1-x_i)

Stack: qiskit-optimization dựng QuadraticProgram trực tiếp từ ràng buộc
(không cần tự viết QUBO tay).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from qiskit_optimization import QuadraticProgram
except ImportError:  # pragma: no cover
    QuadraticProgram = None


@dataclass(frozen=True)
class Alert:
    """Một cảnh báo trong lô xử lý (cửa sổ 15 phút, Mục 6.1)."""

    alert_id: str
    risk_score: float      # r_i, 0-100 (từ Module 5)
    cost: float             # c_i — chi phí xử lý ước tính
    urgency: float           # u_i — tuổi cảnh báo / mức khẩn cấp

    @property
    def is_critical(self) -> bool:
        return self.risk_score >= 90


def build_quadratic_program(
    alerts: list[Alert],
    budget: float,
    lambda_1: float = 5.0,
    lambda_2: float = 10.0,
    cost_scale: float | None = None,
) -> "QuadraticProgram":
    """Dựng QuadraticProgram trực tiếp từ ràng buộc — cách khuyến nghị
    thay vì tự viết QUBO tay (tránh sai sót dấu/hệ số).

    Ràng buộc x_i = 1 cho các cảnh báo Critical được đưa vào như
    linear constraint cứng (equality), KHÔNG đưa vào objective — chỉ
    ràng buộc ngân sách mới cần penalty method (lambda_1) vì QAOA
    không xử lý trực tiếp inequality constraint tổng quát.

    Args:
        cost_scale: nếu khác None, NHÂN cost/budget với hệ số này rồi LÀM
            TRÒN về số nguyên trước khi đưa vào constraint. BẮT BUỘC khi
            kết quả sẽ đưa qua to_penalized_qubo() — bộ chuyển đổi
            InequalityToEquality của qiskit-optimization đòi hỏi hệ số
            NGUYÊN để tạo biến slack hợp lệ, và raise QiskitOptimizationError
            với thông báo khó hiểu ("Incompatible problem: ... float
            coefficients ...") nếu cost là số thực — Alert.cost trong dự
            án này LUÔN là float (amount/1_000_000), nên lỗi này chắc chắn
            xảy ra nếu bỏ qua tham số này (đã xác nhận qua kiểm thử thực
            tế). Để None khi chỉ dùng QuadraticProgram cho mục đích khác
            (không chuyển QUBO), giữ hệ số gốc cho dễ đọc.
    """
    if QuadraticProgram is None:
        raise ImportError("qiskit-optimization chưa được cài. Xem requirements.txt.")

    def _cost(a: Alert) -> float:
        return round(a.cost * cost_scale) if cost_scale is not None else a.cost

    scaled_budget = round(budget * cost_scale) if cost_scale is not None else budget

    qp = QuadraticProgram(name="QAPE")
    for alert in alerts:
        qp.binary_var(name=alert.alert_id)

    # Objective: max sum_i (r_i * u_i) * x_i  ->  qiskit_optimization tối thiểu hóa
    # mặc định, nên đảo dấu khi minimize, hoặc dùng qp.maximize().
    linear_obj = {a.alert_id: a.risk_score * a.urgency for a in alerts}
    qp.maximize(linear=linear_obj)

    # Ràng buộc ngân sách (soft, xử lý bằng lambda_1 khi chuyển QUBO — xem
    # to_penalized_qubo bên dưới; ở đây khai báo trước dưới dạng constraint
    # tường minh để có thể benchmark với ILP exact).
    qp.linear_constraint(
        linear={a.alert_id: _cost(a) for a in alerts},
        sense="<=",
        rhs=scaled_budget,
        name="budget",
    )

    # Ràng buộc bắt buộc: mọi cảnh báo Critical phải được chọn.
    for alert in alerts:
        if alert.is_critical:
            qp.linear_constraint(
                linear={alert.alert_id: 1}, sense="==", rhs=1,
                name=f"force_critical_{alert.alert_id}",
            )

    return qp


def to_penalized_qubo(qp: "QuadraticProgram", lambda_1: float = 5.0):
    """Chuyển QuadraticProgram có ràng buộc sang QUBO không ràng buộc
    bằng penalty method (dùng converter chuẩn của qiskit-optimization
    thay vì viết H_C thủ công, giảm rủi ro sai công thức).
    """
    from qiskit_optimization.converters import QuadraticProgramToQubo

    converter = QuadraticProgramToQubo(penalty=lambda_1)
    return converter.convert(qp)
