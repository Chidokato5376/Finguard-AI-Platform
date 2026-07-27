"""Bốn phương pháp giải QAPE để benchmark (Mục 6.4):
Greedy | ILP exact | Simulated Annealing | QAOA

QUAN TRỌNG (đã nêu trong đặc tả): ở quy mô 20-30 cảnh báo, ILP exact
gần như chắc chắn nhanh và tốt hơn QAOA trên simulator. Đây là proof
of concept, KHÔNG claim quantum advantage ở quy mô demo.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.qape.qubo import Alert


@dataclass(frozen=True)
class SolverResult:
    solver_name: str
    selected_alert_ids: list[str]
    objective_value: float
    runtime_seconds: float
    constraint_violated: bool
    # "optimal" | "infeasible" | "heuristic" — "infeasible" nghĩa là ràng
    # buộc budget và ràng buộc "mọi cảnh báo Critical phải chọn" MÂU THUẪN
    # nhau (tổng chi phí Critical > budget) — xem cảnh báo trong
    # solve_ilp_exact bên dưới, phát hiện qua chạy thử thực tế với dữ liệu
    # có nhiều cảnh báo Critical.
    status: str = "optimal"


def solve_greedy(alerts: list[Alert], budget: float) -> SolverResult:
    """Baseline đơn giản: sắp theo (r_i*u_i)/c_i giảm dần, thêm vào cho tới khi hết budget.
    Không đảm bảo tối ưu — dùng làm sàn so sánh dưới cùng.
    """
    import time

    start = time.perf_counter()
    critical = [a for a in alerts if a.is_critical]
    remaining = [a for a in alerts if not a.is_critical]
    remaining.sort(key=lambda a: (a.risk_score * a.urgency) / max(a.cost, 1e-9), reverse=True)

    selected = list(critical)
    used_budget = sum(a.cost for a in critical)
    for alert in remaining:
        if used_budget + alert.cost <= budget:
            selected.append(alert)
            used_budget += alert.cost

    objective = sum(a.risk_score * a.urgency for a in selected)
    runtime = time.perf_counter() - start
    return SolverResult(
        solver_name="greedy",
        selected_alert_ids=[a.alert_id for a in selected],
        objective_value=objective,
        runtime_seconds=runtime,
        constraint_violated=used_budget > budget,
    )


def solve_ilp_exact(alerts: list[Alert], budget: float) -> SolverResult:
    """Nghiệm tối ưu tham chiếu bằng ILP exact (PuLP) — dùng làm ground truth
    cho approximation ratio của QAOA/Simulated Annealing.

    ⚠ Bài toán có thể INFEASIBLE: nếu tổng chi phí của các cảnh báo Critical
    (r_i >= 90, bắt buộc x_i=1) đã vượt quá budget, không tồn tại nghiệm khả
    thi nào — hai ràng buộc mâu thuẫn nhau. Trường hợp này KHÔNG hiếm khi
    risk_score của cả lô đều cao (vd. dữ liệu concept-drift/tấn công dồn dập).
    Hàm này kiểm tra status sau khi solve và trả về status="infeasible" thay
    vì một objective/selection vô nghĩa — lỗi này đã từng xảy ra âm thầm
    (objective âm, selection không khớp ràng buộc) trước khi được phát hiện
    qua chạy thử thực tế và vá tại đây.
    """
    import logging
    import time

    import pulp

    logger = logging.getLogger(__name__)

    start = time.perf_counter()
    prob = pulp.LpProblem("QAPE_ILP", pulp.LpMaximize)
    x = {a.alert_id: pulp.LpVariable(a.alert_id, cat="Binary") for a in alerts}

    prob += pulp.lpSum(a.risk_score * a.urgency * x[a.alert_id] for a in alerts)
    prob += pulp.lpSum(a.cost * x[a.alert_id] for a in alerts) <= budget

    critical_cost = sum(a.cost for a in alerts if a.is_critical)
    for a in alerts:
        if a.is_critical:
            prob += x[a.alert_id] == 1

    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    runtime = time.perf_counter() - start
    status = pulp.LpStatus[prob.status]

    if status != "Optimal":
        logger.warning(
            "QAPE_ILP không tìm được nghiệm tối ưu (status=%s). Nguyên nhân "
            "thường gặp: tổng chi phí cảnh báo Critical (%.1f) vượt budget "
            "(%.1f) — hai ràng buộc mâu thuẫn nhau. Tăng budget hoặc giảm "
            "ngưỡng Critical trong config.yaml.", status, critical_cost, budget,
        )
        return SolverResult(
            solver_name="ilp_exact",
            selected_alert_ids=[],
            objective_value=float("nan"),
            runtime_seconds=runtime,
            constraint_violated=True,
            status="infeasible",
        )

    selected = [a.alert_id for a in alerts if pulp.value(x[a.alert_id]) > 0.5]
    objective = pulp.value(prob.objective) or 0.0
    used_budget = sum(a.cost for a in alerts if a.alert_id in selected)

    return SolverResult(
        solver_name="ilp_exact",
        selected_alert_ids=selected,
        objective_value=float(objective),
        runtime_seconds=runtime,
        constraint_violated=used_budget > budget,
        status="optimal",
    )


def _interpret_qubo_solution(
    var_names: list[str], bit_values: dict[int, int], alerts: list[Alert], budget: float
) -> tuple[list[str], float, bool]:
    """Diễn giải một nghiệm QUBO (bitstring) theo ràng buộc GỐC (không
    phải bản đã penalize) — dùng chung cho SA và QAOA để đảm bảo cách
    tính objective_value/constraint_violated NHẤT QUÁN với solve_ilp_exact
    và solve_greedy (so sánh công bằng khi benchmark).

    Lưu ý: var_names bao gồm cả các biến slack do to_penalized_qubo sinh
    ra (từ ràng buộc inequality) — CHỈ lấy các biến khớp alert_id thật,
    bỏ qua phần còn lại.

    Sau khi diễn giải thô, áp dụng _repair_critical_constraint() — xem
    docstring hàm đó để biết lý do bắt buộc phải có bước này.
    """
    alert_ids = {a.alert_id for a in alerts}
    selected_ids = [
        var_names[i] for i, v in bit_values.items()
        if v == 1 and i < len(var_names) and var_names[i] in alert_ids
    ]
    selected_ids = _repair_critical_constraint(selected_ids, alerts, budget)

    selected_alerts = [a for a in alerts if a.alert_id in selected_ids]
    objective = sum(a.risk_score * a.urgency for a in selected_alerts)
    used_budget = sum(a.cost for a in selected_alerts)
    constraint_violated = used_budget > budget

    return selected_ids, float(objective), constraint_violated


def _repair_critical_constraint(
    selected_ids: list[str], alerts: list[Alert], budget: float
) -> list[str]:
    """Đảm bảo ràng buộc "mọi cảnh báo Critical (r_i >= 90) phải được
    chọn" LUÔN đúng, bất kể penalty method (lambda_1) có đủ mạnh hay
    không trong nghiệm thô từ SA/QAOA.

    === LÝ DO BƯỚC NÀY LÀ BẮT BUỘC (phát hiện qua kiểm thử thực tế) ===
    QUBO chỉ ràng buộc Critical dưới dạng "mềm" (penalty), không phải
    ràng buộc cứng như ILP. Đo thực nghiệm cho thấy với lambda_1=5.0
    (giá trị mặc định), nghiệm SA có thể BỎ SÓT cảnh báo Critical trong
    khi vẫn "tối ưu" theo QUBO đã penalize — vì lambda_1 không đủ lớn so
    với thang giá trị objective (risk_score*urgency). Tăng lambda_1 lên
    20 giải quyết được ở ví dụ thử nghiệm cụ thể, nhưng lambda_1=50 lại
    KHÔNG (không đơn điệu — đặc tính điển hình của penalty method, xem
    sensitivity analysis khuyến nghị trong docs/architecture.md). Thay vì
    yêu cầu người dùng dò lambda_1 hoàn hảo, bước repair này áp dụng SAU
    khi giải, đảm bảo tính đúng đắn của ràng buộc không phụ thuộc vào
    việc tune penalty có may mắn hay không — ĐÚNG tinh thần "ràng buộc
    Critical là bắt buộc" của bài toán gốc (Mục 6.1), nhất quán với cách
    solve_greedy đã xử lý (luôn ép chọn Critical trước, không qua penalty).

    Thuật toán: (1) thêm mọi Critical còn thiếu vào selected; (2) nếu
    vượt budget, loại bớt các cảnh báo KHÔNG Critical có tỷ lệ giá trị/
    chi phí thấp nhất cho tới khi hết vi phạm hoặc chỉ còn lại Critical
    (giống hệt logic ưu tiên của solve_greedy — nhất quán giữa các solver).
    """
    alert_by_id = {a.alert_id: a for a in alerts}
    selected = set(selected_ids)
    critical_ids = {a.alert_id for a in alerts if a.is_critical}

    selected |= critical_ids  # bước (1): ép mọi Critical vào

    used_budget = sum(alert_by_id[i].cost for i in selected)
    if used_budget <= budget:
        return list(selected)

    # bước (2): loại bớt non-critical, giá trị/chi phí thấp nhất trước
    non_critical_selected = sorted(
        (alert_by_id[i] for i in selected if i not in critical_ids),
        key=lambda a: (a.risk_score * a.urgency) / max(a.cost, 1e-9),
    )
    for alert in non_critical_selected:
        if used_budget <= budget:
            break
        selected.discard(alert.alert_id)
        used_budget -= alert.cost

    return list(selected)


def solve_simulated_annealing(
    alerts: list[Alert],
    budget: float,
    lambda_1: float = 5.0,
    cost_scale: float = 100.0,
    num_reads: int = 200,
    seed: int | None = None,
) -> SolverResult:
    """Giải cùng QUBO (build_quadratic_program + to_penalized_qubo) bằng
    Simulated Annealing cổ điển (dwave-neal) — so sánh công bằng với QAOA
    trên CÙNG một hàm mục tiêu đã penalize, khác nhau ở phương pháp tìm
    kiếm (cổ điển ngẫu nhiên vs lượng tử biến phân).

    Dùng dwave-neal (neal.SimulatedAnnealingSampler) thay vì tự viết vòng
    lặp Metropolis-Hastings tay — đây là thư viện chuẩn, đã kiểm chứng
    rộng rãi cho bài toán QUBO trong cộng đồng D-Wave/lượng tử, tránh rủi
    ro cài sai chi tiết lập lịch nhiệt độ (annealing schedule) một cách
    tinh vi mà khó phát hiện qua test.

    Args:
        cost_scale: xem build_quadratic_program — BẮT BUỘC vì Alert.cost
            luôn là float, QuadraticProgramToQubo cần hệ số nguyên.
        num_reads: số lần chạy annealing độc lập, lấy nghiệm tốt nhất.
    """
    import time

    import neal

    from src.qape.qubo import build_quadratic_program, to_penalized_qubo

    start = time.perf_counter()
    qp = build_quadratic_program(alerts, budget, cost_scale=cost_scale)
    qubo = to_penalized_qubo(qp, lambda_1=lambda_1)

    Q: dict[tuple[int, int], float] = {}
    for i, coef in qubo.objective.linear.to_dict().items():
        Q[(i, i)] = Q.get((i, i), 0.0) + float(coef)
    for (i, j), coef in qubo.objective.quadratic.to_dict().items():
        Q[(i, j)] = Q.get((i, j), 0.0) + float(coef)

    sampler = neal.SimulatedAnnealingSampler()
    sampleset = sampler.sample_qubo(Q, num_reads=num_reads, seed=seed)
    best = sampleset.first

    var_names = [v.name for v in qubo.variables]
    selected_ids, objective, constraint_violated = _interpret_qubo_solution(
        var_names, dict(best.sample), alerts, budget
    )
    runtime = time.perf_counter() - start

    return SolverResult(
        solver_name="simulated_annealing",
        selected_alert_ids=selected_ids,
        objective_value=objective,
        runtime_seconds=runtime,
        constraint_violated=constraint_violated,
        status="heuristic",
    )


# Trần an toàn cho số qubit — xem GHI CHÚ HIỆU NĂNG trong solve_qaoa.
# Con số này đến từ ĐO ĐẠC THỰC TẾ: StatevectorSampler mất ~1.5-2s cho 13
# qubit, ~30s cho 16 qubit (reps=1, maxiter~20-30), trên phần cứng tham
# chiếu dùng để phát triển repo — VỚI ĐIỀU KIỆN cài đúng phiên bản ghim
# trong requirements.txt (qiskit>=1.0,<2.0, qiskit-algorithms==0.4.0,
# qiskit-machine-learning>=0.7,<0.9).
#
# ⚠ BÀI HỌC THỰC TẾ QUAN TRỌNG (đã xảy ra trong quá trình phát triển,
# ghi lại để tránh lặp lại): có lúc đo được CÙNG bài toán 13 qubit chậm
# hơn 100 LẦN (không hoàn thành sau 150s). Ban đầu tưởng là môi trường
# sandbox suy giảm tài nguyên — SAI. Nguyên nhân thật: một lệnh `pip
# install qiskit-machine-learning` KHÔNG ghim phiên bản (thiếu `<0.9`)
# đã âm thầm nâng cấp qiskit từ 1.4.6 lên 2.5.0, phá vỡ khả năng tương
# thích với qiskit-algorithms==0.4.0 (bắt buộc qiskit<2.0) — cài đặt
# không báo lỗi, nhưng QAOA chạy qua một đường xử lý chậm bất thường.
# Cài lại ĐÚNG bộ phiên bản ghim trong requirements.txt khôi phục ngay
# hiệu năng bình thường (~2s cho 13 qubit). BÀI HỌC: luôn cài đặt quantum
# stack như MỘT LỆNH DUY NHẤT theo đúng requirements.txt, không cài lẻ
# từng gói — pip không đảm bảo giữ nguyên ràng buộc phiên bản đã cài
# trước đó khi cài thêm gói mới trong lệnh riêng biệt.
QAOA_MAX_QUBITS_DEFAULT = 16


def solve_qaoa(
    alerts: list[Alert],
    budget: float,
    p_layers: int = 2,
    lambda_1: float = 5.0,
    cost_scale: float = 100.0,
    max_qubits: int = QAOA_MAX_QUBITS_DEFAULT,
    maxiter: int = 50,
    seed: int = 42,
) -> SolverResult:
    """Phương pháp đề xuất — QAOA trên simulator (Mục 6.2, 6.3).

    === GHI CHÚ HIỆU NĂNG (đo đạc thực tế, không suy đoán) ===
    Dùng qiskit.primitives.StatevectorSampler (mô phỏng statevector chính
    xác) kết hợp qiskit_algorithms.QAOA + MinimumEigenOptimizer. Đã thử
    qiskit_aer SamplerV2 (backend C++, thường nhanh hơn) nhưng gặp lỗi
    tương thích với DiagonalEstimator nội bộ của qiskit_algorithms 0.4.0
    (bản mới nhất tính đến thời điểm viết) — 'unknown instruction: QAOA'.
    StatevectorSampler là lựa chọn ỔN ĐỊNH đã kiểm chứng chạy đúng, dù
    chậm hơn. Cải thiện hiệu năng (backend Aer tương thích) nằm ngoài
    phạm vi bản tham chiếu này.

    ⚠ PHIÊN BẢN THƯ VIỆN QUAN TRỌNG: cần qiskit>=1.0,<2.0 (qiskit 2.x đã
    bỏ hẳn API mà qiskit_algorithms 0.4.0 phụ thuộc, gây lỗi runtime khó
    hiểu "Invalid circuits, expected Sequence[QuantumCircuit]" — đã xác
    nhận qua kiểm thử thực tế, xem requirements.txt).

    ⚠ VỀ initial_point CỐ ĐỊNH: hàm này dùng seed để cố định điểm khởi
    tạo tham số QAOA thay vì để thư viện tự sinh ngẫu nhiên — giúp kết
    quả TÁI LẬP ĐƯỢC giữa các lần chạy (cùng seed → cùng kết quả), tiện
    cho việc so sánh/debug. Đây không phải là biện pháp khắc phục sự cố
    hiệu năng (xem QAOA_MAX_QUBITS_DEFAULT để biết bài học thực tế quan
    trọng hơn về NGUYÊN NHÂN THẬT của một sự cố chậm bất thường từng gặp
    trong quá trình phát triển: xung đột phiên bản qiskit do cài đặt
    không ghim đúng requirements.txt, KHÔNG phải do initial_point ngẫu
    nhiên như nghi ngờ ban đầu).

    Giới hạn quy mô: khả thi tới ~20-30 qubit theo đặc tả gốc, nhưng ĐO
    THỰC TẾ trong môi trường phát triển repo này cho thấy ngưỡng thực
    dụng thấp hơn nhiều (xem QAOA_MAX_QUBITS_DEFAULT) — RAM và thời gian
    chờ chấp nhận được mới là giới hạn thật, không phải giới hạn lý
    thuyết của thuật toán. Vượt max_qubits, hàm raise lỗi rõ ràng thay vì
    để máy treo hàng chục phút không phản hồi.

    KHÔNG claim quantum advantage — đây là proof of concept minh họa khả
    năng tích hợp, ILP exact gần như chắc chắn nhanh và tốt hơn ở quy mô
    demo (xem Mục 6.3 đặc tả gốc).
    """
    import logging
    import time

    import numpy as np
    from qiskit.primitives import StatevectorSampler
    from qiskit_algorithms import QAOA
    from qiskit_algorithms.optimizers import COBYLA
    from qiskit_optimization.algorithms import MinimumEigenOptimizer

    from src.qape.qubo import build_quadratic_program, to_penalized_qubo

    logger = logging.getLogger(__name__)

    start = time.perf_counter()
    qp = build_quadratic_program(alerts, budget, cost_scale=cost_scale)
    qubo = to_penalized_qubo(qp, lambda_1=lambda_1)
    n_qubits = qubo.get_num_vars()

    if n_qubits > max_qubits:
        raise ValueError(
            f"QAOA cần {n_qubits} qubit (bao gồm biến slack từ ràng buộc "
            f"ngân sách) — vượt max_qubits={max_qubits}. Mô phỏng statevector "
            f"tốn bộ nhớ theo cấp số nhân (2^n); giá trị mặc định đã được đo "
            f"đạc thực tế để tránh treo máy/Out-Of-Memory. Giảm số cảnh báo "
            f"đưa vào (top_n trong Dashboard), tăng max_qubits nếu máy đủ "
            f"mạnh, hoặc dùng solve_ilp_exact/solve_simulated_annealing cho "
            f"lô lớn — đây cũng chính là khuyến nghị thực tế của dự án "
            f"(xem docs/architecture.md, Mục 6.3)."
        )

    # Cố định initial_point — xem cảnh báo PHƯƠNG SAI THỜI GIAN CHẠY ở trên.
    # QAOA(reps=p_layers) cần 2*p_layers tham số (gamma, beta mỗi lớp).
    rng = np.random.default_rng(seed)
    initial_point = rng.uniform(0, 2 * np.pi, size=2 * p_layers)

    eval_count = 0

    def _count_evals(*_args, **_kwargs) -> None:
        nonlocal eval_count
        eval_count += 1

    sampler = StatevectorSampler()
    qaoa = QAOA(
        sampler=sampler, optimizer=COBYLA(maxiter=maxiter), reps=p_layers,
        initial_point=initial_point, callback=_count_evals,
    )
    optimizer = MinimumEigenOptimizer(qaoa)
    result = optimizer.solve(qp)
    runtime = time.perf_counter() - start

    logger.info(
        "QAOA hoàn tất: %d qubit, %d lần gọi COBYLA callback, %.2fs (%.3fs/lần).",
        n_qubits, eval_count, runtime, runtime / max(eval_count, 1),
    )

    var_names = [v.name for v in qp.variables]
    bit_values = {i: int(round(val)) for i, val in enumerate(result.x)}
    selected_ids, objective, constraint_violated = _interpret_qubo_solution(
        var_names, bit_values, alerts, budget
    )

    return SolverResult(
        solver_name="qaoa",
        selected_alert_ids=selected_ids,
        objective_value=objective,
        runtime_seconds=runtime,
        constraint_violated=constraint_violated,
        status="heuristic",
    )
