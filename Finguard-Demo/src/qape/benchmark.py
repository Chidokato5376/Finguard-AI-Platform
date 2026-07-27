"""Benchmark bốn phương pháp QAPE (Mục 6.4).

Chỉ số: approximation ratio (so với ILP), tỷ lệ vi phạm ràng buộc,
thời gian chạy, số vòng lặp hội tụ tham số (riêng cho QAOA).

⚠ Sensitivity analysis bắt buộc: chạy sweep lambda_1/lambda_2 (xem
docs/architecture.md) — không chỉ báo cáo một cấu hình penalty duy nhất.

Hai nguồn cảnh báo để benchmark:
  1. Tổng hợp (--n-alerts): dữ liệu giả lập, dùng khi CHƯA có dữ liệu
     thật/checkpoint — kiểm tra logic solver độc lập với chất lượng model.
  2. Thật (--source): đọc data/processed/{source}_{split}.parquet, chấm
     điểm bằng đúng pipeline Dashboard đang dùng (scoring_service — tự
     chọn DRRE đã train hoặc Heuristic Scorer), rồi benchmark trên chính
     các cảnh báo đó. Đây là bước "Vòng 5" trong roadmap README §7.
"""

from __future__ import annotations

import argparse
import logging
import random

from src.qape.qubo import Alert
from src.qape.solvers import (
    QAOA_MAX_QUBITS_DEFAULT,
    solve_greedy,
    solve_ilp_exact,
    solve_qaoa,
    solve_simulated_annealing,
)

logger = logging.getLogger(__name__)


def generate_synthetic_batch(n_alerts: int, seed: int = 42) -> list[Alert]:
    """Sinh một lô cảnh báo tổng hợp — dùng khi CHƯA có dữ liệu thật/
    checkpoint (kiểm tra logic solver độc lập với chất lượng model).
    Khi đã có dữ liệu thật, dùng build_alerts_from_real_data() thay thế.
    """
    rng = random.Random(seed)
    alerts = []
    for i in range(n_alerts):
        risk_score = rng.uniform(0, 100)
        alerts.append(
            Alert(
                alert_id=f"alert_{i}",
                risk_score=risk_score,
                cost=rng.uniform(1, 10),
                urgency=rng.uniform(0.5, 2.0),
            )
        )
    return alerts


def build_alerts_from_real_data(
    source: str,
    split: str = "test",
    processed_dir: str = "data/processed",
    checkpoint_path: str = "models/drre_checkpoint.pt",
    top_n: int = 30,
    max_rows: int = 500,
) -> list[Alert]:
    """Đọc dữ liệu đã tiền xử lý, chấm điểm bằng ĐÚNG pipeline Dashboard
    đang dùng, rồi chuyển thành Alert để benchmark QAPE.

    Tái dùng trực tiếp data_service/scoring_service/qape_service (không
    viết lại logic) — đảm bảo benchmark phản ánh CHÍNH XÁC những gì
    Dashboard sẽ thấy, không phải một pipeline chấm điểm riêng biệt có
    thể lệch pha với sản phẩm thật.

    Args:
        source: "paysim" | "ieee_cis" | "synthetic_vn".
        split: "train" | "val" | "test" — mặc định "test" (dữ liệu model
            chưa từng thấy, giống Dashboard mô phỏng giao dịch mới).
        checkpoint_path: nếu tồn tại, dùng DRRE đã huấn luyện; nếu không,
            tự động rơi về Heuristic Scorer (xem scoring_service.py).
        top_n: số cảnh báo risk_score cao nhất đưa vào QAPE — QAOA/ILP
            simulator chỉ khả thi tới ~20-30 cảnh báo/lô.
        max_rows: giới hạn số dòng đọc trước khi chấm điểm — tránh chấm
            điểm toàn bộ triệu dòng chỉ để lấy top_n cảnh báo.

    Raises:
        FileNotFoundError: nếu chưa chạy src/data/preprocessing.py cho
            nguồn/split này.
    """
    from src.dashboard.data_service import get_recent_transactions, load_processed_split
    from src.dashboard.qape_service import build_alerts_from_scored_df
    from src.dashboard.scoring_service import classify_tier, score_batch

    df = load_processed_split(source, split, processed_dir)
    logger.info("Đã đọc %d dòng từ %s_%s.parquet", len(df), source, split)

    # QUAN TRỌNG: chấm điểm TRƯỚC khi cắt cửa sổ hiển thị (giống bug đã
    # phát hiện và sửa trong dashboard/app.py — Heuristic Scorer cần lịch
    # sử đầy đủ theo account_id để tính z-score có ý nghĩa, cắt trước sẽ
    # làm mất lịch sử và khiến mọi điểm số về mức trung tính giả tạo).
    scored_full = score_batch(df)
    scored_full["tier"] = scored_full["risk_score"].apply(classify_tier)
    logger.info("Đã chấm điểm bằng method=%s", scored_full["method"].unique().tolist())

    time_col = "step" if "step" in scored_full.columns else "timestamp"
    windowed = get_recent_transactions(scored_full, n=max_rows, time_col=time_col)

    alerts = build_alerts_from_scored_df(windowed, top_n=top_n)
    logger.info("Đã chuyển %d giao dịch risk_score cao nhất thành Alert cho QAPE", len(alerts))
    return alerts


def run_benchmark_on_alerts(
    alerts: list[Alert], budget: float, lambda_1_values: list[float],
    run_qaoa: bool = True, cost_scale: float = 1.0,
) -> None:
    """Lõi benchmark — nhận list[Alert] có sẵn (từ dữ liệu tổng hợp HOẶC
    dữ liệu thật), không quan tâm nguồn gốc. Tách riêng khỏi việc SINH
    alerts để hai luồng (run_benchmark synthetic / run_benchmark_from_real_data)
    dùng chung một lõi, tránh lệch logic giữa hai đường.

    Args:
        run_qaoa: mặc định True, nhưng QAOA CHẬM (statevector simulator,
            xem QAOA_MAX_QUBITS_DEFAULT/solve_qaoa) — đặt False để chỉ
            benchmark Greedy/ILP/SA khi n_alerts lớn hoặc cần vòng lặp
            nhanh trong lúc phát triển.
        cost_scale: mặc định 1.0 (KHÁC với mặc định 100.0 của solve_qaoa/
            solve_simulated_annealing) — phát hiện qua đo đạc thực tế:
            cost_scale=100.0 làm số qubit slack cho ràng buộc ngân sách
            tăng vọt (19 qubit cho 8 cảnh báo, thay vì 13 với cost_scale=
            1.0), vì slack cần ~log2(budget*cost_scale) qubit. Đánh đổi:
            cost_scale=1.0 làm tròn cost về số nguyên gần nhất (mất độ
            chính xác dưới 1 đơn vị chi phí) — chấp nhận được vì cost
            trong dự án này vốn đã là ước lượng (amount/1_000_000), không
            phải số đo chính xác cần giữ nguyên phần thập phân.
    """
    if not alerts:
        logger.warning("Danh sách alerts rỗng — không có gì để benchmark.")
        return

    ilp_result = solve_ilp_exact(alerts, budget)
    greedy_result = solve_greedy(alerts, budget)

    logger.info("ILP (reference optimum): status=%s, objective=%.2f, runtime=%.4fs",
                ilp_result.status, ilp_result.objective_value, ilp_result.runtime_seconds)
    logger.info("Greedy: objective=%.2f, approx_ratio=%.3f, runtime=%.4fs",
                greedy_result.objective_value,
                greedy_result.objective_value / max(ilp_result.objective_value, 1e-9),
                greedy_result.runtime_seconds)

    if ilp_result.status != "optimal":
        logger.warning(
            "ILP infeasible với budget=%.1f — approximation ratio không tính được "
            "(chia cho objective ILP không tồn tại). Tăng budget hoặc giảm số cảnh báo.",
            budget,
        )
        return

    # Số qubit THẬT (không phải ước lượng) — đo trực tiếp từ QUBO đã dựng,
    # vì số slack qubit cho ràng buộc ngân sách phụ thuộc phi tuyến vào
    # budget*cost_scale (~log2), KHÔNG chỉ đơn giản là n_alerts+1 (công
    # thức cũ đã sai, phát hiện qua đo đạc thực tế — chênh lệch hơn 2 lần).
    from src.qape.qubo import build_quadratic_program, to_penalized_qubo

    qp_probe = build_quadratic_program(alerts, budget, cost_scale=cost_scale)
    n_qubits_actual = to_penalized_qubo(qp_probe, lambda_1=1.0).get_num_vars()

    qaoa_feasible = run_qaoa and n_qubits_actual <= QAOA_MAX_QUBITS_DEFAULT
    if run_qaoa and not qaoa_feasible:
        logger.warning(
            "Bỏ qua QAOA: %d qubit thật > QAOA_MAX_QUBITS_DEFAULT=%d. "
            "Giảm --top-n/--n-alerts, giảm --cost-scale, hoặc gọi solve_qaoa trực tiếp "
            "với max_qubits cao hơn nếu máy đủ RAM.",
            n_qubits_actual, QAOA_MAX_QUBITS_DEFAULT,
        )

    logger.info("=== Sensitivity analysis: sweep lambda_1 (SA%s) — %d qubit thật cho QAOA ===",
                " + QAOA" if qaoa_feasible else "", n_qubits_actual)
    for lam in lambda_1_values:
        sa_result = solve_simulated_annealing(
            alerts, budget, lambda_1=lam, cost_scale=cost_scale, seed=42
        )
        sa_ratio = sa_result.objective_value / max(ilp_result.objective_value, 1e-9)
        logger.info(
            "lambda_1=%.1f | SA: objective=%.2f approx_ratio=%.3f violated=%s runtime=%.4fs",
            lam, sa_result.objective_value, sa_ratio,
            sa_result.constraint_violated, sa_result.runtime_seconds,
        )

        if qaoa_feasible:
            # QUAN TRỌNG: dùng p_layers/maxiter THẤP hơn mặc định của solve_qaoa
            # (reps=2, maxiter=50) — đo thực tế cho thấy sweep 4 giá trị lambda_1
            # ở cấu hình mặc định không hoàn thành nổi trong thời gian hợp lý dù
            # đã giảm cost_scale. Cấu hình nhẹ này đánh đổi độ chính xác QAOA lấy
            # khả năng chạy sweep thực tế — dùng solve_qaoa() trực tiếp với tham
            # số mặc định nếu cần kết quả chính xác hơn cho MỘT lambda_1 cụ thể.
            qaoa_result = solve_qaoa(
                alerts, budget, lambda_1=lam, cost_scale=cost_scale, p_layers=1, maxiter=20
            )
            qaoa_ratio = qaoa_result.objective_value / max(ilp_result.objective_value, 1e-9)
            logger.info(
                "lambda_1=%.1f | QAOA: objective=%.2f approx_ratio=%.3f violated=%s runtime=%.4fs",
                lam, qaoa_result.objective_value, qaoa_ratio,
                qaoa_result.constraint_violated, qaoa_result.runtime_seconds,
            )

    logger.info(
        "KHÔNG kết luận 'quantum advantage' từ kết quả trên — ở quy mô demo "
        "này ILP exact gần như chắc chắn nhanh và tối ưu hơn QAOA (đã nêu rõ "
        "trong đặc tả, Mục 6.3). Benchmark này minh họa khả năng tích hợp, "
        "không phải bằng chứng lợi thế lượng tử."
    )


def run_benchmark(
    n_alerts: int, budget: float, lambda_1_values: list[float],
    run_qaoa: bool = True, cost_scale: float = 1.0,
) -> None:
    """Benchmark trên dữ liệu TỔNG HỢP — dùng khi chưa có dữ liệu thật/
    checkpoint. Xem run_benchmark_from_real_data() cho dữ liệu thật.
    """
    alerts = generate_synthetic_batch(n_alerts)
    run_benchmark_on_alerts(alerts, budget, lambda_1_values, run_qaoa, cost_scale)


def run_benchmark_from_real_data(
    source: str, budget: float, lambda_1_values: list[float],
    split: str = "test", processed_dir: str = "data/processed",
    checkpoint_path: str = "models/drre_checkpoint.pt",
    top_n: int = 30, max_rows: int = 500,
    run_qaoa: bool = True, cost_scale: float = 1.0,
) -> None:
    """Benchmark trên risk score THẬT — đọc dữ liệu đã tiền xử lý, chấm
    điểm bằng đúng pipeline Dashboard (scoring_service), rồi benchmark
    QAPE trên các cảnh báo đó. Đây là bước dùng khi đã có dữ liệu PaySim/
    IEEE-CIS thật (và tùy chọn checkpoint DRRE đã huấn luyện).
    """
    alerts = build_alerts_from_real_data(
        source, split=split, processed_dir=processed_dir,
        checkpoint_path=checkpoint_path, top_n=top_n, max_rows=max_rows,
    )
    run_benchmark_on_alerts(alerts, budget, lambda_1_values, run_qaoa, cost_scale)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark QAPE solvers")
    parser.add_argument(
        "--source", default=None,
        help="Nếu đặt (vd. 'paysim'), benchmark trên risk score THẬT đọc từ "
             "data/processed/{source}_{split}.parquet thay vì dữ liệu tổng hợp. "
             "Không đặt -> dùng --n-alerts (dữ liệu tổng hợp).",
    )
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--checkpoint-path", default="models/drre_checkpoint.pt",
                         help="Nếu tồn tại, benchmark dùng risk score từ DRRE đã huấn luyện; "
                              "nếu không, tự động rơi về Heuristic Scorer.")
    parser.add_argument("--top-n", type=int, default=15,
                         help="Số cảnh báo risk_score cao nhất lấy từ dữ liệu thật để đưa vào QAPE "
                              "(chỉ áp dụng khi dùng --source)")
    parser.add_argument("--max-rows", type=int, default=500,
                         help="Giới hạn số dòng đọc trước khi chấm điểm (chỉ áp dụng khi dùng --source)")
    parser.add_argument("--n-alerts", type=int, default=15,
                         help="Số cảnh báo tổng hợp (chỉ áp dụng khi KHÔNG dùng --source). "
                              "Mặc định 15 (không phải 25) để QAOA khả thi trong "
                              "QAOA_MAX_QUBITS_DEFAULT — tăng lên nếu chỉ cần Greedy/ILP/SA")
    parser.add_argument("--budget", type=float, default=50.0)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--no-qaoa", action="store_true",
                         help="Bỏ qua QAOA (chỉ Greedy/ILP/SA) — nhanh hơn nhiều")
    parser.add_argument("--cost-scale", type=float, default=1.0,
                         help="Mặc định 1.0 (không phải 100.0) — xem docstring "
                              "run_benchmark_on_alerts về lý do cost_scale ảnh hưởng mạnh "
                              "tới số qubit QAOA")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    if args.source:
        logger.info("=== Benchmark trên dữ liệu THẬT: %s (%s) ===", args.source, args.split)
        run_benchmark_from_real_data(
            args.source, args.budget, lambda_1_values=[1.0, 5.0, 10.0, 20.0],
            split=args.split, processed_dir=args.processed_dir,
            checkpoint_path=args.checkpoint_path, top_n=args.top_n, max_rows=args.max_rows,
            run_qaoa=not args.no_qaoa, cost_scale=args.cost_scale,
        )
    else:
        logger.info("=== Benchmark trên dữ liệu TỔNG HỢP (chưa có --source) ===")
        run_benchmark(
            args.n_alerts, args.budget, lambda_1_values=[1.0, 5.0, 10.0, 20.0],
            run_qaoa=not args.no_qaoa, cost_scale=args.cost_scale,
        )


if __name__ == "__main__":
    main()
