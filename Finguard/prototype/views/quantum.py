import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import core
import ui
from src.qape.qubo import Alert
from src.qape.solvers import solve_greedy, solve_ilp_exact

df = core.sample()

ui.header("Quantum Optimization", "QAPE — chọn cảnh báo cho ca trực dưới ràng buộc thời gian. Đây là bài toán tối ưu "
          "phân bổ nguồn lực, KHÔNG phải mô hình dự đoán. Bộ giải chạy thật: src/qape/solvers.py.")

st.subheader("Optimization Pipeline")
p = st.columns(5)
for col, (t, d) in zip(p, [
        ("1 · Candidate Alerts", "Cảnh báo 🔴/🟡 của giờ đã chọn, theo thứ tự phát sinh, kèm xác suất ước tính."),
        ("2 · Chi phí & ngân sách", "Thời gian xử lý mỗi cảnh báo (giả định minh họa) và quỹ giờ của ca."),
        ("3 · QUBO Formulation", "Knapsack + ràng buộc bắt buộc → QuadraticProgram → QUBO phạt (qiskit-optimization)."),
        ("4 · Solvers", "Greedy, ILP chính xác (PuLP/CBC), ủ mô phỏng (dwave-neal), QAOA (Qiskit Aer)."),
        ("5 · So sánh", "Rủi ro kỳ vọng, số vụ thật bắt được, thời gian giải — so với đến-trước-làm-trước.")]):
    with col:
        ui.card(t, d)

with st.expander("QUBO Formulation (đúng như src/qape/qubo.py)", expanded=False):
    st.latex(r"\max_{x\in\{0,1\}^n}\ \sum_i r_i\,u_i\,x_i \quad \text{s.t.}\quad \sum_i c_i x_i \le B,"
             r"\qquad x_i = 1\ \ orall i:\ r_i \ge 90")
    st.latex(r"\min_{x,s}\ -\sum_i r_i u_i x_i + \lambda_1\Big(\sum_i c_i x_i + \sum_k 2^k s_k - B\Big)^2"
             r" + \lambda_2 \sum_{i\,\in\,\text{critical}} (1-x_i)")
    st.caption("r_i = 100 × xác suất rửa tiền ước tính; c_i = phút xử lý; B = quỹ phút của ca; s_k = biến bù (slack) "
               "nhị phân của ràng buộc ngân sách. Chuyển QUBO bằng QuadraticProgramToQubo (penalty λ₁ = 5).")

with st.container(border=True):
    c1, c2, c3 = st.columns(3)
    hour = c1.selectbox("Cảnh báo phát sinh trong giờ", sorted(df["timestamp"].dt.floor("h").unique()),
                        format_func=lambda t: pd.Timestamp(t).strftime("%H:00–%H:59"))
    n_alerts = c2.slider("Số cảnh báo đầu tiên trong hàng đợi", 4, 40, 20)
    hours = c3.slider("Quỹ thời gian của ca (giờ)", 0.5, 6.0, 2.5, step=0.5)

# Hàng đợi thực tế: các cảnh báo (🔴 + 🟡) của giờ đã chọn, theo thứ tự phát sinh.
pool = df[(df["timestamp"].dt.floor("h") == hour) & (df["level"] != "low")]
pool = pool.sort_values("timestamp", kind="stable").head(n_alerts).copy()
if len(pool) < 3:
    st.info("Giờ này có quá ít cảnh báo. Chọn giờ khác.")
    st.stop()

# Giả định chi phí xử lý (phút): 8 phút cơ bản, +5 nếu chuyển khác ngân hàng, +5 nếu không phải USD,
# +2 phút cho mỗi bậc độ lớn số tiền trên 1.000. Đây là GIẢ ĐỊNH để minh họa, không phải số đo thực tế.
pool["p"] = core.probability(pool["score_he_thong"].to_numpy())
pool["minutes"] = (8 + 5 * (pool["cb_cross_bank"] > 0.5) + 5 * (pool["currency"].astype(str) != "US Dollar")
                   + 2 * np.maximum(np.log10(pool["amount"].clip(lower=1)) - 3, 0)).round(0)
alerts = [Alert(alert_id=str(int(r.tx_id)), risk_score=float(100 * r.p), cost=float(r.minutes) / 10,
                urgency=1.0) for r in pool.itertuples()]
budget = hours * 60 / 10

results = {}
order_fifo = pool.sort_values("timestamp")
used, chosen = 0.0, []
for r in order_fifo.itertuples():
    if used + r.minutes <= hours * 60:
        chosen.append(str(int(r.tx_id)))
        used += r.minutes
results["Đến trước làm trước"] = (chosen, 0.0)
t = time.perf_counter()
g = solve_greedy(alerts, budget)
results["Tham lam (Greedy)"] = (g.selected_alert_ids, time.perf_counter() - t)
try:
    t = time.perf_counter()
    ilp = solve_ilp_exact(alerts, budget)
    if ilp.status != "infeasible":
        results["Tối ưu chính xác (ILP)"] = (ilp.selected_alert_ids, time.perf_counter() - t)
    else:
        st.warning("Riêng các cảnh báo bắt buộc (xác suất ước tính ≥ 90%) đã cần nhiều thời gian hơn quỹ "
                   "của ca — không có lời giải khả thi. Tăng quỹ thời gian hoặc bổ sung người.",
                   icon=":material/warning:")
except ImportError:
    st.caption("Chưa cài pulp — bỏ qua ILP.")

quantum = st.toggle(
    "Chạy thêm QUBO: ủ mô phỏng, và QAOA khi có ≤ 6 cảnh báo (mô phỏng lượng tử, ~30 giây)", value=False)
if quantum:
    try:
        from src.qape.solvers import solve_simulated_annealing

        t = time.perf_counter()
        sa = solve_simulated_annealing(alerts, budget, seed=42, cost_scale=10)
        results["QUBO — ủ mô phỏng"] = (sa.selected_alert_ids, time.perf_counter() - t)
    except ImportError:
        st.caption("Chưa cài dwave-neal — bỏ qua ủ mô phỏng.")
    if len(alerts) <= 6:
        try:
            from src.qape.solvers import solve_qaoa

            with st.spinner("QAOA đang chạy trên bộ mô phỏng lượng tử..."):
                t = time.perf_counter()
                qa = solve_qaoa(alerts, budget, seed=42, cost_scale=10)
            results["QUBO — QAOA"] = (qa.selected_alert_ids, time.perf_counter() - t)
        except ImportError:
            st.caption("Chưa cài qiskit — bỏ qua QAOA (chạy được trên máy cài đủ requirements).")
        except Exception as e:  # noqa: BLE001
            st.caption(f"QAOA lỗi: {e}")
    else:
        st.caption("QAOA chỉ chạy khi có ≤ 6 cảnh báo: mỗi cảnh báo là một qubit, cộng ~8 qubit phụ cho ràng buộc ngân sách; "
                   "mô phỏng trên máy cổ điển tốn bộ nhớ 2^n. Kéo thanh số cảnh báo xuống 6 để thử.")

rows = []
for name, (ids, secs) in results.items():
    sel = pool[pool["tx_id"].astype(str).isin(ids)]
    rows.append({"Cách chọn": name, "Số cảnh báo xử lý": len(sel),
                 "Thời gian dùng (phút)": int(sel["minutes"].sum()),
                 "Rủi ro kỳ vọng xử lý được (Σ xác suất)": round(float(sel["p"].sum()), 2),
                 "Vụ rửa tiền thật bắt được": int(sel["label"].sum()),
                 "Tiền rửa thật được chặn": round(float(sel.loc[sel["label"].astype(bool), "amount"].sum()), 0),
                 "Thời gian giải (ms)": round(1000 * secs, 1)})
board = pd.DataFrame(rows)
st.subheader(f"So sánh trên {len(pool)} cảnh báo, quỹ {hours:g} giờ".replace(".", ","))
st.markdown("**Optimization Results · Classical vs Quantum Comparison**")
st.dataframe(board, hide_index=True, use_container_width=True)

fig = go.Figure(go.Bar(x=board["Cách chọn"], y=board["Rủi ro kỳ vọng xử lý được (Σ xác suất)"],
                       marker_color=["#2B4763"] + [ui.CYAN] * (len(board) - 1),
                       text=board["Vụ rửa tiền thật bắt được"].map(lambda v: f"{v} vụ thật"),
                       textposition="outside"))
fig.update_layout(yaxis_title="Rủi ro kỳ vọng xử lý được")
st.plotly_chart(ui.style(fig, 320, legend=False), use_container_width=True)

best = results.get("Tối ưu chính xác (ILP)", results["Tham lam (Greedy)"])[0]
show = pool.assign(chon=pool["tx_id"].astype(str).isin(best)).sort_values(["chon", "p"], ascending=False)
st.markdown("**Candidate Alerts · danh sách đề xuất (lời giải tối ưu):**")
st.dataframe(pd.DataFrame({
    "Xử lý trong ca": np.where(show["chon"], "✅", "⏭ ca sau"),
    "Mã GD": show["tx_id"], "Thời điểm": show["timestamp"].dt.strftime("%H:%M"),
    "Xác suất rửa tiền ước tính": show["p"].map(lambda v: core.fmt_pct(v, 0)),
    "Thời gian xử lý (phút)": show["minutes"].astype(int),
    "Số tiền": show["amount"].round(2), "Tiền tệ": show["currency"].astype(str),
    "Nhãn thật": np.where(show["label"].astype(bool), "Rửa tiền", "—"),
}), hide_index=True, use_container_width=True)
st.caption("Cảnh báo có xác suất ước tính ≥ 90% luôn bắt buộc xử lý. Xác suất ước tính lấy từ điểm hệ thống qua hồi quy đẳng trương trên val HI-Medium. "
           "Thời gian xử lý từng cảnh báo là giả định minh họa (xem chú thích trong mã nguồn). "
           f"Mỗi đơn vị chi phí QUBO = 10 phút; quỹ = {budget:g} đơn vị.")
