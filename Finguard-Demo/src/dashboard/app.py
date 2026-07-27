"""Dashboard giám sát rủi ro giao dịch — FinGuard AI (Mục 7.3).

Kiến trúc: app.py CHỈ lo hiển thị (UI). Toàn bộ logic nằm trong:
  - data_service.py     : đọc dữ liệu đã tiền xử lý, dựng đồ thị tài khoản
  - scoring_service.py  : tính risk score (heuristic mặc định, DRRE khi có checkpoint)
  - qape_service.py     : phân bổ cảnh báo (Greedy/ILP)
  - src/explainability/ : diễn giải kết quả

MẶC ĐỊNH DÙNG HEURISTIC SCORER (không cần huấn luyện, con số có ý nghĩa
thống kê thật ngay) thay vì DRRE chưa train — xem cảnh báo trung thực
trong src/dashboard/scoring_service.py. Khi có checkpoint đã huấn luyện
tại models/drre_checkpoint.pt, dashboard TỰ ĐỘNG chuyển sang dùng DRRE
thật, cột `method` luôn hiển thị rõ đang dùng backend nào.

Chạy: streamlit run src/dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Cho phép chạy `streamlit run src/dashboard/app.py` từ thư mục gốc repo
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

st.set_page_config(page_title="FinGuard AI — Risk Monitoring", layout="wide")


# --------------------------------------------------------------------------
# Data & scoring (cached)
# --------------------------------------------------------------------------

@st.cache_data
def _list_sources() -> list[str]:
    from src.dashboard.data_service import list_available_sources

    return list_available_sources()


@st.cache_data
def _load_split(source: str, split: str) -> pd.DataFrame:
    from src.dashboard.data_service import load_processed_split

    return load_processed_split(source, split)


@st.cache_data(show_spinner="Đang tính Risk Score...")
def _score(source: str, split: str, max_rows: int) -> pd.DataFrame:
    from src.dashboard.data_service import get_recent_transactions
    from src.dashboard.scoring_service import classify_tier, score_batch

    df = _load_split(source, split)
    # QUAN TRỌNG: tính điểm trên TOÀN BỘ lịch sử trước, rồi mới cắt lấy N
    # giao dịch gần nhất để hiển thị — không được làm ngược lại. Heuristic
    # Scorer cần lịch sử tích lũy theo account_id (rolling z-score); cắt
    # trước khi tính sẽ khiến hầu hết tài khoản "mất" lịch sử của chính
    # mình trong cửa sổ hiển thị, mọi z_score về 0 (trung tính) một cách
    # giả tạo. Lỗi này đã được phát hiện và sửa qua kiểm thử thực tế.
    scored_full = score_batch(df)
    scored_full["tier"] = scored_full["risk_score"].apply(classify_tier)

    time_col = "step" if "step" in scored_full.columns else "timestamp"
    return get_recent_transactions(scored_full, n=max_rows, time_col=time_col)


# --------------------------------------------------------------------------
# Panel 1 — Hàng đợi cảnh báo ưu tiên (Risk Scoring + QAPE)
# --------------------------------------------------------------------------

def render_alert_queue(scored: pd.DataFrame) -> pd.DataFrame:
    from src.dashboard.qape_service import run_qape_for_dashboard

    st.subheader("1. Hàng đợi cảnh báo ưu tiên")

    top_n = st.slider("Số cảnh báo xét trong lô hiện tại", 5, min(100, len(scored)), min(25, len(scored)))
    budget = st.number_input("Ngân sách xử lý (đơn vị chi phí)", min_value=1.0, value=50.0, step=5.0)

    qape_result = run_qape_for_dashboard(scored, budget=budget, top_n=top_n)
    candidates = scored.sort_values("risk_score", ascending=False).head(top_n).reset_index(drop=True)

    if qape_result["ilp_infeasible"]:
        st.warning(
            "⚠ Không tìm được phân bổ khả thi bằng ILP: tổng chi phí các "
            "cảnh báo bắt buộc xử lý (risk_score ≥ 90) đã vượt "
            "ngân sách hiện tại. Đang hiển thị kết quả từ Greedy (không tối "
            "ưu). Tăng ngân sách ở trên để có phân bổ khả thi bằng ILP."
        )

    result = qape_result["ilp"] or qape_result["greedy"]
    reference_label = "ILP (tối ưu)" if qape_result["ilp"] else "Greedy (fallback)"

    selected_ids = {a.alert_id for a in qape_result["alerts"]} & set(result.selected_alert_ids) \
        if result else set()
    id_to_row = {f"alert_{i}_{row['account_id']}": i for i, row in candidates.iterrows()}
    selected_rows = {id_to_row[aid] for aid in selected_ids if aid in id_to_row}
    candidates["qape_selected"] = candidates.index.isin(selected_rows)

    col1, col2, col3 = st.columns(3)
    col1.metric("Tổng cảnh báo trong lô", len(candidates))
    col2.metric("Được chọn xử lý", int(candidates["qape_selected"].sum()))
    col3.metric("Phương pháp", reference_label)

    display_cols = ["account_id", "counterparty_id", "risk_score", "tier", "method", "qape_selected"]
    display_cols = [c for c in display_cols if c in candidates.columns]
    st.dataframe(
        candidates[display_cols].style.background_gradient(subset=["risk_score"], cmap="Reds"),
        use_container_width=True,
    )
    return candidates


# --------------------------------------------------------------------------
# Panel 2 — Giải thích quyết định (Module 6)
# --------------------------------------------------------------------------

def render_explanation_panel(candidates: pd.DataFrame) -> str | None:
    st.subheader("2. Giải thích quyết định")

    if candidates.empty:
        st.info("Không có cảnh báo nào để giải thích.")
        return None

    options = candidates.index.tolist()
    labels = {
        i: f"#{i} — {candidates.loc[i, 'account_id']} (score={candidates.loc[i, 'risk_score']:.1f})"
        for i in options
    }
    selected_idx = st.selectbox("Chọn giao dịch để xem giải thích", options, format_func=lambda i: labels[i])
    row = candidates.loc[selected_idx]
    method = row.get("method", "heuristic_fallback")

    if method == "drre_trained":
        st.success("✅ Giải thích từ DRRE đã huấn luyện (attention/gate weights)")
        beta_weights = {
            "beta_transaction": row["beta_transaction"],
            "beta_behavior": row["beta_behavior"],
            "beta_graph": row["beta_graph"],
        }
        dominant = max(beta_weights, key=beta_weights.get)
        labels_vn = {
            "beta_transaction": "đặc trưng giao dịch thô",
            "beta_behavior": "lịch sử hành vi cá nhân",
            "beta_graph": "quan hệ mạng lưới tài khoản",
        }
        st.markdown(f"**Nguồn tín hiệu chủ đạo:** {labels_vn[dominant]} ({dominant} = {beta_weights[dominant]:.2f})")
        st.markdown(f"**Độ lệch ngữ cảnh (δ_t):** {row['delta_t']:.2f}")
        st.bar_chart(pd.Series(beta_weights, name="Trọng số gate (β)"))
    else:
        st.info("ℹ️ Giải thích từ Heuristic Scorer (chưa có checkpoint DRRE đã huấn luyện)")
        st.markdown(f"**Z-score biên độ giao dịch:** {row['z_score']:.2f} "
                    f"({'bất thường rõ rệt' if abs(row['z_score']) > 2 else 'trong khoảng bình thường'})")
        st.markdown(f"**Người nhận lần đầu xuất hiện:** {'Có' if row['is_new_counterparty'] else 'Không'}")
        st.markdown(f"**Số giao dịch lịch sử của tài khoản:** {int(row['account_history_count'])}")
        st.caption(
            "Công thức: risk_score = 100·σ(z_score + 0.8·[người nhận mới VÀ "
            "đã có ≥3 giao dịch lịch sử] − 1.2). Xem chi tiết trong "
            "src/dashboard/scoring_service.py::compute_heuristic_scores."
        )

    return row["account_id"]


# --------------------------------------------------------------------------
# Panel 3 — Sơ đồ mạng lưới tài khoản liên quan (Module 4.3)
# --------------------------------------------------------------------------

def render_network_graph(scored: pd.DataFrame, focus_account: str | None) -> None:
    from src.dashboard.data_service import build_account_graph, get_account_neighborhood

    st.subheader("3. Sơ đồ mạng lưới tài khoản liên quan")

    if focus_account is None:
        st.info("Chọn một giao dịch ở Khối 2 để xem mạng lưới liên quan.")
        return

    graph = build_account_graph(scored)
    if focus_account not in graph:
        st.info(f"Tài khoản {focus_account} không có trong đồ thị của lô hiện tại.")
        return

    ego = get_account_neighborhood(graph, focus_account, hops=1)
    if len(ego) == 0:
        st.info("Không có tài khoản liên quan nào trong lô hiện tại.")
        return

    import networkx as nx
    import plotly.graph_objects as go

    ego_undirected = ego.to_undirected()
    pos = nx.spring_layout(ego_undirected, seed=42)

    edge_x, edge_y = [], []
    for u, v in ego_undirected.edges():
        edge_x += [pos[u][0], pos[v][0], None]
        edge_y += [pos[u][1], pos[v][1], None]

    node_x = [pos[n][0] for n in ego_undirected.nodes()]
    node_y = [pos[n][1] for n in ego_undirected.nodes()]
    node_color = ["crimson" if n == focus_account else "steelblue" for n in ego_undirected.nodes()]
    node_text = list(ego_undirected.nodes())

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines",
                              line=dict(width=1, color="lightgray"), hoverinfo="none"))
    fig.add_trace(go.Scatter(x=node_x, y=node_y, mode="markers+text",
                              marker=dict(size=16, color=node_color),
                              text=node_text, textposition="top center"))
    fig.update_layout(showlegend=False, height=400, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Tài khoản trung tâm (đỏ): {focus_account}. "
               "Đồ thị chỉ dựng từ lô giao dịch hiện tại đang hiển thị, không phải toàn bộ lịch sử.")


# --------------------------------------------------------------------------
# Panel 4 — Phân bổ ngân sách xử lý (QAPE, Module 8)
# --------------------------------------------------------------------------

def render_qape_shift_summary(candidates: pd.DataFrame, budget: float = 50.0) -> None:
    st.subheader("4. Phân bổ ca trực (QAPE)")

    if "qape_selected" not in candidates.columns:
        st.info("Chưa có kết quả QAPE — xem Khối 1.")
        return

    selected = candidates[candidates["qape_selected"]]
    not_selected = candidates[~candidates["qape_selected"]]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Được chọn xử lý ngay: {len(selected)}**")
        st.dataframe(selected[["account_id", "risk_score", "tier"]], use_container_width=True, height=200)
    with col2:
        st.markdown(f"**Chờ ca sau: {len(not_selected)}**")
        st.dataframe(not_selected[["account_id", "risk_score", "tier"]], use_container_width=True, height=200)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    st.title("FinGuard AI — Bảng điều khiển Quản trị Rủi ro Giao dịch")

    sources = _list_sources()
    if not sources:
        st.warning(
            "Không tìm thấy dữ liệu đã tiền xử lý trong `data/processed/`. "
            "Chạy trước:\n\n```bash\npython -m src.data.preprocessing --source paysim "
            "--raw-path data/raw/paysim.csv --output data/processed\n```"
        )
        st.stop()

    from src.dashboard.scoring_service import DEFAULT_CHECKPOINT_PATH

    checkpoint_exists = DEFAULT_CHECKPOINT_PATH.exists()

    with st.sidebar:
        st.header("Nguồn dữ liệu")
        source_name = st.selectbox("Chọn nguồn", sources)
        split = st.selectbox("Tập dữ liệu", ["test", "val", "train"], index=0)
        max_rows = st.slider("Kích thước lô (số giao dịch)", 20, 500, 100)

        if checkpoint_exists:
            st.success("✅ Đang dùng checkpoint DRRE đã huấn luyện")
        else:
            st.info(
                "ℹ️ **Đang dùng Heuristic Scorer** (z-score biên độ giao dịch). "
                "Không cần huấn luyện, con số có ý nghĩa thống kê thật. "
                "Khi có `models/drre_checkpoint.pt`, dashboard tự chuyển sang DRRE thật."
            )

    scored = _score(source_name, split, max_rows)

    col_left, col_right = st.columns(2)
    with col_left:
        candidates = render_alert_queue(scored)
    with col_right:
        # Lấy giao dịch được chọn TRƯỚC khi vẽ sơ đồ mạng lưới, trong CÙNG
        # một lần chạy — tránh lỗi trễ 1 nhịp (trước đây đọc focus_account từ
        # session_state được ghi ở lần chạy TRƯỚC, nên sơ đồ luôn hiển thị
        # giao dịch chọn ở lần trước, không khớp lựa chọn hiện tại).
        focus_account = render_explanation_panel(candidates)
        render_qape_shift_summary(candidates)
    with col_left:
        render_network_graph(scored, focus_account)


if __name__ == "__main__":
    main()
