from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.dashboard.story import LEVELS, fmt_money, method_label, get_view

source, scored, view = get_view()

st.title("Overview Dashboard")
st.markdown("### Intelligent Transaction Risk Management")
st.caption("Tổng quan từ tập dữ liệu đang chọn. Chỉ số của demo không đồng nghĩa hiệu năng production.")
st.info(f"**Chế độ demo · {method_label(scored)}**  \n{st.session_state.get('source', source)} — số liệu bên dưới được tính từ dữ liệu đã nạp; không phải kết quả phân tích file người dùng.", icon="ℹ️")

count = len(scored)
suspicious = int((scored["level"] != "low").sum())
accounts = int(scored.loc[scored["level"] == "critical", "account_id"].nunique())
avg_score = float(scored["risk_score"].mean()) if count else 0.0
alerts = scored[scored["level"] != "low"].sort_values("risk_score", ascending=False)
cols = st.columns(4)
cols[0].metric("Total Transactions Analyzed", f"{count:,}")
cols[1].metric("Suspicious Transactions", f"{suspicious:,}")
cols[2].metric("High-Risk Accounts", f"{accounts:,}")
cols[3].metric("Average Risk Score", f"{avg_score:.1f} / 100")
cols = st.columns(2)
cols[0].metric("Active Fraud Networks", "Chưa đo", help="Không có chỉ số fraud clusters đã xác thực trong nguồn dashboard hiện tại.")
cols[1].metric("Recent Risk Alerts", f"{len(alerts):,}", help="Giao dịch được xếp mức medium/critical theo scoring đang hiển thị.")

left, right = st.columns([1.25, 1], gap="large")
with left:
    st.subheader("Transaction Analysis Timeline")
    time_col = "timestamp" if "timestamp" in scored else "step"
    timeline = scored.copy()
    timeline["_time"] = pd.to_datetime(timeline[time_col], errors="coerce")
    timeline = timeline.dropna(subset=["_time"])
    if not timeline.empty:
        daily = timeline.set_index("_time").resample("D").agg(
            transactions=("risk_score", "size"), suspicious=("level", lambda x: int((x != "low").sum()))
        ).reset_index()
        fig = px.area(daily, x="_time", y=["transactions", "suspicious"], template="plotly_dark", color_discrete_sequence=["#22D3EE", "#FB7185"])
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", legend_title_text="", margin=dict(l=0,r=0,t=15,b=0), xaxis_title="", yaxis_title="Giao dịch")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Không có trường thời gian hợp lệ để dựng timeline.")
with right:
    st.subheader("Risk Level Distribution")
    dist = scored["level"].value_counts().reindex(["low", "medium", "critical"], fill_value=0).rename_axis("level").reset_index(name="transactions")
    dist["Risk level"] = dist["level"].map(lambda x: LEVELS[x]["name"])
    fig = px.bar(dist, x="Risk level", y="transactions", color="level", color_discrete_map={"low":"#34D399", "medium":"#FBBF24", "critical":"#FB7185"}, template="plotly_dark", text="transactions")
    fig.update_layout(showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0,r=0,t=15,b=0), xaxis_title="", yaxis_title="Số giao dịch")
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Transaction Risk Distribution")
fig = px.histogram(scored, x="risk_score", nbins=20, template="plotly_dark", color_discrete_sequence=["#22D3EE"])
fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0,r=0,t=10,b=0), xaxis_title="Risk score (0–100)", yaxis_title="Số giao dịch")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Recent Risk Alerts")
if alerts.empty:
    st.success("Không có giao dịch medium/critical trong tập hiện tại.")
else:
    display = alerts.head(8).copy()
    show_cols = [c for c in ["timestamp", "account_id", "counterparty_id", "amount", "risk_score", "level", "method"] if c in display]
    st.dataframe(display[show_cols], hide_index=True, use_container_width=True)
st.caption("Fraud Network Overview: trang Fraud Network hiện dùng đồ thị dựng từ tập dữ liệu đang xem; cụm mạng lưới chưa được xác nhận là fraud cluster.")
