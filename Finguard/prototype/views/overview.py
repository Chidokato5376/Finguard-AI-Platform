import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import core
import inference as inf
import ui

df = core.sample()
res = core.results()
th = res["thresholds"]
clusters = inf.alert_clusters()
alerts = df[df["level"] != "low"]

ui.header("Overview Dashboard", "Tổng quan hoạt động giám sát trên 3 giờ giao dịch thật của AMLWorld HI-Medium "
          "(tập validation). Mọi số liệu dưới đây được tính từ điểm của mô hình đã huấn luyện.")

c = st.columns(3) + st.columns(3)
c[0].metric("Total Transactions Analyzed", core.fmt_num(len(df)))
c[1].metric("Suspicious Transactions", core.fmt_num(len(alerts)),
            f"{core.fmt_pct(len(alerts) / len(df), 2)} tổng số", delta_color="off")
c[2].metric("High-Risk Accounts", core.fmt_num(df.loc[df["level"] == "critical", "account_id"].nunique()),
            help="Tài khoản gửi có ít nhất một giao dịch mức Chặn ngay.")
c[3].metric("Average Risk Score", core.dec(100 * alerts["score_he_thong"].mean(), 2),
            help="Điểm rủi ro trung bình của các giao dịch bị cảnh báo. Điểm = phân vị so với toàn bộ "
                 "4,6 triệu giao dịch val (99,9 = đáng ngờ hơn 99,9% giao dịch).")
c[4].metric("Active Fraud Networks", len(clusters),
            help="Cụm ≥ 3 tài khoản nối với nhau bởi các giao dịch bị cảnh báo (thành phần liên thông).")
c[5].metric("Recent Risk Alerts (🔴)", core.fmt_num((df["level"] == "critical").sum()))

left, right = st.columns([1.6, 1], gap="large")
with left:
    st.subheader("Transaction Analysis Timeline")
    t = (df.assign(t=df["timestamp"].dt.floor("10min")).groupby("t")
         .agg(total=("level", "size"), alerts=("level", lambda s: int((s != "low").sum()))).reset_index())
    fig = go.Figure()
    fig.add_trace(go.Bar(x=t["t"], y=t["total"], name="Giao dịch", marker_color="#1E3A5F"))
    fig.add_trace(go.Scatter(x=t["t"], y=t["alerts"], name="Cảnh báo", yaxis="y2", mode="lines+markers",
                             line=dict(color=ui.CYAN, width=2)))
    fig.update_layout(yaxis=dict(title="Giao dịch / 10 phút"),
                      yaxis2=dict(title="Cảnh báo", overlaying="y", side="right", showgrid=False))
    st.plotly_chart(ui.style(fig, 320), use_container_width=True)
with right:
    st.subheader("Risk Level Distribution")
    counts = df["level"].value_counts().reindex(["critical", "medium", "low"], fill_value=0)
    fig = go.Figure(go.Pie(labels=[f"{core.LEVELS[k][0]} {core.LEVELS[k][1]}" for k in counts.index],
                           values=counts.values, hole=.62, sort=False,
                           marker=dict(colors=[ui.RED, ui.AMBER, ui.GREEN]), textinfo="value"))
    fig.add_annotation(text=f"{core.fmt_pct(len(alerts) / len(df), 2)}<br>cảnh báo", showarrow=False,
                       font=dict(size=16, color="#E6F0FA"))
    st.plotly_chart(ui.style(fig, 320), use_container_width=True)

left, right = st.columns([1.2, 1], gap="large")
with left:
    st.subheader("Transaction Risk Distribution")
    p = core.probability(df["score_he_thong"].to_numpy())
    bins = [0, 0.001, 0.01, 0.05, 0.2, 0.5, 0.8, 1.0001]
    names = ["< 0,1%", "0,1–1%", "1–5%", "5–20%", "20–50%", "50–80%", "≥ 80%"]
    cut = pd.cut(p, bins, labels=names, right=False).value_counts().reindex(names)
    fig = px.bar(x=names, y=cut.values, log_y=True, color=names,
                 color_discrete_sequence=["#1E3A5F", "#22597A", "#1F7A8C", "#22D3EE", "#FBBF24", "#FB923C", "#FB7185"],
                 labels={"x": "Xác suất rửa tiền ước tính", "y": "Số giao dịch (thang log)"}, text=cut.values)
    st.plotly_chart(ui.style(fig, 320, legend=False), use_container_width=True)
    st.caption("Xác suất ước tính = điểm hệ thống hiệu chỉnh đẳng trương trên val HI-Medium.")
with right:
    st.subheader("Fraud Network Overview")
    if clusters.empty:
        st.info("Không có cụm ≥ 3 tài khoản.")
    else:
        top = clusters.head(15)
        fig = px.bar(top, x="cluster", y="accounts", color="label_share", color_continuous_scale="Teal",
                     range_color=(0, 1), labels={"accounts": "Số tài khoản", "cluster": "",
                                                 "label_share": "Tỷ lệ GD có nhãn rửa tiền"},
                     hover_data={"transactions": True, "critical": True})
        st.plotly_chart(ui.style(fig, 320, legend=False), use_container_width=True)
        share = float((clusters["label_share"] * clusters["transactions"]).sum() / clusters["transactions"].sum())
        st.caption(f"{len(clusters)} cụm, {int(clusters['accounts'].sum())} tài khoản. Kiểm chứng bằng nhãn thật: "
                   f"{core.fmt_pct(share, 0)} giao dịch bị cảnh báo trong các cụm là rửa tiền thật.")

st.subheader("Recent Risk Alerts")
recent = alerts.sort_values("timestamp", ascending=False).head(12)
st.dataframe(pd.DataFrame({
    "Alert time": recent["timestamp"].dt.strftime("%d/%m %H:%M:%S"),
    "Transaction ID": recent["tx_id"].map(lambda v: f"TX-{int(v):06d}"),
    "Account": recent["account_id"].astype(str), "Counterparty": recent["counterparty_id"].astype(str),
    "Amount": recent["amount"].round(2), "Currency": recent["currency"].astype(str),
    "Risk Level": [f"{core.LEVELS[v][0]} {core.LEVELS[v][1]}" for v in recent["level"]],
    "Risk Score": (100 * recent["score_he_thong"]).round(3),
}), hide_index=True, use_container_width=True)

st.subheader("Hiệu năng mô hình (validation HI-Medium, 4,6 triệu giao dịch)")
med = res["leaderboard_hi_medium"]["models"]
p2 = res["p2_hi_medium"]
k = st.columns(4)
k[0].metric("PR-AUC hệ thống (nửa sau val)", core.dec(p2["auprc_b"]["tron_thu_hang"], 3),
            f"+{core.dec(p2['auprc_b']['tron_thu_hang'] - p2['auprc_b']['gbdt_28'], 3)} nhờ DRRE", delta_color="normal")
k[1].metric("Precision của 500 cảnh báo đầu", core.fmt_pct(med["gbdt_28"]["caught@500"] / 500, 0))
k[2].metric("Mức Chặn ngay: Precision / Recall",
            f"{core.fmt_pct(th['critical_precision'], 0)} / {core.fmt_pct(th['critical_recall'], 0)}")
k[3].metric("Bắt 70% vụ khi chỉ xem", core.fmt_pct(th["medium_top_share"], 2) + " GD")
ui.source_note("prototype/bundle/results.json (bảng xếp hạng reports/r7/leaderboard, P2 không rò rỉ). "
               "Chi tiết ở trang Model Insights.")
