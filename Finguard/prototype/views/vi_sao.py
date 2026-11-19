import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import core

df = core.sample()
st.title("Vì sao giao dịch này đáng ngờ?")

top = df[df["level"] != "low"].sort_values("score_he_thong", ascending=False).head(300)
default = core.default_alert()
labels = {int(r.tx_id): f"#{int(r.tx_id)} · {core.LEVELS[r.level][0]} điểm {100 * r.score_he_thong:.3f} · "
                        f"{r.account_id} → {r.counterparty_id} · {r.amount:,.2f} {r.currency}"
          for r in top.itertuples()}
c1, c2 = st.columns([3, 1])
with c1:
    choice = st.selectbox("Chọn một cảnh báo (300 cảnh báo điểm cao nhất)", list(labels),
                          index=list(labels).index(default), format_func=labels.get)
with c2:
    typed = st.number_input("…hoặc nhập Mã GD bất kỳ", min_value=0, max_value=len(df) - 1, value=choice)
tx = int(typed) if typed != choice else int(choice)
row = df.iloc[tx]

emoji, name, action = core.LEVELS[row["level"]]
m = st.columns([1.4, 1, 1, 1])
m[0].metric("Mức", f"{emoji} {name}", border=True)
m[1].metric("Điểm hệ thống", f"{100 * row['score_he_thong']:.3f}", border=True)
m[2].metric("Phân vị GBDT 28", f"{100 * row['pct_gbdt_28']:.3f}", border=True)
m[3].metric("Phân vị DRRE R7", f"{100 * row['pct_R7']:.3f}", border=True)
st.markdown(f"**Đề xuất:** {action}")
if st.toggle("Hiện nhãn thật (để kiểm chứng)", value=True):
    if bool(row["label"]):
        pat = row["laundering_pattern"] or "UNSPECIFIED"
        st.error(f"Nhãn AMLWorld: **rửa tiền**, kiểu **{pat}** — {core.PATTERN_LABELS.get(pat, '')}.",
                 icon=":material/gpp_bad:")
    else:
        st.success("Nhãn AMLWorld: giao dịch hợp lệ.", icon=":material/verified:")

st.subheader("Yếu tố nào đẩy điểm lên?")
st.caption("Phép thử 'nếu bình thường': thay từng đặc trưng bằng giá trị trung vị trên val rồi chấm lại "
           "bằng GBDT. Thanh đỏ = đặc trưng làm giao dịch đáng ngờ hơn; xanh = làm bớt đáng ngờ.")
att = core.gbdt_attribution(row).head(10).iloc[::-1]
fig = go.Figure(go.Bar(
    x=att["Đóng góp (log-odds)"], y=att["Đặc trưng"], orientation="h",
    marker_color=np.where(att["Đóng góp (log-odds)"] > 0, "#d64545", "#3a8f6a"),
    customdata=np.c_[att["Phân vị trên val"], att["Giá trị"]],
    hovertemplate="%{y}<br>Đóng góp %{x:.2f}<br>Phân vị trên val: %{customdata[0]:.0f}<br>"
                  "Giá trị: %{customdata[1]:.3f}<extra></extra>"))
fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Đóng góp vào log-odds")
st.plotly_chart(fig, use_container_width=True)

pct = core.feature_percentiles(row)
mods = {
    "Chuỗi tài khoản trung gian (money mule)": ["mc_forward_ratio_24h", "mc_amt_sim_last_in",
                                                "mc_log_min_since_in", "mc_log_age_days_cap3"],
    "Chuyển liên ngân hàng / ngoại tệ": ["cb_cross_bank", "cb_log_n_banks_7d", "cb_log_amount_usd"],
    "Dòng tiền vòng tròn": ["cy_reciprocity_7d", "cy_log_n_reverse_7d", "cy_log_min_since_reverse"],
}
cols = st.columns(3, border=True)
for col, (title, feats) in zip(cols, mods.items()):
    with col:
        st.markdown(f"**{title}**")
        for f in feats:
            st.caption(f"{core.FEATURE_LABELS[f].split('] ')[-1]}: {core.dec(float(row[f]), 3)} (phân vị {pct[f]:.0f})")

st.subheader("Đường đi của tiền quanh giao dịch")
flow = core.money_flow(df, row)
g = nx.MultiDiGraph()
for r in flow.itertuples():
    g.add_edge(str(r.account_id), str(r.counterparty_id), main=bool(r.is_alert), label=bool(r.label),
               amount=r.amount, cur=r.currency, t=r.timestamp)
pos = nx.spring_layout(nx.DiGraph(g).to_undirected(), seed=7, k=1.5 / max(len(g), 1) ** 0.5, iterations=200)
fig = go.Figure()
for u, v, d in g.edges(data=True):
    x0, y0 = pos[u]
    x1, y1 = pos[v]
    main = d["main"]
    color = "#111111" if main else ("#d64545" if d["label"] else "#9aa9b3")
    fig.add_annotation(x=x1, y=y1, ax=x0, ay=y0, xref="x", yref="y", axref="x", ayref="y",
                       showarrow=True, arrowhead=3, arrowsize=1.2, arrowwidth=3.5 if main else 1.4,
                       arrowcolor=color, opacity=0.9,
                       hovertext=f"{d['amount']:,.2f} {d['cur']} lúc {pd.Timestamp(d['t']):%H:%M}")
nodes = list(g.nodes)
special = {str(row["account_id"]): "Người gửi", str(row["counterparty_id"]): "Người nhận"}
fig.add_trace(go.Scatter(
    x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes], mode="markers+text",
    text=[special.get(n, "") for n in nodes], textposition="top center",
    marker=dict(size=[18 if n in special else 10 for n in nodes],
                color=["#0e6375" if n in special else "#c9d6de" for n in nodes],
                line=dict(width=1, color="#0e6375")),
    hovertext=nodes, hoverinfo="text"))
fig.update_layout(height=460, showlegend=False, margin=dict(l=10, r=10, t=10, b=10),
                  xaxis=dict(visible=False), yaxis=dict(visible=False), plot_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig, use_container_width=True)
st.caption("Mũi tên đen đậm: giao dịch đang xét. Mũi tên đỏ: giao dịch có nhãn rửa tiền. Chỉ gồm các giao "
           f"dịch nối với người gửi/người nhận trong 2 bước, ±24 giờ quanh cảnh báo, lấy từ toàn bộ val ({len(flow)} giao dịch).")

st.dataframe(pd.DataFrame({
    "Thời điểm": flow["timestamp"].dt.strftime("%d/%m %H:%M:%S"),
    "Từ": flow["account_id"].astype(str), "Đến": flow["counterparty_id"].astype(str),
    "Số tiền": flow["amount"].round(2), "Tiền tệ": flow["currency"].astype(str),
    "Điểm": (100 * flow["score_he_thong"]).round(3),
    "Nhãn thật": np.where(flow["label"].astype(bool), "Rửa tiền", "—"),
}), hide_index=True, height=260)
