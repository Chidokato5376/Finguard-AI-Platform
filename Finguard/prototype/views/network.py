import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import core
import inference as inf
import ui

ui.header("Fraud Network Intelligence", "Mạng lưới tài khoản – giao dịch dựng từ dữ liệu thật (khung 3 giờ + ±24 giờ "
          "quanh các cảnh báo cao nhất). Cụm = thành phần liên thông của các giao dịch bị cảnh báo.")

edges = inf.edges()
clusters = inf.alert_clusters()
acc = inf.account_risk()

m = st.columns(5)
m[0].metric("Account nodes", core.fmt_num(len(acc)))
m[1].metric("Transaction edges", core.fmt_num(len(edges)))
m[2].metric("Suspicious accounts", core.fmt_num((acc["alerts"] > 0).sum()))
m[3].metric("Fraud clusters (≥ 3 TK)", len(clusters))
share = float((clusters["label_share"] * clusters["transactions"]).sum() / max(clusters["transactions"].sum(), 1))
m[4].metric("GD trong cụm là rửa tiền thật", core.fmt_pct(share, 0), help="Chỉ để kiểm chứng bằng nhãn AMLWorld.")

st.subheader("Fraud clusters")
if not clusters.empty:
    show = clusters.drop(columns="members").copy()
    show.columns = ["Cluster", "Tài khoản", "Giao dịch cảnh báo", "Tổng tiền", "Điểm cao nhất", "Số 🔴",
                    "Tỷ lệ nhãn rửa tiền (kiểm chứng)"]
    st.dataframe(show, hide_index=True, use_container_width=True, height=240, column_config={
        "Tỷ lệ nhãn rửa tiền (kiểm chứng)": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f"),
        "Tổng tiền": st.column_config.NumberColumn(format="%.0f")})

st.subheader("Khám phá mạng lưới")
c1, c2, c3 = st.columns([1.2, 1.6, 1])
mode = c1.radio("Bắt đầu từ", ["Một cụm", "Một tài khoản"], horizontal=True)
if mode == "Một cụm" and not clusters.empty:
    cid = c2.selectbox("Cluster", clusters["cluster"], format_func=lambda c: (
        f"{c} · {int(clusters.set_index('cluster').loc[c, 'accounts'])} tài khoản"))
    seeds = set(clusters.set_index("cluster").loc[cid, "members"])
    focus = None
else:
    suspicious = acc[acc["alerts"] > 0].index.tolist()
    focus = c2.selectbox("Tài khoản (gõ để tìm)", suspicious + [a for a in acc.index[:2000] if a not in suspicious])
    seeds = {focus}
hops = c3.slider("Số bước mở rộng", 1, 3, 1)

nodes = set(seeds)
for _ in range(hops):
    touch = edges[edges["account_id"].isin(nodes) | edges["counterparty_id"].isin(nodes)]
    nodes = set(touch["account_id"]) | set(touch["counterparty_id"])
    if len(touch) > 400:
        break
sub = edges[edges["account_id"].isin(nodes) & edges["counterparty_id"].isin(nodes)].copy()
if mode == "Một cụm":
    sub = sub[(sub["level"] != "low") | sub["account_id"].isin(seeds) | sub["counterparty_id"].isin(seeds)]
sub = sub.sort_values("score_he_thong", ascending=False).head(300)

g = nx.DiGraph()
for r in sub.itertuples():
    g.add_edge(r.account_id, r.counterparty_id)
if g.number_of_nodes() == 0:
    st.info("Không có giao dịch nào quanh lựa chọn này trong dữ liệu.")
    st.stop()
pos = nx.spring_layout(g.to_undirected(), seed=3, k=1.8 / max(g.number_of_nodes(), 1) ** 0.5, iterations=150)
truth = st.toggle("Tô cạnh có nhãn rửa tiền thật (kiểm chứng)", value=False)
fig = go.Figure()
for lvl, color, width in (("low", "#2B4763", 1), ("medium", ui.AMBER, 1.8), ("critical", ui.RED, 2.4)):
    part = sub[sub["level"] == lvl]
    xs, ys = [], []
    for r in part.itertuples():
        x0, y0 = pos[r.account_id]
        x1, y1 = pos[r.counterparty_id]
        xs += [x0, x1, None]
        ys += [y0, y1, None]
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=color, width=width), hoverinfo="skip",
                             name={"low": "GD bình thường", "medium": "GD cần xác minh", "critical": "GD chặn ngay"}[lvl]))
if truth:
    part = sub[sub["label"].astype(bool)]
    xs, ys = [], []
    for r in part.itertuples():
        x0, y0 = pos[r.account_id]
        x1, y1 = pos[r.counterparty_id]
        xs += [x0, x1, None]
        ys += [y0, y1, None]
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="#F472B6", width=1, dash="dot"),
                             name="Nhãn rửa tiền thật", hoverinfo="skip"))
risk = acc["max_score"].reindex(list(g.nodes)).fillna(0)
alerts_n = acc["alerts"].reindex(list(g.nodes)).fillna(0)
fig.add_trace(go.Scatter(
    x=[pos[n][0] for n in g.nodes], y=[pos[n][1] for n in g.nodes], mode="markers", name="Tài khoản",
    marker=dict(size=[18 if n in seeds else 9 + 2 * min(a, 6) for n, a in zip(g.nodes, alerts_n)],
                color=np.where(alerts_n.values > 0, ui.RED, ui.CYAN),
                line=dict(width=[3 if n in seeds else 0.5 for n in g.nodes], color="#F4FAFF")),
    text=[f"{n}<br>Điểm cao nhất {s:.3f}<br>Cảnh báo {int(a)}" for n, s, a in zip(g.nodes, risk, alerts_n)],
    hoverinfo="text"))
fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False))
st.plotly_chart(ui.style(fig, 560), use_container_width=True)
st.caption(f"{g.number_of_nodes()} tài khoản, {len(sub)} giao dịch. Nút đỏ: tài khoản có giao dịch bị cảnh báo; "
           "viền trắng: điểm xuất phát. Di chuột lên nút để xem điểm.")

n1, n2, n3 = st.columns(3)
n1.metric("Network risk: GD bị cảnh báo", core.fmt_pct(float((sub["level"] != "low").mean()), 0))
n2.metric("Tổng tiền đi qua", core.fmt_num(sub["amount"].sum()))
n3.metric("Số tiền tệ khác nhau", sub["currency"].nunique())

st.subheader("Transaction paths & lịch sử giao dịch")
target = focus or st.selectbox("Xem lịch sử tài khoản", sorted(seeds))
hist = edges[(edges["account_id"] == target) | (edges["counterparty_id"] == target)].sort_values("timestamp")
st.dataframe(pd.DataFrame({
    "Thời điểm": hist["timestamp"].dt.strftime("%d/%m %H:%M:%S"),
    "Chiều": np.where(hist["account_id"] == target, "→ chuyển đi", "← nhận vào"),
    "Đối tác": np.where(hist["account_id"] == target, hist["counterparty_id"], hist["account_id"]),
    "Số tiền": hist["amount"].round(2), "Tiền tệ": hist["currency"].astype(str),
    "Mức": [f"{core.LEVELS[v][0]} {core.LEVELS[v][1]}" for v in hist["level"]],
    "Điểm": (100 * hist["score_he_thong"]).round(3),
    "Nhãn thật": np.where(hist["label"].astype(bool), "Rửa tiền", "—"),
}), hide_index=True, use_container_width=True, height=300)
