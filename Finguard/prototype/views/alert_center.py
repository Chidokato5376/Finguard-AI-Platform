import numpy as np
import pandas as pd
import streamlit as st

import core
import inference as inf
import ui

ui.header("Alert Center", "Mọi giao dịch được hệ thống xếp mức 🔴 Chặn ngay hoặc 🟡 Cần xác minh. Trạng thái điều tra "
          "được lưu trong phiên làm việc hiện tại.")

STATUSES = ["Mới", "Đang điều tra", "Đã xác nhận nghi vấn", "Đã báo cáo STR", "Báo động giả", "Đã đóng"]
df = core.sample()
alerts = df[df["level"] != "low"].sort_values("score_he_thong", ascending=False).copy()
alerts["alert_id"] = [f"ALR-{i + 1:05d}" for i in range(len(alerts))]
if "alert_status" not in st.session_state:
    st.session_state["alert_status"] = {}
status = st.session_state["alert_status"]
alerts["status"] = alerts["alert_id"].map(lambda a: status.get(a, "Mới"))

m = st.columns(5)
m[0].metric("Tổng cảnh báo", core.fmt_num(len(alerts)))
m[1].metric("🔴 Chặn ngay", int((alerts["level"] == "critical").sum()))
m[2].metric("🟡 Cần xác minh", int((alerts["level"] == "medium").sum()))
m[3].metric("Chưa xử lý", int((alerts["status"] == "Mới").sum()))
m[4].metric("Tài khoản liên quan", core.fmt_num(alerts["account_id"].nunique()))

with st.container(border=True):
    f1, f2, f3, f4 = st.columns([1.2, 1.4, 1.6, 1.6])
    lv = f1.multiselect("Risk level", ["critical", "medium"], default=["critical", "medium"],
                        format_func=lambda v: f"{core.LEVELS[v][0]} {core.LEVELS[v][1]}")
    stt = f2.multiselect("Investigation status", STATUSES, default=STATUSES)
    hours = sorted(df["timestamp"].dt.floor("h").unique())
    rng = f3.select_slider("Alert time", options=hours, value=(hours[0], hours[-1]),
                           format_func=lambda t: pd.Timestamp(t).strftime("%H:00"))
    q = f4.text_input("Tìm Alert ID / Transaction ID / Account", placeholder="vd. ALR-00012, TX-133397, B0129...")

view = alerts[alerts["level"].isin(lv) & alerts["status"].isin(stt)
              & (alerts["timestamp"] >= rng[0]) & (alerts["timestamp"] < pd.Timestamp(rng[1]) + pd.Timedelta(hours=1))]
if q.strip():
    s = q.strip().upper()
    view = view[view["alert_id"].str.contains(s, regex=False)
                | view["tx_id"].map(lambda v: f"TX-{int(v):06d}").str.contains(s, regex=False)
                | view["account_id"].astype(str).str.upper().str.contains(s, regex=False)
                | view["counterparty_id"].astype(str).str.upper().str.contains(s, regex=False)]

table = pd.DataFrame({
    "Alert ID": view["alert_id"], "Transaction ID": view["tx_id"].map(lambda v: f"TX-{int(v):06d}"),
    "Account ID": view["account_id"].astype(str), "Counterparty": view["counterparty_id"].astype(str),
    "Risk Level": [f"{core.LEVELS[v][0]} {core.LEVELS[v][1]}" for v in view["level"]],
    "Risk Score": (100 * view["score_he_thong"]).round(3),
    "Alert Time": view["timestamp"].dt.strftime("%d/%m/%Y %H:%M:%S"),
    "Amount": view["amount"].round(2), "Currency": view["currency"].astype(str),
    "Investigation Status": view["status"],
})
st.caption(f"{core.fmt_num(len(table))} cảnh báo khớp bộ lọc. Bấm vào một dòng để xem chi tiết.")
event = st.dataframe(table, hide_index=True, use_container_width=True, height=360, on_select="rerun",
                     selection_mode="single-row", column_config={"Risk Score": st.column_config.ProgressColumn(
                         min_value=99.0, max_value=100.0, format="%.3f")})
rows = event.selection.rows if event and hasattr(event, "selection") else []
if not len(view):
    st.stop()
sel = view.iloc[rows[0]] if rows else view.iloc[0]

st.subheader(f"Chi tiết {sel['alert_id']} · TX-{int(sel['tx_id']):06d}")
a, b = st.columns([1.1, 1])
with a:
    st.markdown(f"**{core.LEVELS[sel['level']][0]} {core.LEVELS[sel['level']][1]}** · điểm "
                f"{core.dec(100 * sel['score_he_thong'], 3)} · xác suất ước tính "
                f"{core.fmt_pct(float(core.probability(np.array([sel['score_he_thong']]))[0]), 1)}")
    st.markdown(f"{sel['account_id']} → {sel['counterparty_id']} · **{sel['amount']:,.2f} {sel['currency']}** · "
                f"{pd.Timestamp(sel['timestamp']):%H:%M:%S %d/%m/%Y}")
    st.caption(f"Đề xuất: {core.LEVELS[sel['level']][2]}")
    if inf.shap_available():
        _, sv = inf.shap_values(df.iloc[[int(sel["tx_id"])]])
        st.markdown("**Yếu tố chính (SHAP):** " + inf.key_factors(sv, df.iloc[[int(sel["tx_id"])]], k=4)[0])
    new = st.selectbox("Cập nhật trạng thái điều tra", STATUSES, index=STATUSES.index(sel["status"]),
                       key=f"st_{sel['alert_id']}")
    if new != sel["status"]:
        status[sel["alert_id"]] = new
        st.rerun()
    if st.toggle("Hiện nhãn thật (kiểm chứng)", value=False):
        st.markdown("Nhãn AMLWorld: " + (f"**rửa tiền** · {sel['laundering_pattern'] or 'UNSPECIFIED'}"
                                         if bool(sel["label"]) else "hợp lệ"))
with b:
    flow = core.money_flow(df, sel)
    st.markdown("**Giao dịch liên quan (±24 giờ, 2 bước)**")
    st.dataframe(pd.DataFrame({
        "Thời điểm": pd.to_datetime(flow["timestamp"]).dt.strftime("%d/%m %H:%M"),
        "Từ": flow["account_id"].astype(str), "Đến": flow["counterparty_id"].astype(str),
        "Số tiền": flow["amount"].round(2),
        "Mức": [core.LEVELS[v][0] for v in core.level_of(flow["score_he_thong"].to_numpy())],
    }), hide_index=True, use_container_width=True, height=280)
