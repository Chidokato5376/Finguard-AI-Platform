import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

import core

df = core.sample()
th = core.results()["thresholds"]

st.title("Hàng đợi cảnh báo")
st.markdown(
    "Mỗi giao dịch nhận một **điểm rủi ro 0–100**: điểm 99,9 nghĩa là đáng ngờ hơn 99,9% giao dịch "
    "trong tập val. Ngưỡng ba mức được chọn trên val HI-Medium:")
cols = st.columns(3, border=True)
desc = {
    "critical": f"Từ {core.dec(core.score_100(th['critical']), 3)} điểm (top {core.fmt_pct(th['critical_top_share'], 3)}). "
                f"Trên val: chính xác {core.fmt_pct(th['critical_precision'], 0)}, bắt {core.fmt_pct(th['critical_recall'], 0)} vụ.",
    "medium": f"Từ {core.dec(core.score_100(th['medium']), 3)} điểm (top {core.fmt_pct(th['medium_top_share'], 2)}). "
              f"Cộng dồn với mức trên: bắt {core.fmt_pct(th['medium_recall'], 0)} vụ, chính xác {core.fmt_pct(th['medium_precision'], 0)}.",
    "low": "Còn lại.",
}
for col, lv in zip(cols, ("critical", "medium", "low")):
    emoji, name, action = core.LEVELS[lv]
    with col:
        st.markdown(f"**{emoji} {name}**")
        st.caption(desc[lv])
        st.caption(action)

hours = sorted(df["timestamp"].dt.floor("h").unique())
pick = st.select_slider("Khung giờ đang xem", options=hours,
                        value=(hours[0], hours[-1]), format_func=lambda t: pd.Timestamp(t).strftime("%H:%M"))
view = df[(df["timestamp"] >= pick[0]) & (df["timestamp"] < pd.Timestamp(pick[1]) + pd.Timedelta(hours=1))]

counts = view["level"].value_counts()
alerts = view[view["level"] != "low"]
m = st.columns(5)
m[0].metric("Giao dịch trong khung", core.fmt_num(len(view)), border=True)
m[1].metric("🔴 Chặn ngay", int(counts.get("critical", 0)), border=True)
m[2].metric("🟡 Cần xác minh", int(counts.get("medium", 0)), border=True)
m[3].metric("Tỷ lệ phải xem", core.fmt_pct(len(alerts) / max(len(view), 1), 2), border=True)
m[4].metric("Vụ rửa tiền thật bị bỏ sót", int(view.loc[view["level"] == "low", "label"].sum()),
            help="Nhãn thật của AMLWorld, chỉ dùng để kiểm chứng — hệ thống không thấy nhãn khi chấm.",
            border=True)

show_truth = st.toggle("Hiện nhãn thật (để kiểm chứng)", value=True)
filt = st.segmented_control("Lọc", ["Mọi cảnh báo", "Chỉ 'Chặn ngay'", "Tất cả giao dịch"],
                            default="Mọi cảnh báo")
table = view if filt == "Tất cả giao dịch" else (
    view[view["level"] == "critical"] if filt == "Chỉ 'Chặn ngay'" else alerts)
table = table.sort_values("score_he_thong", ascending=False).head(500)

out = pd.DataFrame({
    "Mã GD": table["tx_id"],
    "Mức": [f"{core.LEVELS[lv][0]} {core.LEVELS[lv][1]}" for lv in table["level"]],
    "Điểm rủi ro": (100 * table["score_he_thong"]).round(3),
    "Thời điểm": table["timestamp"].dt.strftime("%H:%M %d/%m"),
    "Người gửi": table["account_id"].astype(str),
    "Người nhận": table["counterparty_id"].astype(str),
    "Số tiền": table["amount"].round(2),
    "Tiền tệ": table["currency"].astype(str),
    "Kênh": table["channel"].astype(str),
    "Phân vị GBDT": (100 * table["pct_gbdt_28"]).round(2),
    "Phân vị DRRE": (100 * table["pct_R7"]).round(2),
})
if show_truth:
    out["Nhãn thật"] = np.where(table["label"].astype(bool),
                                "Rửa tiền · " + table["laundering_pattern"].replace("", "?"), "—")
st.dataframe(out, hide_index=True, height=420,
             column_config={"Điểm rủi ro": st.column_config.ProgressColumn(
                 "Điểm rủi ro", min_value=99.0, max_value=100.0, format="%.3f")})
st.caption("Bảng hiện tối đa 500 dòng có điểm cao nhất. Chọn *Mã GD* để xem giải thích ở trang "
           "*Vì sao đáng ngờ*.")

st.subheader("Số cảnh báo theo giờ")
hourly = (view.assign(h=view["timestamp"].dt.floor("15min"))
          .groupby(["h", "level"]).size().reset_index(name="n"))
hourly = hourly[hourly["level"] != "low"]
fig = px.bar(hourly, x="h", y="n", color="level",
             color_discrete_map={"critical": "#d64545", "medium": "#e8a33d"},
             labels={"h": "Thời điểm (khung 15 phút)", "n": "Số cảnh báo", "level": "Mức"})
fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig, use_container_width=True)
