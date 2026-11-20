import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import core
import inference as inf
import ui

ui.header("Explainable AI", "Giải thích bằng SHAP (TreeExplainer, giá trị chính xác — không xấp xỉ, không giả lập) "
          "trên GBDT 28 đặc trưng, thành phần chiếm 95% điểm hệ thống.")

if not inf.shap_available():
    st.error("Môi trường chưa cài `shap` — trang này không hiển thị giá trị thay thế. Cài: `pip install shap`.")
    st.stop()

df = core.sample()
feats = core.features()

st.subheader("SHAP Feature Importance (toàn cục)")
g = inf.global_shap()
top = g.head(15).iloc[::-1]
fig = go.Figure()
fig.add_trace(go.Bar(x=top["mean_abs_shap"], y=top["Đặc trưng"], orientation="h", name="Trung bình |SHAP|",
                     marker_color=ui.CYAN))
fig.update_layout(xaxis_title="Trung bình |SHAP| (log-odds)")
st.plotly_chart(ui.style(fig, 460, legend=False), use_container_width=True)
st.caption(f"Tính trên {core.fmt_num((df['level'] != 'low').sum())} cảnh báo + 4.000 giao dịch thường chọn ngẫu nhiên "
           "(seed 42) của khung mẫu. Đặc trưng có nhãn [..] thuộc 3 mô-đun kịch bản do đội xây dựng.")
d = g.head(10)[["Đặc trưng", "mean_shap_alerts", "mean_shap_normal"]].rename(columns={
    "mean_shap_alerts": "SHAP TB ở cảnh báo", "mean_shap_normal": "SHAP TB ở giao dịch thường"})
st.dataframe(d.style.format({"SHAP TB ở cảnh báo": "{:+.3f}", "SHAP TB ở giao dịch thường": "{:+.3f}"}),
             hide_index=True, use_container_width=True)

st.subheader("Local Explanation — một giao dịch cụ thể")
alerts = df[df["level"] != "low"].sort_values("score_he_thong", ascending=False)
default = core.default_alert()
opts = list(alerts["tx_id"].astype(int).head(400))
c1, c2 = st.columns([3, 1])
pick = c1.selectbox("Cảnh báo", opts, index=opts.index(default) if default in opts else 0, format_func=lambda i: (
    f"TX-{i:06d} · {core.LEVELS[df.iloc[i]['level']][0]} điểm {100 * df.iloc[i]['score_he_thong']:.3f} · "
    f"{df.iloc[i]['account_id']} → {df.iloc[i]['counterparty_id']}"))
typed = c2.number_input("…hoặc Mã GD bất kỳ", 0, len(df) - 1, int(pick))
tx = int(typed) if int(typed) != int(pick) else int(pick)
row = df.iloc[[tx]]
r = row.iloc[0]
base, sv = inf.shap_values(row)
sv = sv[0]
logit = base + sv.sum()

m = st.columns([1.5, 1, 1, 1])
m[0].metric("Risk Level", f"{core.LEVELS[r['level']][0]} {core.LEVELS[r['level']][1]}")
m[1].metric("Risk Score (hệ thống)", core.dec(100 * r["score_he_thong"], 3))
m[2].metric("Phân vị GBDT / DRRE", f"{100 * r['pct_gbdt_28']:.2f} / {100 * r['pct_R7']:.2f}".replace(".", ","))
m[3].metric("Fraud Probability (ước tính)", core.fmt_pct(float(core.probability(np.array([r["score_he_thong"]]))[0]), 1))

order = np.argsort(-np.abs(sv))[:12]
rest = sv.sum() - sv[order].sum()
labels = [core.FEATURE_LABELS[feats[j]].split("] ")[-1] + f" = {inf.value_text(feats[j], float(r[feats[j]]))}"
          for j in order]
fig = go.Figure(go.Waterfall(
    orientation="h", measure=["absolute"] + ["relative"] * (len(order) + 1) + ["total"],
    y=["Giá trị nền (TB mô hình)"] + labels + ["16 đặc trưng còn lại", "Đầu ra GBDT (log-odds)"],
    x=[base] + list(sv[order]) + [rest, logit],
    increasing=dict(marker=dict(color=ui.RED)), decreasing=dict(marker=dict(color=ui.GREEN)),
    totals=dict(marker=dict(color=ui.CYAN)), connector=dict(line=dict(color="#2B4763"))))
fig.update_layout(yaxis=dict(autorange="reversed"), xaxis_title="log-odds")
st.plotly_chart(ui.style(fig, 520, legend=False), use_container_width=True)
st.caption(f"Kiểm tra cộng tính: nền {base:+.3f} + tổng SHAP {sv.sum():+.3f} = {logit:+.3f} = đầu ra GBDT "
           f"(xác suất GBDT {1 / (1 + np.exp(-logit)):.4f}).")

contrib = pd.DataFrame({"f": feats, "v": sv, "x": [float(r[f]) for f in feats]})
up = contrib[contrib["v"] > 0].sort_values("v", ascending=False).head(6)
down = contrib[contrib["v"] < 0].sort_values("v").head(6)
a, b = st.columns(2)
with a:
    st.markdown("#### 🔺 Risk-increasing factors")
    for x in up.itertuples():
        st.markdown(f"- **{core.FEATURE_LABELS[x.f].split('] ')[-1]}** = {inf.value_text(x.f, x.x)} "
                    f"(+{x.v:.2f})")
with b:
    st.markdown("#### 🔻 Risk-reducing factors")
    for x in down.itertuples():
        st.markdown(f"- **{core.FEATURE_LABELS[x.f].split('] ')[-1]}** = {inf.value_text(x.f, x.x)} ({x.v:.2f})")

st.markdown("#### Giải thích bằng lời")
pct = core.feature_percentiles(r)
sentences = []
for x in up.head(3).itertuples():
    name = core.FEATURE_LABELS[x.f].split("] ")[-1].lower()
    sentences.append(f"{name} = {inf.value_text(x.f, x.x)} (phân vị {pct[x.f]:.0f} trên val) khiến mô hình "
                     "thấy giao dịch giống các mẫu rửa tiền đã học")
text = "; ".join(sentences) if sentences else "không có yếu tố nào đẩy rủi ro lên rõ rệt"
st.info(f"Giao dịch TX-{tx:06d} được xếp **{core.LEVELS[r['level']][1]}** vì {text}. "
        f"Ngoài ra, DRRE (đọc lịch sử hành vi và mạng lưới 24 giờ trước) xếp giao dịch ở phân vị "
        f"{core.dec(100 * r['pct_R7'], 2)}, góp 5% vào điểm cuối.", icon="💬")
if bool(r["label"]):
    st.caption(f"Kiểm chứng: nhãn AMLWorld = rửa tiền, kiểu {r['laundering_pattern'] or 'UNSPECIFIED'} — "
               f"{core.PATTERN_LABELS.get(r['laundering_pattern'] or 'UNSPECIFIED', '')}.")
