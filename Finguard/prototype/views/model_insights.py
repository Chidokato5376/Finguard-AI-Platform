import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import core
import inference as inf
import ui

res = core.results()
boards = {"HI-Medium": res["leaderboard_hi_medium"], "HI-Small": res["leaderboard_hi_small"]}

ui.header("Model Insights", "Kết quả đánh giá đã lưu của các mô hình đã huấn luyện — không sinh thêm số. Phân biệt rõ "
          "train / validation / test.")

st.subheader("Model version")
v = st.columns(3)
with v[0]:
    ui.card("DRRE R7 · drre_r7_hi_medium.pt",
            "DRRE v2: bộ nhớ hành vi GRU + đồ thị giao dịch hai chiều chỉ dùng 24 giờ quá khứ (3 lớp, fanout 10), "
            "28 đặc trưng. Train 4 epoch trên HI-Medium (seed 42, CPU, ~20 giờ); checkpoint chọn theo điểm làm mượt = "
            "<b>epoch 3</b>.")
with v[1]:
    ui.card("GBDT 28 · gbdt_28_hi_medium.joblib",
            "HistGradientBoosting (scikit-learn 1.4.2), 200 cây, cân bằng lớp, không dừng sớm, seed 42. "
            "Học lại đúng cấu hình bảng xếp hạng — điểm khớp tới sai số 3×10⁻⁸.")
with v[2]:
    ui.card("Hệ thống FINGUARD · v2026.11",
            f"Trộn thứ hạng {100 * (1 - res['w_drre']):.0f}% GBDT + {100 * res['w_drre']:.0f}% DRRE (trọng số chọn trên "
            "nửa đầu val). 3 mức cảnh báo, ngưỡng chọn trên val. Giải thích: SHAP TreeExplainer.")

st.subheader("Dataset information")
data = pd.DataFrame([
    ("HI-Small", "Train", "01/09 – 07/09/2022", 3_554_065, 2_856, 2_956_734),
    ("HI-Small", "Validation", "07/09 – 09/09/2022", 761_585, 760, 733_086),
    ("HI-Small", "Test", "09/09 – 10/09/2022", 761_587, 906, 734_506),
    ("HI-Medium", "Train", "01/09 – 12/09/2022", 22_323_875, 21_077, 19_620_681),
    ("HI-Medium", "Validation", "12/09 – 15/09/2022", 4_783_687, 5_489, 4_603_545),
    ("HI-Medium", "Test", "15/09 – 16/09/2022", 4_783_689, 4_580, 4_601_640),
], columns=["Bộ dữ liệu", "Tập", "Thời gian", "Giao dịch", "Vụ rửa tiền", "Vũ trụ chấm điểm"])
st.dataframe(data, hide_index=True, use_container_width=True)
st.caption("AMLWorld (IBM, Kaggle `ealtman2019/ibm-transactions-for-anti-money-laundering-aml`). Chia 70/15/15 THEO THỜI "
           "GIAN; vũ trụ chấm điểm loại các ngày có khối lượng bất thường (lỗi đã biết của bộ dữ liệu).")

split = st.radio("Tập đánh giá", ["Validation", "Train", "Test"], horizontal=True)
if split == "Train":
    st.info("Không báo cáo chỉ số trên tập train: mô hình đã thấy dữ liệu này nên số sẽ lạc quan và dễ gây hiểu nhầm. "
            "Theo dõi quá trình học bằng AUPRC trên validation sau mỗi epoch (biểu đồ dưới).", icon="ℹ️")
    fig = go.Figure()
    for name, ys in res["curves"].items():
        fig.add_trace(go.Scatter(x=list(range(1, len(ys) + 1)), y=ys, mode="lines+markers", name=name,
                                 line=dict(dash="dot" if name.startswith("v1") else "solid")))
    fig.update_layout(xaxis_title="Epoch", yaxis_title="Val AUPRC (đo trong lúc train)")
    st.plotly_chart(ui.style(fig, 380), use_container_width=True)
    st.stop()
if split == "Test":
    st.warning("Chưa đánh giá trên tập test. Tập test được giữ nguyên, chỉ chạy MỘT lần sau khi chốt mô hình "
               "(bước R10) để tránh chọn mô hình theo test. Trang này không hiển thị số nào cho test.", icon="🔒")
    st.stop()

which = st.radio("Bộ dữ liệu", list(boards), horizontal=True)
b = boards[which]
names = sorted(b["models"], key=lambda n: b["ranks"]["auprc"][n])
rows = []
for n in names:
    m = b["models"][n]
    rows.append({"Mô hình": n, "Precision": m["precision_at_f1"], "Recall": m["recall_at_f1"], "F1-score": m["f1_best"],
                 "ROC-AUC": m["roc_auc"], "PR-AUC (AUPRC)": m["auprc"], "Bắt @500": m["caught@500"],
                 "Bắt @2000": m["caught@2000"], "Hạng TB": b["mean_rank"][n]})
st.dataframe(pd.DataFrame(rows).style.format({c: "{:.4f}" for c in ["Precision", "Recall", "F1-score", "ROC-AUC",
                                                                        "PR-AUC (AUPRC)"]} | {"Hạng TB": "{:.2f}"}),
             hide_index=True, use_container_width=True)
st.caption(f"Validation {which}: {core.fmt_num(b['n_rows'])} giao dịch, {core.fmt_num(b['n_pos'])} vụ (tỷ lệ nền "
           f"{core.dec(100 * b['base_rate'], 3)}%). Precision/Recall/F1 tại ngưỡng F1 tốt nhất (chọn trên chính val nên "
           "hơi lạc quan). R* = DRRE; gbdt/rf/logistic cùng 28 cột; gbdt_20 = 20 cột gốc; ngau_nhien = chấm ngẫu nhiên.")

st.subheader("Confusion Matrix")
mats = {}
if which == "HI-Medium":
    mats["Hệ thống · mức Chặn ngay"] = inf.confusion_at_level("critical")
    mats["Hệ thống · từ mức Cần xác minh"] = inf.confusion_at_level("medium")
mats["GBDT 28 · ngưỡng F1 tốt nhất"] = inf.confusion_at_f1(b, "gbdt_28")
drre = "R7" if which == "HI-Medium" else "R5"
mats[f"DRRE {drre} · ngưỡng F1 tốt nhất"] = inf.confusion_at_f1(b, drre)
cols = st.columns(len(mats))
for col, (title, cm) in zip(cols, mats.items()):
    z = [[cm["TN"], cm["FP"]], [cm["FN"], cm["TP"]]]
    fig = px.imshow(z, x=["Dự đoán: bình thường", "Dự đoán: đáng ngờ"], y=["Thật: bình thường", "Thật: rửa tiền"],
                    text_auto=",d", color_continuous_scale=[[0, "#0E1B2D"], [1, "#0891B2"]], zmax=max(cm["TP"], cm["FP"]) * 3)
    fig.update_layout(title=dict(text=title, font=dict(size=13)), coloraxis_showscale=False)
    col.plotly_chart(ui.style(fig, 300, legend=False), use_container_width=True)
    p = cm["TP"] / max(cm["TP"] + cm["FP"], 1)
    r = cm["TP"] / max(cm["TP"] + cm["FN"], 1)
    col.caption(f"Precision {core.fmt_pct(p, 1)} · Recall {core.fmt_pct(r, 1)} · cảnh báo "
                f"{core.fmt_num(cm['TP'] + cm['FP'])}")

st.subheader("DRRE bổ sung thông tin cho GBDT (P2, không rò rỉ)")
p2 = res["p2_hi_medium"]
ci = p2["vs_base_ci95"]["tron_thu_hang"]
k = st.columns(4)
k[0].metric("PR-AUC GBDT 28 (nửa sau val)", core.dec(p2["auprc_b"]["gbdt_28"], 4))
k[1].metric("PR-AUC hệ thống (trộn 5% DRRE)", core.dec(p2["auprc_b"]["tron_thu_hang"], 4),
            f"+{core.dec(p2['auprc_b']['tron_thu_hang'] - p2['auprc_b']['gbdt_28'], 4)}")
k[2].metric("KTC 95% bootstrap ghép cặp", f"[{core.dec(ci['lo'], 4, True)}; {core.dec(ci['hi'], 4, True)}]")
k[3].metric("Trọng số DRRE (chọn trên nửa đầu)", core.dec(p2["rank_weight_drre"], 2))

st.subheader("Thang cải tiến DRRE (ablation, validation)")
small, med = boards["HI-Small"]["models"], boards["HI-Medium"]["models"]
ladder = pd.DataFrame([
    ("R2", "DRRE v1 + 28 đặc trưng", small["R2"]["auprc"]), ("R3", "+ tầng chấm điểm v2", small["R3"]["auprc"]),
    ("R4", "+ đồ thị chỉ dùng quá khứ (bỏ nhìn trước)", small["R4"]["auprc"]),
    ("R5", "+ đồ thị hai chiều có đặc trưng cạnh", small["R5"]["auprc"]),
    ("R7", "R5 trên HI-Medium (dữ liệu ×6)", med["R7"]["auprc"]),
], columns=["Bậc", "Thay đổi", "PR-AUC"])
fig = px.bar(ladder, x="Bậc", y="PR-AUC", text="PR-AUC", hover_data=["Thay đổi"], color_discrete_sequence=[ui.CYAN])
fig.update_traces(texttemplate="%{text:.3f}")
st.plotly_chart(ui.style(fig, 300, legend=False), use_container_width=True)
ui.source_note("reports/leaderboard/processed_aml_v2_val, reports/r7/leaderboard/processed_hi_medium_val, "
               "reports/r7/leaderboard/.../p2_stacking_R7.json (đóng gói trong prototype/bundle/results.json).")
