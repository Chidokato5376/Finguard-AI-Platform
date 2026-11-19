import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import precision_recall_curve

import core

st.title("Chạy thử mô hình trực tiếp")
st.markdown(
    "Trang này **chấm điểm lại ngay trên máy chủ** bằng chính các mô hình đã huấn luyện: GBDT 28 cột "
    "(`gbdt_28_hi_medium.joblib`) và checkpoint DRRE R7 (`drre_r7_hi_medium.pt`, 4 epoch trên 22,3 triệu "
    "giao dịch HI-Medium). Kết quả được đối chiếu với nhãn thật để Ban Giám khảo tự kiểm tra.")

feats = core.features()
required = feats + ["account_id", "counterparty_id", "timestamp"]
tmpl = io.StringIO()
core.sample().head(2000)[required + ["amount", "currency", "label"]].to_csv(tmpl, index=False)
st.download_button("Tải tệp mẫu đúng định dạng (2.000 giao dịch)", tmpl.getvalue(), "finguard_tep_mau.csv",
                   "text/csv", icon=":material/description:")

src = st.radio("Dữ liệu đầu vào", ["Khung mẫu AMLWorld trong gói", "Tải tệp của tôi lên"], horizontal=True)

if src == "Khung mẫu AMLWorld trong gói":
    base = core.sample()
    # Giới hạn 100.000 để máy chủ miễn phí trả kết quả trong vài phút (DRRE ~800 GD/giây trên CPU 8 nhân).
    sizes = {"5.000 giao dịch đầu": 5_000, "20.000": 20_000, "50.000": 50_000, "100.000": 100_000}
    n = sizes[st.select_slider("Số giao dịch chấm", list(sizes), value="20.000")]
    batch = base.head(n)
else:
    st.caption("Tệp CSV hoặc Parquet cần các cột: " + ", ".join(f"`{c}`" for c in required) +
               ". Có thêm cột `label` (0/1) thì trang tính luôn chỉ số. Tải tệp mẫu ở đầu trang.")
    up = st.file_uploader("Chọn tệp", type=["csv", "parquet"])
    if up is None:
        st.stop()
    batch = pd.read_parquet(up) if up.name.endswith(".parquet") else pd.read_csv(up)
    missing = [c for c in required if c not in batch.columns]
    if missing:
        st.error(f"Thiếu cột: {missing}")
        st.stop()
    batch["timestamp"] = pd.to_datetime(batch["timestamp"])
    batch = batch.head(100_000)

use_drre = st.checkbox("Chạy cả DRRE R7 (chậm hơn: khoảng 1–3 phút cho 20.000 giao dịch trên máy chủ miễn phí)",
                       value=True)
if not st.button("Chấm điểm", type="primary", icon=":material/play_arrow:"):
    st.stop()

with st.spinner("Đang chấm điểm..."):
    scored, timing = core.run_live(batch, use_drre=use_drre)

c = st.columns(4)
c[0].metric("Giao dịch đã chấm", core.fmt_num(len(scored)), border=True)
c[1].metric("GBDT 28", f"{timing['gbdt_28']:.1f} s", f"{len(scored) / max(timing['gbdt_28'], 1e-9):,.0f} GD/giây".replace(",", "."),
            delta_color="off", border=True)
if use_drre:
    c[2].metric("DRRE R7", f"{timing['R7']:.1f} s", f"{len(scored) / max(timing['R7'], 1e-9):,.0f} GD/giây".replace(",", "."),
                delta_color="off", border=True)
lv = scored["live_level"].value_counts()
c[3].metric("Cảnh báo (🔴 + 🟡)", f"{int(lv.get('critical', 0))} + {int(lv.get('medium', 0))}", border=True)

if "label" in scored and scored["label"].fillna(False).astype(bool).any():
    st.subheader("Chỉ số trên lô vừa chấm (so với nhãn thật)")
    cols = {"Hệ thống (chạy trực tiếp)": "live_he_thong", "GBDT 28 (chạy trực tiếp)": "live_gbdt_28"}
    if use_drre:
        cols["DRRE R7 (chạy trực tiếp, bộ nhớ khởi động lạnh)"] = "live_R7"
    if "score_he_thong" in scored:
        cols["Hệ thống (điểm trong gói, đủ lịch sử)"] = "score_he_thong"
        cols["DRRE R7 (điểm trong gói, đủ lịch sử)"] = "score_R7"
    table = core.evaluate(scored, cols)
    st.dataframe(table.style.format({"AUPRC": "{:.4f}", "ROC-AUC": "{:.4f}", "Độ chính xác @100": "{:.0%}",
                                     "Recall @ top 1%": "{:.0%}"}), hide_index=True)
    st.caption(
        "Điểm GBDT chạy trực tiếp phải trùng điểm trong gói (cùng mô hình, cùng đặc trưng). DRRE chạy "
        "trực tiếp bắt đầu với bộ nhớ hành vi TRỐNG nên thấp hơn bản trong gói, vốn được nối trạng thái từ "
        "đầu tập train — đây là đặc tính của mô hình có trạng thái, không phải lỗi.")
    if "score_gbdt_28" in scored:
        gap = float(np.max(np.abs(scored["live_gbdt_28"] - scored["score_gbdt_28"])))
        st.caption(f"Kiểm tra tái lập: lệch lớn nhất giữa GBDT chạy trực tiếp và điểm trong gói = {gap:.2e}.")

    keep = scored["eval_universe"].fillna(True).to_numpy(bool) if "eval_universe" in scored else np.ones(len(scored), bool)
    y = scored["label"].fillna(False).astype(bool).to_numpy()[keep]
    fig = go.Figure()
    for name, col in cols.items():
        p, r, _ = precision_recall_curve(y, scored[col].to_numpy()[keep])
        fig.add_trace(go.Scatter(x=r, y=p, mode="lines", name=name))
    fig.update_layout(height=380, xaxis_title="Recall (tỷ lệ vụ bắt được)", yaxis_title="Precision (độ chính xác)",
                      margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Kết quả chấm")
show = scored.sort_values("live_he_thong", ascending=False)
keep_cols = [c for c in ["timestamp", "account_id", "counterparty_id", "amount", "currency", "live_level",
                         "live_he_thong", "live_gbdt_28", "live_R7", "label", "laundering_pattern"] if c in show]
st.dataframe(show[keep_cols].head(300), hide_index=True, height=320)
buf = io.StringIO()
show[keep_cols].to_csv(buf, index=False)
st.download_button("Tải kết quả chấm (CSV)", buf.getvalue(), "finguard_ket_qua_cham.csv", "text/csv")

