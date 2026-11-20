from __future__ import annotations

import pandas as pd
import streamlit as st

from src.dashboard.data_service import load_processed_split
from src.dashboard.story import SOURCE_LABELS

st.title("Transaction Analysis")
st.caption("Kiểm tra dữ liệu và xem trước file CSV. Inference trực tiếp trên file tải lên chưa được bật trong bản giao diện này.")
st.warning("Không chạy scoring trên upload ở phiên bản UI preview. Điều này tránh trình bày kết quả heuristic hoặc preprocessing không tương thích như kết quả model AI.", icon="⚠️")

mode = st.radio("Chọn cách nhập", ["Tải file CSV", "Chọn dữ liệu demo đã tiền xử lý"], horizontal=True)
if mode == "Tải file CSV":
    uploaded = st.file_uploader("Chọn CSV giao dịch", type=["csv"], help="Tải lên để kiểm tra header và preview. File chưa được đưa qua model.")
    if uploaded is not None:
        try:
            data = pd.read_csv(uploaded, nrows=5000)
            st.success(f"Đọc được {len(data):,} dòng preview và {len(data.columns)} cột (giới hạn xem trước 5.000 dòng).")
            st.write("**Các trường tìm thấy**", ", ".join(map(str, data.columns)))
            st.dataframe(data.head(20), use_container_width=True, hide_index=True)
            required = {"account_id", "counterparty_id", "amount", "channel", "timestamp"}
            missing = sorted(required - set(data.columns))
            if missing:
                st.warning("Thiếu các trường Unified Schema thường dùng: " + ", ".join(missing) + ". Schema này chỉ là kiểm tra tham khảo, chưa chứng minh tương thích với checkpoint cụ thể.")
            else:
                st.info("Có các trường cốt lõi Unified Schema. Cần xác định model/source và tái sử dụng preprocessing tương ứng trước khi inference.")
        except Exception as exc:
            st.error(f"Không đọc được CSV: {exc}")
else:
    sources = st.session_state.get("source")
    source = st.selectbox("Nguồn", [sources] if sources else [], format_func=lambda s: SOURCE_LABELS.get(s, s))
    if source:
        try:
            data = load_processed_split(source, "test")
            st.success(f"Nguồn {SOURCE_LABELS.get(source, source)} · test split · {len(data):,} dòng")
            st.write("**Schema thực tế:**", ", ".join(data.columns))
            st.dataframe(data.head(20), use_container_width=True, hide_index=True)
            st.caption("Xem trước dữ liệu đã tiền xử lý; chưa chạy inference từ trang này.")
        except Exception as exc:
            st.error(f"Không tải được split: {exc}")

st.markdown("#### Trạng thái tích hợp")
steps = [
    ("Input preview", "Có — CSV và test split"),
    ("Schema validation", "Cơ bản — kiểm tra header/Unified Schema"),
    ("Model inference", "Chưa kết nối cho dữ liệu upload"),
    ("SHAP explanation", "Chưa chạy trên upload"),
    ("CSV report export", "Chưa có kết quả inference để xuất"),
]
st.dataframe(pd.DataFrame(steps, columns=["Capability", "Status"]), hide_index=True, use_container_width=True)
