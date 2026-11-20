import streamlit as st

st.title("Model Insights")
st.caption("Model metrics chỉ được hiển thị khi liên kết được báo cáo evaluation với đúng checkpoint và split.")
st.info("Chưa có bảng registry/report evaluation được nối vào trang này. Vì vậy Precision, Recall, F1, ROC-AUC, PR-AUC và Confusion Matrix không được hiển thị dưới dạng số để tránh gán nhầm kết quả.", icon="ℹ️")

st.subheader("Artifact status")
col1, col2, col3 = st.columns(3)
col1.metric("Selected model", "Chưa chọn")
col2.metric("Evaluation split", "Chưa xác minh")
col3.metric("Model version", "Chưa xác minh")
st.markdown("**Thông tin cần đối chiếu:** checkpoint · model config · source/dataset · manifest · train/validation/test report.")
st.caption("Không có metric nào được suy diễn từ dữ liệu demo trên trang này.")
