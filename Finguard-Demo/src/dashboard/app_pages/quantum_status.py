import streamlit as st

st.title("Quantum Optimization")
st.caption("Tối ưu thứ tự xử lý cảnh báo — không phải mô hình dự đoán gian lận.")
st.info("Trạng thái hiện tại: trang demo có thể dùng bộ tối ưu cổ điển (Greedy/ILP) ở màn hình ưu tiên. QAOA không được gọi trong luồng dashboard tương tác; quantum kernel và VQC đang tắt theo cấu hình. Không có kết quả quantum nào được giả lập ở đây.", icon="⚛️")

st.subheader("Pipeline đã có trong repository")
for title, detail in [
    ("Candidate alerts", "Cảnh báo lấy từ kết quả scoring hiện tại."),
    ("QUBO formulation", "Có module QAPE/QUBO; đầu vào và giả định chi phí phải được công bố."),
    ("Classical solvers", "Dashboard ưu tiên Greedy/ILP khi có dữ liệu cảnh báo."),
    ("Quantum comparison", "QAOA/benchmark chưa được tích hợp để chạy và hiển thị tại trang này."),
]:
    with st.container(border=True):
        st.markdown(f"**{title}**")
        st.caption(detail)
