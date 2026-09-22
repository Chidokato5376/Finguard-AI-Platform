import streamlit as st

from src.dashboard.story import FORCE_SELECT_MIN, LOW_MAX, MEDIUM_MAX, MINUTES_PER_COST_UNIT, get_view, method_label

source, scored, view = get_view()

st.title("Hệ thống hoạt động thế nào?")
st.markdown("Từ lúc một giao dịch xuất hiện đến lúc nhân viên ngân hàng nhận được đề xuất, "
            "FinGuard đi qua 5 bước:")

steps = [
    (":material/swap_horiz:", "1. Giao dịch mới đến", "Ai chuyển, chuyển cho ai, bao nhiêu tiền, lúc nào."),
    (":material/psychology:", "2. So với thói quen", "Khoản này có giống cách khách hàng thường tiêu tiền không?"),
    (":material/hub:", "3. Nhìn mạng lưới", "Tiền đi tiếp đến đâu? Có chuyển nối qua nhiều tài khoản không?"),
    (":material/speed:", "4. Chấm điểm và giải thích", "Mức độ đáng ngờ từ 0 đến 100, kèm lý do bằng lời."),
    (":material/checklist:", "5. Sắp xếp cho ca trực", "Chọn cảnh báo nên xử lý trước trong số giờ có hạn."),
]
with st.container(horizontal=True):
    for icon, title, text in steps:
        with st.container(border=True):
            st.markdown(f"**{icon} {title}**")
            st.caption(text)

st.subheader("Điều hệ thống không làm")
st.markdown(
    "- **Không tự khóa tài khoản hay chặn giao dịch.** Hệ thống chỉ đề xuất, nhân viên ngân hàng quyết định.\n"
    "- **Không đưa ra kết luận mà không có lý do.** Mỗi cảnh báo đều kèm giải thích bằng lời.\n"
    "- **Không dùng dữ liệu thật của khách hàng trong bản trình diễn này.**"
)

st.subheader("Cách chấm điểm đang dùng trên trang này")
st.markdown(f"**{method_label(scored)}.**")
if not (scored["method"] == "drre_trained").any():
    st.markdown(
        "Với mỗi giao dịch, hệ thống so số tiền với các giao dịch trước đó **của chính khách hàng đó**, "
        "và xem người nhận có phải người lạ không. Cách chấm này đơn giản, nhưng mọi con số đều giải thích được. "
        "Mô hình AI đầy đủ (DRRE) đang được nhóm kiểm chứng trên dữ liệu thật."
    )

st.subheader("Những điều chúng tôi nói thẳng")
st.markdown(
    "- Dữ liệu trên trang là **dữ liệu mô phỏng**: dữ liệu giao dịch thật của khách hàng Việt Nam không được "
    "công khai vì lý do bảo vệ dữ liệu cá nhân.\n"
    "- Mạng lưới tiền chỉ được dựng từ **các giao dịch đang xem**, chưa phải toàn bộ lịch sử.\n"
    "- Phần lượng tử hiện là **bước chuẩn bị**: bài toán đã được viết đúng dạng cho máy tính lượng tử, "
    "nhưng ở quy mô hiện tại máy tính thông thường vẫn nhanh hơn."
)

with st.expander("Chi tiết kỹ thuật (dành cho người trong ngành)", icon=":material/code:"):
    st.markdown(
        f"- **Phân tầng:** mức 0–{LOW_MAX} = thấp, {LOW_MAX + 1}–{MEDIUM_MAX} = trung bình, "
        f"{MEDIUM_MAX + 1}–100 = cao.\n"
        "- **Quy tắc thống kê (Heuristic Scorer):** `risk_score = 100·σ(z + 0,8·[người nhận mới và đã có ≥ 3 "
        "giao dịch] − 1,2)`, với *z* là độ lệch của log(số tiền) so với lịch sử của chính tài khoản.\n"
        "- **DRRE (Dynamic Risk Representation Engine):** Behavior Memory (GRU + K ô nhớ), Context-aware "
        "Attention (độ lệch ngữ cảnh δ_t), Graph Encoder (GraphSAGE), Adaptive Fusion (trọng số β theo "
        "nguồn: giao dịch / hành vi / mạng lưới).\n"
        "- **QAPE (Quantum Alert Prioritization Engine):** bài toán knapsack 0/1 — tối đa Σ rủi ro với ràng "
        f"buộc thời gian; cảnh báo từ mức {FORCE_SELECT_MIN} bắt buộc chọn. Trang *Xử lý gì trước* giải bằng "
        "quy hoạch nguyên (ILP); bản đầy đủ có chuyển sang QUBO và giải bằng QAOA (Qiskit) để so sánh.\n"
        f"- **Quy đổi thời gian:** 1 đơn vị chi phí của QAPE ≈ {MINUTES_PER_COST_UNIT} phút làm việc; chi phí "
        "mỗi cảnh báo = 1 + log(1 + số tiền)/10 — đây là giả định đơn giản hóa, chưa phải số đo thực tế.\n"
        "- **Tài khoản** được hiển thị dạng che số (TK ••• xxxxxx) như trên sao kê ngân hàng."
    )
