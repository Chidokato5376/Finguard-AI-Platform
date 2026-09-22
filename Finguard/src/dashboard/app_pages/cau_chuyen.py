import streamlit as st

from src.dashboard.story import LEVELS, chain_members, fmt_money, fmt_num, get_view

source, scored, view = get_view()

st.title("Phát hiện dòng tiền lừa đảo trước khi quá muộn")
st.markdown(
    "Mỗi ngày có hàng triệu lượt chuyển tiền đi qua ngân hàng. Phần lớn là bình thường — "
    "nhưng xen giữa chúng là tiền của những người **đang bị lừa**. FinGuard AI giúp ngân hàng "
    "**nhận ra những giao dịch đó sớm hơn**, **giải thích được vì sao**, và **biết nên xử lý việc gì trước**."
)

with st.container(border=True):
    st.markdown("#### :material/newspaper: Một câu chuyện có thật")
    st.markdown(
        "Vụ án **Mr Pips**: theo [VTV](https://vtv.vn/vu-an-mr-pips-so-bi-can-tang-tu-75-len-188-nguoi-tien-lua-dao-xac-dinh-hon-1568-ty-dong-100260704120924929.htm) "
        "và [VnExpress](https://vnexpress.net/188-nguoi-bi-de-nghi-truy-to-trong-vu-mr-pips-lua-hon-1-568-ty-dong-5093351.html), "
        "số tiền lừa đảo được xác định **hơn 1.568 tỷ đồng**, với 188 người bị đề nghị truy tố. Theo "
        "[Tuổi Trẻ](https://tuoitre.vn/mr-pips-va-mr-hunter-dung-755-ti-mua-biet-thu-vang-khoi-va-ngoai-te-de-rua-tien-lua-dao-100260704210804222.htm), "
        "khoảng **755 tỷ đồng** đã được dùng mua biệt thự, vàng và ngoại tệ để “rửa” tiền."
    )
    st.markdown(
        "**Câu hỏi đặt ra:** những dòng tiền như vậy đều đi qua tài khoản ngân hàng. Làm sao để "
        "phát hiện chúng *ngay khi tiền đang chảy*, thay vì sau khi đã mất?"
    )

st.subheader("FinGuard AI làm 3 việc")
col1, col2, col3 = st.columns(3, border=True)
with col1:
    st.markdown("##### :material/psychology: 1. Nhớ thói quen")
    st.write("Mỗi người có cách tiêu tiền riêng. Hệ thống ghi nhớ thói quen đó và nhận ra khi có "
             "điều khác thường — ví dụ một khoản chuyển lớn gấp mười lần mọi khi.")
    st.caption("Giống người bán hàng quen mặt khách: khách quen bỗng mua món đắt gấp mười lần thì sẽ để ý.")
with col2:
    st.markdown("##### :material/hub: 2. Nhìn cả mạng lưới")
    st.write("Kẻ gian hiếm khi chuyển tiền một lần. Chúng chuyển qua nhiều tài khoản trung gian để "
             "xóa dấu vết. Hệ thống nhìn đường đi của tiền, không chỉ từng giao dịch riêng lẻ.")
    st.caption("Giống việc lần theo cả dãy dấu chân, thay vì chỉ nhìn một bước.")
with col3:
    st.markdown("##### :material/checklist: 3. Sắp xếp việc cần làm")
    st.write("Nhân viên ngân hàng không thể kiểm tra hết mọi cảnh báo. Hệ thống chọn ra những "
             "cảnh báo quan trọng nhất cho mỗi ca trực.")
    st.caption("Giống phòng cấp cứu: người nặng nhất được khám trước.")

st.subheader(f"Ngay lúc này, trong {fmt_num(len(view), 0)} giao dịch gần nhất")
counts = view["level"].value_counts()
with st.container(horizontal=True):
    for level in ("critical", "medium", "low"):
        info = LEVELS[level]
        st.metric(f"{info['emoji']} {info['name']}", f"{int(counts.get(level, 0))} giao dịch", border=True)
    st.metric(":material/payments: Tiền trong giao dịch cần chặn",
              fmt_money(view.loc[view["level"] == "critical", "amount"].sum(), source), border=True)

chains = chain_members(source, len(view))
if chains:
    st.info(
        f"**Nhìn thêm mạng lưới**, hệ thống thấy **{len(chains)} giao dịch** có khoản tiền bị chuyển tiếp nối "
        "tiếp qua nhiều tài khoản chỉ trong vài phút — kiểu gian lận mà cách *so với thói quen* bỏ sót, "
        "vì các tài khoản trung gian thường còn mới.", icon=":material/hub:")

st.page_link("app_pages/giam_sat.py", label="Xem các giao dịch đáng lo", icon=":material/arrow_forward:")
st.caption("FinGuard không tự ý chặn giao dịch nào: hệ thống chỉ đề xuất, "
           "**quyết định cuối cùng luôn thuộc về nhân viên ngân hàng**.")
