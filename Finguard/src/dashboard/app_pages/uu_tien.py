import pandas as pd
import streamlit as st

from src.dashboard.story import (
    FORCE_SELECT_MIN, LEVELS, fmt_money, fmt_num, get_view, mask_account, plan_shift,
)

source, scored, view = get_view()

st.title("Ca trực nên xử lý gì trước?")
st.markdown(
    "Một nhân viên kiểm soát rủi ro **không thể kiểm tra mọi cảnh báo** — mỗi ca chỉ có số giờ nhất định, "
    "và mỗi cảnh báo cần thời gian gọi điện, đối chiếu, xác minh. Câu hỏi là: **nên kiểm tra những cảnh báo "
    "nào để ngăn được nhiều rủi ro nhất?**"
)

alerts = view[view["level"] != "low"]
if len(alerts) < 2:
    st.info("Hiện có quá ít cảnh báo để cần sắp xếp. Tăng số giao dịch đang xem ở thanh bên.")
    st.stop()

with st.container(border=True, horizontal=True):
    top_n = st.slider("Số cảnh báo đang chờ", 2, min(40, len(alerts)), min(20, len(alerts)))
    hours = st.slider("Ca trực có bao nhiêu giờ làm việc?", 1, 8, 2)

cands, meta = plan_shift(alerts, top_n=top_n, hours=hours)
now, later = cands[cands["handle_now"]], cands[~cands["handle_now"]]

st.markdown(
    f"Với **{hours} giờ** làm việc, ca trực kịp xử lý **{len(now)} / {len(cands)} cảnh báo**. "
    "Hệ thống chọn những cảnh báo dưới đây để **ngăn được nhiều rủi ro nhất** trong khoảng thời gian đó:"
)
if meta["not_enough_time"]:
    st.warning(
        f"Riêng các cảnh báo khẩn cấp (mức từ {FORCE_SELECT_MIN} trở lên) đã cần khoảng "
        f"**{fmt_num(meta['must_hours'])} giờ** — nhiều hơn thời gian của ca trực. Hệ thống đang ưu tiên "
        "những cảnh báo nặng nhất trước; nên bổ sung người hoặc kéo dài ca.", icon=":material/warning:")


def _list(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "Mức": [f"{LEVELS[lv]['emoji']} {LEVELS[lv]['name']}" for lv in df["level"]],
        "Tài khoản": df["account_id"].map(mask_account).to_numpy(),
        "Số tiền": [fmt_money(a, source) for a in df["amount"]],
        "Thời gian xử lý": [f"~{m:.0f} phút" for m in df["minutes"]],
    })


col1, col2 = st.columns(2, border=True)
with col1:
    st.markdown(f"**:material/task_alt: Xử lý ngay trong ca này — {len(now)} cảnh báo** "
                f"(~{now['minutes'].sum() / 60:.1f} giờ)".replace(".", ","))
    st.dataframe(_list(now), hide_index=True)
with col2:
    st.markdown(f"**:material/schedule: Chuyển cho ca sau — {len(later)} cảnh báo**")
    st.dataframe(_list(later), hide_index=True)

st.subheader("So với cách làm thông thường: xử lý theo thứ tự đến")
st.markdown("Nếu không có FinGuard, nhân viên thường xử lý cảnh báo **đến trước làm trước**. "
            "Với cùng số giờ làm việc, hai cách cho kết quả như sau:")

fifo = cands[cands["fifo"]]
red_now, red_fifo = int((now["level"] == "critical").sum()), int((fifo["level"] == "critical").sum())
with st.container(horizontal=True):
    st.metric("Cảnh báo đỏ được xử lý — theo FinGuard", f"{red_now}",
              f"{red_now - red_fifo:+d} so với làm theo thứ tự", border=True)
    if "label" in cands.columns:
        fraud_now, fraud_fifo = int(now["label"].sum()), int(fifo["label"].sum())
        st.metric("Vụ lừa đảo thật được chặn — theo FinGuard", f"{fraud_now} / {int(cands['label'].sum())}",
                  f"{fraud_now - fraud_fifo:+d} so với làm theo thứ tự", border=True)
    else:
        total = cands["risk_score"].sum()
        share_now, share_fifo = now["risk_score"].sum() / total, fifo["risk_score"].sum() / total
        st.metric("Phần rủi ro được xử lý — theo FinGuard", f"{share_now:.0%}",
                  f"{(share_now - share_fifo) * 100:+.0f} điểm % so với làm theo thứ tự", border=True)

if "label" in cands.columns:
    compare = pd.DataFrame({
        "Cách làm": ["Theo thứ tự đến", "Theo FinGuard"],
        "Tiền lừa đảo chặn được": [fifo.loc[fifo["label"] == 1, "amount"].sum(),
                                   now.loc[now["label"] == 1, "amount"].sum()],
    })
    st.bar_chart(compare, x="Cách làm", y="Tiền lừa đảo chặn được", horizontal=True, height=160)
    st.caption("Dữ liệu mô phỏng có sẵn đáp án (giao dịch nào thực sự là lừa đảo), nên có thể so sánh công bằng "
               "hai cách làm. Kết quả thay đổi theo số giờ làm việc và số cảnh báo bạn chọn ở trên.")

with st.container(border=True):
    st.markdown("#### :material/memory: Máy tính lượng tử nằm ở đâu trong chuyện này?")
    combos = fmt_num(2 ** len(cands), 0)
    st.markdown(
        f"Chọn cảnh báo nào để xử lý là một **bài toán chọn tổ hợp**: với {len(cands)} cảnh báo có tới "
        f"**{combos} cách chọn** khác nhau, và con số này tăng gấp đôi mỗi khi có thêm một cảnh báo. "
        "Nhóm đã viết bài toán này theo đúng dạng mà **máy tính lượng tử** có thể giải (thuật toán QAOA)."
    )
    st.markdown(
        "Nói thẳng: ở quy mô một ca trực như trên, **máy tính thông thường vẫn giải nhanh hơn** — trang này đang "
        "dùng máy tính thông thường. Giá trị của cách làm này là khi số cảnh báo và số ràng buộc tăng lên "
        "(nhiều nhóm nhân viên, nhiều chi nhánh), đây chính là loại bài toán mà máy tính lượng tử hứa hẹn giúp được."
    )
    st.caption(f"Giả định: mỗi cảnh báo mất khoảng 20–30 phút xử lý, tùy số tiền. Cảnh báo từ mức {FORCE_SELECT_MIN} "
               "trở lên luôn được ưu tiên xử lý.")
