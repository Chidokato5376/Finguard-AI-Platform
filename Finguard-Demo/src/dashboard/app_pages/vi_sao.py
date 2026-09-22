import pandas as pd
import streamlit as st

from src.dashboard.story import (
    CHANNEL_NAMES, LEVELS, chain_members, explain, flow_figure, fmt_money, fmt_when, get_view,
    mask_account, money_flow, tx_label,
)

source, scored, view = get_view()
chains = chain_members(source, len(view))

st.title("Vì sao hệ thống nghi ngờ giao dịch này?")

MODES = ["Giao dịch đáng ngờ", "Dòng tiền nối tiếp", "Tất cả giao dịch"]
incoming = st.session_state.pop("queue_pick", None)  # giao dịch vừa bấm ở trang danh sách (dùng một lần)
if incoming in set(view["tx_id"]):
    level = view.loc[view["tx_id"] == incoming, "level"].iloc[0]
    st.session_state["pick_mode"] = MODES[0] if level != "low" else MODES[1] if incoming in chains else MODES[2]
    st.session_state["selected_tx"] = incoming
st.session_state.setdefault("pick_mode", MODES[0])

mode = st.segmented_control("Chọn trong", MODES, key="pick_mode")
if mode == "Dòng tiền nối tiếp":
    pool = view[view["tx_id"].isin(chains)]
elif mode == "Tất cả giao dịch":
    pool = view
else:
    pool = view[view["level"] != "low"]
pool = pool.sort_values("risk_score", ascending=False)
if pool.empty:
    st.success("Không có giao dịch nào trong nhóm này.", icon=":material/check_circle:")
    st.stop()

options = pool["tx_id"].tolist()
if st.session_state.get("selected_tx") not in options:
    st.session_state["selected_tx"] = options[0]
labels = {t: tx_label(r, source) + ("  🔗" if t in chains else "")
          for t, r in zip(pool["tx_id"], pool.to_dict("records"))}
tx = st.selectbox("Chọn một giao dịch", options, format_func=labels.get, key="selected_tx")
row = scored.loc[scored["tx_id"] == tx].iloc[0]
info = LEVELS[row["level"]]
network_only = tx in chains and row["level"] == "low"

left, right = st.columns([1, 1], gap="large")
with left:
    channel = CHANNEL_NAMES.get(str(row.get("channel")), str(row.get("channel")))
    st.markdown(
        f"Lúc **{fmt_when(row)}**, tài khoản **{mask_account(row['account_id'])}** chuyển "
        f"**{fmt_money(row['amount'], source)}** qua {channel} cho **{mask_account(row['counterparty_id'])}**."
    )
    with st.container(border=True):
        st.metric("Mức độ đáng ngờ", f"{row['risk_score']:.0f} / 100")
        st.progress(min(max(row["risk_score"] / 100, 0.0), 1.0))
        st.markdown(f"**Đề xuất theo thói quen: {info['emoji']} {info['name']}**")
        if tx in chains:
            st.markdown("**Dấu hiệu mạng lưới: 🔗 khoản tiền bị chuyển tiếp nối tiếp**")
    if network_only:
        st.warning(
            "Nếu chỉ so với thói quen, giao dịch này trông **bình thường** — tài khoản còn mới nên chưa có gì để "
            "so. Nhưng khi **nhìn mạng lưới** (bên phải), khoản tiền bị chuyển tiếp liên tục qua nhiều tài khoản: "
            "dấu hiệu tài khoản trung gian. Đây chính là loại gian lận mà việc chỉ nhìn từng giao dịch sẽ bỏ sót.",
            icon=":material/hub:")

    reasons, signs = explain(scored, row, source)
    st.markdown("##### Những điều khiến hệ thống chú ý")
    for icon, title, text in reasons:
        with st.container(border=True):
            st.markdown(f"**{icon} {title}**")
            st.markdown(text)

    if row.get("method") == "drre_trained":
        st.markdown("##### Hệ thống dựa vào đâu nhiều nhất")
        weights = pd.DataFrame({
            "Nguồn thông tin": ["Thói quen của khách hàng", "Mối quan hệ giữa các tài khoản", "Bản thân giao dịch"],
            "Tỷ trọng (%)": [100 * row["beta_behavior"], 100 * row["beta_graph"], 100 * row["beta_transaction"]],
        })
        st.bar_chart(weights, x="Nguồn thông tin", y="Tỷ trọng (%)", horizontal=True, height=180)

    if signs:
        st.markdown("##### Dấu hiệu khác nên kiểm tra thêm")
        for sign in signs:
            st.markdown(f"- {sign}")
        st.caption("Những dấu hiệu này được ghi nhận để nhân viên tham khảo, không tính vào mức độ đáng ngờ.")

    st.markdown("##### Nên làm gì?")
    if network_only:
        st.warning("Đưa vào danh sách kiểm tra: xác minh chủ của các tài khoản trong chuỗi chuyển tiền, "
                   "tạm giữ nếu không giải thích được nguồn tiền.", icon=":material/support_agent:")
    else:
        {"critical": st.error, "medium": st.warning, "low": st.success}[row["level"]](
            info["action"], icon=":material/support_agent:")

with right:
    st.markdown("##### Tiền đang chảy đi đâu?")
    st.caption("Mỗi chấm tròn là một tài khoản, mũi tên là chiều chuyển tiền. Chấm lớn màu xanh đậm là "
               "người chuyển của giao dịch đang xem, mũi tên đỏ đậm là đường đi của khoản tiền này. Màu các "
               "chấm khác cho biết mức độ đáng ngờ cao nhất của tài khoản đó.")
    flow = money_flow(view, row)
    if flow["nodes"]:
        st.plotly_chart(flow_figure(flow, row["account_id"]), config={"displayModeBar": False})
    for sentence in flow["sentences"]:
        st.markdown(sentence)
    st.caption("Mạng lưới chỉ dựng từ các giao dịch đang xem (chỉnh ở thanh bên), không phải toàn bộ lịch sử.")
