import pandas as pd
import streamlit as st

from src.dashboard.story import LEVELS, chain_members, fmt_money, fmt_when, get_view, mask_account

source, scored, view = get_view()
chains = chain_members(source, len(view))

st.title("Giao dịch nào đang đáng lo?")
st.markdown(
    "Mỗi giao dịch được chấm một **mức độ đáng ngờ từ 0 đến 100**. Càng cao, giao dịch càng giống "
    "lừa đảo. Dựa vào con số này, hệ thống đề xuất một trong ba cách xử lý:"
)
for col, (level, band) in zip(st.columns(3, border=True),
                              (("low", "0 – 30"), ("medium", "31 – 70"), ("critical", "71 – 100"))):
    info = LEVELS[level]
    with col:
        st.markdown(f"**{info['emoji']} {info['name']}** · mức {band}")
        st.caption(info["action"])

counts = view["level"].value_counts()
with st.container(horizontal=True):
    st.metric("Giao dịch đang xem", f"{len(view):,}".replace(",", "."), border=True)
    st.metric("🔴 Chặn ngay", int(counts.get("critical", 0)), border=True)
    st.metric("🟡 Cần xác minh", int(counts.get("medium", 0)), border=True)
    st.metric(":material/payments: Tiền đang bị nghi ngờ",
              fmt_money(view.loc[view["level"] != "low", "amount"].sum(), source), border=True)
    st.metric("🔗 Dòng tiền nối tiếp", len(chains), border=True)

show = st.segmented_control(
    "Hiển thị", ["Cần chú ý", "Chỉ loại chặn ngay", "Dòng tiền nối tiếp", "Tất cả"],
    default="Cần chú ý", key="queue_filter")
table = view.sort_values("risk_score", ascending=False)
if show == "Cần chú ý":
    table = table[table["level"] != "low"]
elif show == "Chỉ loại chặn ngay":
    table = table[table["level"] == "critical"]
elif show == "Dòng tiền nối tiếp":
    table = view[view["tx_id"].isin(chains)]
    st.info(
        "Đây là những giao dịch mà **khoản tiền bị chuyển tiếp liên tục qua nhiều tài khoản chỉ trong vài phút**, "
        "số tiền gần như giữ nguyên — dấu hiệu của tài khoản trung gian. Nhiều giao dịch trong số này có mức "
        "đáng ngờ **thấp**: tài khoản còn mới nên cách *so với thói quen* không có gì để so. Đó chính là lý do "
        "FinGuard phải **nhìn cả mạng lưới**, không chỉ từng giao dịch.", icon=":material/hub:")

if table.empty:
    st.success("Không có giao dịch nào cần chú ý trong nhóm này.", icon=":material/check_circle:")
    st.stop()

display = pd.DataFrame({
    "Mức": [f"{LEVELS[lv]['emoji']} {LEVELS[lv]['name']}" for lv in table["level"]],
    "Mức độ đáng ngờ": table["risk_score"].round(0).to_numpy(),
    "Người chuyển": table["account_id"].map(mask_account).to_numpy(),
    "Người nhận": table["counterparty_id"].map(mask_account).to_numpy(),
    "Số tiền": [fmt_money(a, source) for a in table["amount"]],
    "Thời điểm": [fmt_when(r) for _, r in table.iterrows()],
    "Vì sao": table["reason"].to_numpy(),
    "Dòng tiền": ["🔗 Chuyển tiếp nối tiếp" if t in chains else "" for t in table["tx_id"]],
})
st.caption("Bấm vào một dòng để xem hệ thống giải thích vì sao.")
event = st.dataframe(
    display, hide_index=True, on_select="rerun", selection_mode="single-row", key="queue_table",
    column_config={
        "Mức độ đáng ngờ": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%d"),
        "Vì sao": st.column_config.TextColumn(width="large"),
    },
)
rows = event.selection.rows
if rows:
    st.session_state["queue_pick"] = int(table["tx_id"].iloc[rows[0]])
    st.page_link("app_pages/vi_sao.py", label="Xem vì sao giao dịch này bị nghi ngờ",
                 icon=":material/arrow_forward:")
