import streamlit as st

import core

res = core.results()
med = res["leaderboard_hi_medium"]["models"]
small = res["leaderboard_hi_small"]["models"]
p2 = res["p2_hi_medium"]
th = res["thresholds"]
df = core.sample()

st.title("FinGuard AML — phát hiện rửa tiền qua giao dịch liên ngân hàng")
st.markdown(
    "Mỗi ngày một ngân hàng xử lý hàng triệu lệnh chuyển tiền; chỉ khoảng **1/1.000** là rửa tiền, "
    "và chúng được chia nhỏ qua nhiều tài khoản trung gian, nhiều ngân hàng, nhiều loại tiền. "
    "Quy tắc ngưỡng cố định sinh rất nhiều cảnh báo sai, còn chuyên viên chỉ kiểm tra được vài trăm "
    "cảnh báo mỗi ca. FinGuard xếp hạng mọi giao dịch theo mức đáng ngờ, giải thích lý do, "
    "và đề xuất ca trực nên kiểm tra gì trước."
)

c1, c2, c3, c4 = st.columns(4)
base_b = p2["half_b"]["positives"] / p2["half_b"]["rows"]
c1.metric("Giao dịch đã dùng (HI-Small + HI-Medium)", "≈ 37 triệu", border=True)
c2.metric("AUPRC hệ thống (nửa sau val HI-Medium)", f"{core.dec(p2['auprc_b']['tron_thu_hang'], 3)}",
          f"gấp {p2['auprc_b']['tron_thu_hang'] / base_b:,.0f} lần chọn ngẫu nhiên".replace(",", "."),
          delta_color="off", border=True)
c3.metric("Độ chính xác của 500 cảnh báo đầu (GBDT 28)",
          core.fmt_pct(med["gbdt_28"]["caught@500"] / 500), border=True)
c4.metric("Mức 'Chặn ngay': chính xác / bắt được",
          f"{core.fmt_pct(th['critical_precision'], 0)} / {core.fmt_pct(th['critical_recall'], 0)}", border=True)

st.subheader("Luồng xử lý")
st.graphviz_chart("""
digraph G {
  rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#eef4f7", color="#0e6375", fontname="Arial"];
  tx [label="Giao dịch mới\\n(người gửi, người nhận,\\nsố tiền, kênh, tiền tệ)"];
  feat [label="18 đặc trưng hành vi\\n+ 3 mô-đun kịch bản\\n(chuỗi trung gian,\\nliên ngân hàng, vòng tròn)"];
  gbdt [label="GBDT 28 cột\\n(xương sống)"];
  drre [label="DRRE R7\\nbộ nhớ hành vi +\\nđồ thị 24h quá khứ"];
  blend [label="Trộn thứ hạng\\n95% GBDT + 5% DRRE"];
  lv [label="3 mức cảnh báo\\n+ giải thích"];
  qape [label="QAPE: chọn cảnh báo\\ncho ca trực\\n(Greedy / ILP / QUBO)"];
  tx -> feat -> gbdt -> blend; feat -> drre -> blend; blend -> lv -> qape;
}
""")

st.subheader("Điều đã chứng minh được (trên tập val, 1 seed)")
a, b = st.columns(2, border=True)
with a:
    st.markdown("**Đặc trưng kịch bản của đội có giá trị độc lập**")
    st.markdown(
        f"GBDT dùng 28 cột (thêm 3 mô-đun kịch bản) đạt AUPRC **{core.dec(med['gbdt_28']['auprc'], 3)}** so với "
        f"**{core.dec(med['gbdt_20']['auprc'], 3)}** của 20 cột gốc trên HI-Medium "
        f"(+{core.dec(med['gbdt_28']['auprc'] - med['gbdt_20']['auprc'], 3)}, KTC 95% không chứa 0). "
        f"F1 tốt nhất {core.dec(med['gbdt_28']['f1_best'], 3)} so với {core.dec(med['gbdt_20']['f1_best'], 3)}.")
    st.markdown("**DRRE bổ sung thông tin mà GBDT không có**")
    ci = p2["vs_base_ci95"]["tron_thu_hang"]
    st.markdown(
        f"Trộn 5% DRRE vào GBDT nâng AUPRC nửa sau val từ **{core.dec(p2['auprc_b']['gbdt_28'], 4)}** lên "
        f"**{core.dec(p2['auprc_b']['tron_thu_hang'], 4)}** (+{core.dec(p2['auprc_b']['tron_thu_hang'] - p2['auprc_b']['gbdt_28'], 4)}, "
        f"KTC 95% [{core.dec(ci['lo'], 4, sign=True)}; {core.dec(ci['hi'], 4, sign=True)}]). Bộ trộn học trên nửa đầu val, đo trên nửa sau.")
with b:
    st.markdown("**DRRE học tốt hơn khi có nhiều dữ liệu**")
    st.markdown(
        f"Từ HI-Small lên HI-Medium (gấp ~6 lần dữ liệu), DRRE tăng "
        f"{core.dec(small['R5']['auprc'], 3)} → **{core.dec(med['R7']['auprc'], 3)}**, còn GBDT 28 chỉ tăng "
        f"{core.dec(small['gbdt_28']['auprc'], 3)} → {core.dec(med['gbdt_28']['auprc'], 3)}. DRRE vượt Random Forest "
        f"cùng 28 cột ({core.dec(med['rf_28']['auprc'], 3)}) và đứng **hạng 1 ở kiểu FAN-OUT** trên cả hai bộ.")
    st.markdown("**Không có rò rỉ thông tin tương lai**")
    st.markdown(
        "Đồ thị của DRRE chỉ thấy cạnh trong 24 giờ TRƯỚC mỗi khối giao dịch; chia train/val/test theo "
        "thời gian; tập test chưa chạm. Đội đã tự phát hiện và sửa hai lỗi đo (đồ thị nhìn trước, "
        "trộn mô hình trên chính tập huấn luyện) — xem trang *Kết quả đánh giá*.")

st.info(
    f"Web đang hiển thị **{core.fmt_num(len(df))} giao dịch** thật của AMLWorld (3 giờ liên tục trong tập val), "
    f"trong đó **{int(df['label'].sum())} giao dịch rửa tiền** đã biết nhãn. Điểm trên các trang được "
    "chấm với đủ lịch sử trước đó; trang *Chạy thử mô hình* chấm lại trực tiếp ngay trên web.",
    icon=":material/info:")
