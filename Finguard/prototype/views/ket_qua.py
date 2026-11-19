import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import core

res = core.results()
boards = {"HI-Medium (R7, 4,6 triệu giao dịch val)": res["leaderboard_hi_medium"],
          "HI-Small (R2–R6, 733 nghìn giao dịch val)": res["leaderboard_hi_small"]}

st.title("Kết quả đánh giá")
st.markdown(
    "Mọi số dưới đây đo trên **tập val** (tập test giữ nguyên, chỉ chạy một lần khi chốt), **1 seed (42)**, "
    "cùng 28 đặc trưng cho mọi mô hình. Khoảng tin cậy 95% dùng bootstrap Poisson ghép cặp. Tệp gốc: "
    "`reports/r7/leaderboard/`, `reports/leaderboard/`, `reports/gate1/`.")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Bảng xếp hạng", "Thang cải tiến DRRE", "DRRE bổ sung cho GBDT (P2)",
                                        "Theo kiểu rửa tiền (P4)", "Thang đo & giới hạn"])

with tab1:
    which = st.radio("Bộ dữ liệu", list(boards), horizontal=True)
    b = boards[which]
    names = sorted(b["models"], key=lambda n: b["ranks"]["auprc"][n])
    rows = []
    for n in names:
        m = b["models"][n]
        ci = b["vs_leader"].get(n)
        rows.append({
            "Mô hình": n, "Số cột": b["n_features"].get(n, "—"), "AUPRC": m["auprc"], "ROC-AUC": m["roc_auc"],
            "Bắt@500": m["caught@500"], "Bắt@2000": m["caught@2000"], "F1 tốt nhất": m["f1_best"],
            "Hạng TB": b["mean_rank"][n],
            "Δ AUPRC vs hạng 1 [KTC 95%]": "—" if ci is None else
            f"{m['auprc'] - b['models'][b['leader']]['auprc']:+.3f} [{ci['lo']:+.3f}; {ci['hi']:+.3f}]",
        })
    st.dataframe(pd.DataFrame(rows).style.format({"AUPRC": "{:.4f}", "ROC-AUC": "{:.4f}", "F1 tốt nhất": "{:.3f}",
                                                  "Hạng TB": "{:.2f}"}), hide_index=True)
    st.caption(f"{core.fmt_num(b['n_rows'])} giao dịch, {core.fmt_num(b['n_pos'])} vụ rửa tiền "
               f"(tỷ lệ nền {core.dec(100 * b['base_rate'], 3)}%). "
               "gbdt/rf/logistic: mô hình cổ điển cùng 28 cột; gbdt_20: chỉ 20 cột gốc; ngau_nhien: chấm ngẫu nhiên. "
               "Hạng TB = trung bình hạng trên AUPRC, ROC-AUC, Bắt@500/1000/2000, F1.")
    fig = px.bar(pd.DataFrame(rows), x="AUPRC", y="Mô hình", orientation="h", text_auto=".3f",
                 color=["DRRE" if n.startswith("R") else "Cổ điển" for n in names],
                 color_discrete_map={"DRRE": "#0e6375", "Cổ điển": "#9aa9b3"},
                 category_orders={"Mô hình": names})
    fig.update_layout(height=40 * len(rows) + 80, legend_title="",
                      margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown(
        "Mỗi bậc chỉ thêm **một** thay đổi so với bậc trước, để biết đóng góp của từng thay đổi "
        "(HI-Small, 4 epoch; R7 = cấu hình R5 trên HI-Medium).")
    ladder = pd.DataFrame([
        ("R2", "DRRE v1 + 28 đặc trưng mới"), ("R3", "+ tầng chấm điểm v2 (MLP, chuẩn hóa, δ thói quen)"),
        ("R4", "+ đồ thị CHỈ dùng quá khứ (bỏ nhìn trước tương lai)"),
        ("R5", "+ đồ thị hai chiều, đặc trưng cạnh, đọc theo cạnh"),
        ("R6b", "+ chống sụp embedding (VICReg)"), ("R7", "R5 trên HI-Medium (22,3 triệu giao dịch train)"),
    ], columns=["Bậc", "Thay đổi"])
    small, med = res["leaderboard_hi_small"]["models"], res["leaderboard_hi_medium"]["models"]
    ladder["AUPRC val (bảng xếp hạng)"] = [small[k]["auprc"] if k in small else med[k]["auprc"] for k in ladder["Bậc"]]
    st.dataframe(ladder.style.format({"AUPRC val (bảng xếp hạng)": "{:.4f}"}), hide_index=True)
    st.markdown(
        "**R3 → R4 giảm** là chủ đích: đồ thị cũ dựng cho cả cửa sổ nên giao dịch đầu cửa sổ 'thấy' giao dịch "
        "sau nó. Bỏ lỗi nhìn trước làm số thấp đi nhưng trung thực; R5 lấy lại phần lớn nhờ đồ thị đúng nghĩa.")
    curves = res["curves"]
    fig = go.Figure()
    for name, ys in curves.items():
        fig.add_trace(go.Scatter(x=list(range(1, len(ys) + 1)), y=ys, mode="lines+markers", name=name,
                                 line=dict(dash="dot" if name.startswith("v1") else "solid")))
    fig.update_layout(height=380, xaxis_title="Epoch", yaxis_title="Val AUPRC (in trong lúc huấn luyện)",
                      margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Checkpoint được chọn theo điểm làm mượt, không bao giờ lấy epoch đầu hay cuối (R7 dùng epoch 3). "
               "Đường nét chấm: DRRE v1 đợt 1 trên HI-Medium, còn lỗi nhìn trước — chỉ để đối chiếu.")

with tab3:
    st.markdown(
        "**Câu hỏi:** DRRE có mang thông tin mà GBDT trên cùng đặc trưng không có? **Cách đo không rò rỉ:** "
        "cả hai mô hình học trên train; tập val chia theo thời gian, nửa đầu để học bộ trộn, nửa sau để đo.")
    for label, key in (("HI-Medium · R7", "p2_hi_medium"), ("HI-Small · R5", "p2_hi_small")):
        p = res[key]
        base = p["auprc_b"][p["base"]]
        rows = []
        for n, v in p["auprc_b"].items():
            ci = p["vs_base_ci95"].get(n)
            rows.append({"Cách kết hợp": {"tron_logistic": "Trộn hồi quy logistic", "tron_thu_hang": "Trộn thứ hạng"}.get(n, n),
                         "AUPRC nửa sau": v, "Bắt@2000": p["caught2000_b"][n],
                         "Δ so với GBDT [KTC 95%]": "—" if ci is None else f"{v - base:+.4f} [{ci['lo']:+.4f}; {ci['hi']:+.4f}]"})
        st.markdown(f"**{label}** — trọng số DRRE chọn trên nửa đầu: **{p['rank_weight_drre']}**")
        st.dataframe(pd.DataFrame(rows).style.format({"AUPRC nửa sau": "{:.4f}"}), hide_index=True)
    st.success("Trên HI-Medium, trộn thứ hạng 5% DRRE tăng AUPRC có ý nghĩa thống kê; trên HI-Small trọng số tối ưu "
               "là 0. DRRE bắt đầu bổ sung thông tin khi có đủ dữ liệu.", icon=":material/insights:")
    st.warning("Cách đo cũ (học bộ trộn trên chính tập train của DRRE) cho số âm giả tạo trên HI-Small "
               "(−0,067) và dương giả tạo ở đợt 1 (+0,039). Đội đã tự phát hiện rò rỉ này và thay bằng cách đo trên.",
               icon=":material/report:")

with tab4:
    which = st.radio("Bộ dữ liệu ", list(boards), horizontal=True)
    b = boards[which]
    kinds = sorted(b["typology_n"], key=lambda k: -b["typology_n"][k])
    names = sorted(b["models"], key=lambda n: b["ranks"]["auprc"][n])
    names = [n for n in names if n not in ("ngau_nhien", "logistic_28")]
    z = [[b["models"][n]["typology_recall@2000"].get(k, 0) for k in kinds] for n in names]
    fig = px.imshow(z, x=[f"{k} (n={b['typology_n'][k]})" for k in kinds], y=names, text_auto=".0%",
                    color_continuous_scale="Teal", aspect="auto", zmin=0, zmax=1)
    fig.update_layout(height=60 * len(names) + 120, margin=dict(l=10, r=10, t=10, b=10), coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Tỷ lệ vụ của từng kiểu nằm trong 2.000 cảnh báo đầu. HI-Medium có 5.489 vụ > 2.000 cảnh báo nên "
               "tỷ lệ thấp hơn HI-Small. DRRE hạng 1 ở FAN-OUT trên cả hai bộ; còn yếu ở STACK và BIPARTITE.")

with tab5:
    med = res["leaderboard_hi_medium"]["models"]
    th = res["thresholds"]
    st.markdown("#### Thang đo nghiệp vụ (có lợi và có ý nghĩa cho ngân hàng)")
    st.markdown(f"""
- **Độ chính xác của danh sách đầu (Precision@K):** trong 500 cảnh báo điểm cao nhất của val HI-Medium,
  GBDT 28 đúng **{med['gbdt_28']['caught@500']}/500 = {core.fmt_pct(med['gbdt_28']['caught@500'] / 500, 0)}**; mức "Chặn ngay"
  của hệ thống đúng **{core.fmt_pct(th['critical_precision'], 0)}** và bắt **{core.fmt_pct(th['critical_recall'], 0)}** số vụ
  khi chỉ gắn cờ **{core.fmt_pct(th['critical_top_share'], 3)}** giao dịch.
- **Khối lượng công việc:** bắt **{core.fmt_pct(th['medium_recall'], 0)}** số vụ chỉ cần xem
  **{core.fmt_pct(th['medium_top_share'], 2)}** giao dịch (≈ {core.fmt_num(th['medium_top_share'] * 1e6)} trên mỗi triệu giao dịch).
- **Mức nâng so với chọn ngẫu nhiên (lift):** AUPRC 0,594 trên tỷ lệ nền 0,12% ≈ **500 lần**.
- **Giá trị của đặc trưng kịch bản:** +0,079 AUPRC (GBDT 28 so với 20 cột), KTC 95% [+0,067; +0,088].
- **Bổ sung của DRRE (P2):** +0,018 AUPRC [+0,010; +0,027] khi trộn 5%.
- **Mở rộng theo dữ liệu (P5):** DRRE +0,067 AUPRC từ HI-Small lên HI-Medium, gấp ~3 lần mức tăng của GBDT (+0,021).
""")
    st.markdown("#### Giới hạn cần nói rõ")
    st.markdown("""
- Một seed; số trên val. Ngưỡng 3 mức chọn trên chính val nên độ chính xác/recall của ngưỡng hơi lạc quan.
- DRRE đứng riêng vẫn **thua GBDT** (−0,173 AUPRC trên HI-Medium); vai trò hiện tại là tín hiệu bổ sung và chuyên kiểu FAN-OUT.
- ROC-AUC của mọi mô hình ≈ 0,98 nên không phân biệt được mô hình — dùng AUPRC và Precision@K.
- So với bài báo IBM (XGBoost + đặc trưng đồ thị: F1 lớp thiểu số 65,7% trên HI-Medium, chia 60/20/20), F1 tốt nhất
  của GBDT 28 là 61,9% trên val với cách chia 70/15/15 — chưa so trực tiếp được.
- Thời gian xử lý từng cảnh báo trong trang QAPE là giả định minh họa.
""")
