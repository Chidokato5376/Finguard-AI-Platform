import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import core
import inference as inf
import ui

ui.header("Transaction Analysis", "Nhập một giao dịch hoặc tải tệp nhiều giao dịch → kiểm tra định dạng → chạy "
          "mô hình đã huấn luyện (GBDT 28 + DRRE R7) → điểm rủi ro, mức cảnh báo, yếu tố SHAP.")

feats = core.features()
with st.expander("Lược đồ đầu vào của mô hình (lấy từ checkpoint) và tệp mẫu", icon="📐"):
    st.markdown(
        "Mô hình nhận **28 đặc trưng đã tiền xử lý** theo đúng pipeline huấn luyện (`src/data/preprocessing.py` + "
        "`scripts/build_scenarios.py`): 18 đặc trưng hành vi tính từ lịch sử tài khoản + 3 mô-đun kịch bản. "
        "Vì nhiều đặc trưng cần lịch sử giao dịch trước đó, ứng dụng nhận đầu vào ở dạng đặc trưng này "
        "(đúng như lúc train) chứ không tự suy ra từ một dòng giao dịch thô.")
    st.dataframe(inf.input_schema(), hide_index=True, use_container_width=True, height=300)
    tmpl = io.StringIO()
    core.sample().head(2000)[inf.ID_COLUMNS + feats + ["amount", "currency", "label"]].to_csv(tmpl, index=False)
    st.download_button("Tải tệp mẫu CSV (2.000 giao dịch val, đúng định dạng)", tmpl.getvalue(),
                       "finguard_input_template.csv", "text/csv", icon="⬇️")

tab_batch, tab_single = st.tabs(["📁 Phân tích hàng loạt (CSV)", "✍️ Phân tích một giao dịch"])

# --------------------------------------------------------------------------
with tab_batch:
    src = st.radio("Nguồn dữ liệu", ["Tải tệp CSV / Parquet của tôi", "Dùng dữ liệu AMLWorld trong ứng dụng"],
                   horizontal=True)
    batch = None
    if src.startswith("Tải"):
        up = st.file_uploader("Chọn tệp", type=["csv", "parquet"],
                              help="Tối đa 100.000 dòng mỗi lần. Cột theo lược đồ ở trên; có thể thêm `label` (0/1).")
        if up is not None:
            try:
                batch = pd.read_parquet(up) if up.name.endswith(".parquet") else pd.read_csv(up)
                batch = batch.head(100_000)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Không đọc được tệp: {exc}")
    else:
        base = core.sample().drop(columns=["level"])
        n = st.select_slider("Số giao dịch (theo thứ tự thời gian, từ 09:00)",
                             [2_000, 5_000, 20_000, 50_000, 100_000], value=5_000)
        batch = base.head(n)

    if batch is not None:
        checked, rep = inf.validate(batch)
        st.markdown("##### 1 · Kiểm tra định dạng")
        v = st.columns(4)
        v[0].metric("Số dòng", core.fmt_num(rep["rows"]))
        v[1].metric("Hợp lệ", core.fmt_num(rep.get("ok_rows", 0)))
        v[2].metric("Lỗi dữ liệu", core.fmt_num(rep.get("bad_rows", 0)))
        v[3].metric("Cột thiếu", len(rep["missing_columns"]))
        if rep["missing_columns"]:
            st.error("Thiếu cột bắt buộc: " + ", ".join(f"`{c}`" for c in rep["missing_columns"]))
        if rep.get("bad_rows"):
            st.warning("Các dòng lỗi sẽ không được đưa vào mô hình (không tự điền giá trị để tránh sai lệch kết quả):")
            st.dataframe(checked.loc[checked["_status"] != "Hợp lệ", inf.ID_COLUMNS + ["_status"]].head(50),
                         hide_index=True, use_container_width=True)
        st.markdown("##### 2 · Xem trước")
        st.dataframe(batch.head(10), hide_index=True, use_container_width=True)

        st.markdown("##### 3 · Chạy mô hình")
        o = st.columns([1, 1, 2])
        use_drre = o[0].toggle("DRRE R7 (đồ thị + bộ nhớ)", value=True,
                               help="Cần lịch sử theo thời gian; ~800 giao dịch/giây trên CPU 8 nhân.")
        explain = o[1].toggle("Giải thích SHAP", value=True)
        if not rep["missing_columns"] and st.button("Run analysis", type="primary", icon="▶️"):
            with st.spinner("Đang chấm điểm bằng mô hình đã huấn luyện..."):
                st.session_state["ta_result"] = inf.analyze(batch, use_drre=use_drre, explain=explain)

    if "ta_result" in st.session_state:
        scored, rep = st.session_state["ta_result"]
        table = inf.result_table(scored)
        st.markdown("##### 4 · Kết quả")
        done = scored[scored["Analysis Status"].astype(str).str.startswith("Hoàn tất")]
        r = st.columns(5)
        r[0].metric("Đã phân tích", core.fmt_num(len(done)))
        lv = done["live_level"].value_counts() if "live_level" in done else pd.Series(dtype=int)
        r[1].metric("🔴 Chặn ngay", int(lv.get("critical", 0)))
        r[2].metric("🟡 Cần xác minh", int(lv.get("medium", 0)))
        r[3].metric("Thời gian mô hình", f"{sum(rep['timing'].values()):.1f} s")
        r[4].metric("Tốc độ GBDT", f"{core.fmt_num(len(done) / max(rep['timing'].get('gbdt_28', 1e-9), 1e-9))} GD/s")
        view = st.segmented_control("Hiển thị", ["Chỉ cảnh báo", "Tất cả", "Dòng lỗi"], default="Chỉ cảnh báo")
        show = table
        if view == "Chỉ cảnh báo":
            show = table[table["Prediction"] == "Đáng ngờ"]
        elif view == "Dòng lỗi":
            show = table[~table["Analysis Status"].astype(str).str.startswith("Hoàn tất")]
        show = show.sort_values("Risk Score", ascending=False, na_position="last")
        st.dataframe(show.head(1000), hide_index=True, use_container_width=True, height=420, column_config={
            "Risk Score": st.column_config.ProgressColumn(min_value=99.0, max_value=100.0, format="%.3f"),
            "Fraud Probability": st.column_config.NumberColumn(format="%.4f")})
        buf = io.StringIO()
        table.to_csv(buf, index=False)
        st.download_button("Xuất báo cáo CSV", buf.getvalue(), "finguard_analysis_report.csv", "text/csv",
                           icon="📄", type="primary")
        if "label" in done and done["label"].astype(bool).any():
            st.markdown("##### Đối chiếu với nhãn thật (có cột `label`)")
            cols = {"Hệ thống (chạy trực tiếp)": "live_he_thong", "GBDT 28": "live_gbdt_28"}
            if "live_R7" in done:
                cols["DRRE R7 (bộ nhớ khởi động lạnh)"] = "live_R7"
            if "score_he_thong" in done:
                cols["Hệ thống (điểm đã chấm với đủ lịch sử)"] = "score_he_thong"
            st.dataframe(core.evaluate(done, cols).style.format(
                {"AUPRC": "{:.4f}", "ROC-AUC": "{:.4f}", "Độ chính xác @100": "{:.0%}", "Recall @ top 1%": "{:.0%}"}),
                hide_index=True, use_container_width=True)
            st.caption("DRRE chạy trực tiếp bắt đầu với bộ nhớ hành vi trống nên thấp hơn điểm đã chấm với đủ lịch sử.")

# --------------------------------------------------------------------------
with tab_single:
    st.markdown("Chọn một giao dịch có sẵn làm điểm xuất phát (các trường được điền đúng giá trị đã tiền xử lý), "
                "chỉnh giá trị nếu muốn, rồi chạy phân tích.")
    df = core.sample()
    top = df[df["level"] != "low"].sort_values("score_he_thong", ascending=False).head(200)
    options = list(top["tx_id"].astype(int)) + list(df[df["level"] == "low"].head(50)["tx_id"].astype(int))
    pick = st.selectbox("Giao dịch xuất phát", options, format_func=lambda i: (
        f"TX-{i:06d} · {df.iloc[i]['account_id']} → {df.iloc[i]['counterparty_id']} · "
        f"{df.iloc[i]['amount']:,.2f} {df.iloc[i]['currency']} · {core.LEVELS[df.iloc[i]['level']][1]}"))
    row = df.iloc[int(pick)]
    groups = {
        "Hành vi người gửi": [f for f in feats if f[:3] not in ("mc_", "cb_", "cy_")],
        "Chuỗi trung gian (mule)": [f for f in feats if f.startswith("mc_")],
        "Liên ngân hàng / ngoại tệ": [f for f in feats if f.startswith("cb_")],
        "Dòng tiền vòng tròn": [f for f in feats if f.startswith("cy_")],
    }
    with st.form("single"):
        ids = st.columns(3)
        acc = ids[0].text_input("account_id", str(row["account_id"]))
        cp = ids[1].text_input("counterparty_id", str(row["counterparty_id"]))
        ts = ids[2].text_input("timestamp", str(row["timestamp"]))
        values = {}
        for gname, fs in groups.items():
            st.markdown(f"**{gname}**")
            cols = st.columns(4)
            for i, f in enumerate(fs):
                values[f] = cols[i % 4].number_input(core.FEATURE_LABELS[f].split("] ")[-1], value=float(row[f]),
                                                     format="%.4f", key=f"in_{f}", help=f"Cột `{f}`")
        submitted = st.form_submit_button("Run analysis", type="primary", icon="▶️")
    if submitted:
        one = pd.DataFrame([{**values, "account_id": acc, "counterparty_id": cp, "timestamp": ts}])
        checked, rep = inf.validate(one)
        if checked["_status"].iloc[0] != "Hợp lệ":
            st.error(checked["_status"].iloc[0])
        else:
            changed = any(abs(values[f] - float(row[f])) > 1e-6 for f in feats)
            g = core.score_gbdt(checked)
            pct_g = core.to_percentile(g, core.reference()["gbdt_28"])
            if not changed:
                system = core.system_score(pct_g, np.array([row["pct_R7"]]))
                mode = "GBDT 28 + DRRE R7 (điểm DRRE chấm với đủ lịch sử của tài khoản)"
            else:
                system = pct_g
                mode = "Chỉ GBDT 28 — DRRE cần chuỗi lịch sử thật của tài khoản, không chấm được giá trị chỉnh tay"
            level = core.level_of(system)[0]
            prob = float(core.probability(system)[0])
            base, sv = inf.shap_values(checked)
            m = st.columns([1, 1.4, 1, 1, 1])
            m[0].metric("Prediction", "Đáng ngờ" if level != "low" else "Bình thường")
            m[1].metric("Risk Level", f"{core.LEVELS[level][0]} {core.LEVELS[level][1]}")
            m[2].metric("Risk Score", core.dec(100 * float(system[0]), 3))
            m[3].metric("Fraud Probability", core.fmt_pct(prob, 1))
            m[4].metric("Analysis Status", "Hoàn tất")
            st.caption(f"Chế độ chấm: {mode}. Đề xuất: {core.LEVELS[level][2]}")
            contrib = pd.DataFrame({"f": feats, "v": sv[0], "x": [values[f] for f in feats]})
            contrib = contrib.reindex(contrib["v"].abs().sort_values(ascending=False).index).head(10).iloc[::-1]
            fig = go.Figure(go.Bar(
                x=contrib["v"], orientation="h",
                y=[core.FEATURE_LABELS[f].split("] ")[-1] + f" = {inf.value_text(f, x)}" for f, x in zip(contrib["f"], contrib["x"])],
                marker_color=np.where(contrib["v"] > 0, ui.RED, ui.GREEN)))
            fig.update_layout(title="Key Risk Factors — SHAP (log-odds của GBDT)", xaxis_title="Đóng góp SHAP")
            st.plotly_chart(ui.style(fig, 380, legend=False), use_container_width=True)
            if bool(row["label"]) and not changed:
                st.caption(f"Kiểm chứng: nhãn AMLWorld của giao dịch này là **rửa tiền** "
                           f"({row['laundering_pattern'] or 'UNSPECIFIED'}).")
