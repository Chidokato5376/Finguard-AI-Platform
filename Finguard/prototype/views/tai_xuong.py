import pandas as pd
import streamlit as st

import core

res = core.results()
small, med = res["leaderboard_hi_small"]["models"], res["leaderboard_hi_medium"]["models"]

st.title("Dữ liệu và checkpoint để tự kiểm tra")
st.markdown(
    "Toàn bộ mô hình đã huấn luyện trong đợt 2 được đóng gói kèm web. Ban Giám khảo có thể tải về, "
    "chấm lại trên máy mình và so với số trên trang *Kết quả đánh giá*.")

desc = {
    "R2": ("HI-Small", "DRRE v1 + 28 đặc trưng"), "R3": ("HI-Small", "+ tầng chấm điểm v2"),
    "R4": ("HI-Small", "+ đồ thị chỉ dùng quá khứ"), "R5": ("HI-Small", "+ đồ thị hai chiều có đặc trưng cạnh"),
    "R6": ("HI-Small", "R6b + học lại đầu chấm điểm"), "R6a": ("HI-Small", "R5 + học lại đầu chấm điểm"),
    "R6b": ("HI-Small", "R5 + chống sụp embedding"), "R7": ("HI-Medium", "Cấu hình R5, 4 epoch — **mô hình nộp**"),
}
rows = []
for name, file in res["checkpoints"].items():
    path = core.BUNDLE / "models" / file
    board = med if name == "R7" else small
    rows.append({"Bậc": name, "Dữ liệu huấn luyện": desc[name][0], "Cấu hình": desc[name][1].replace("**", ""),
                 "Val AUPRC": board.get(name, {}).get("auprc"), "Tệp": file,
                 "Dung lượng (KB)": round(path.stat().st_size / 1024)})
st.dataframe(pd.DataFrame(rows).style.format({"Val AUPRC": "{:.4f}"}), hide_index=True)

st.subheader("Tải về")
c1, c2 = st.columns(2, border=True)
with c1:
    st.markdown("**Checkpoint DRRE**")
    pick = st.selectbox("Chọn bậc", list(res["checkpoints"]), index=len(res["checkpoints"]) - 1)
    f = core.BUNDLE / "models" / res["checkpoints"][pick]
    st.download_button(f"Tải {f.name}", f.read_bytes(), f.name, "application/octet-stream",
                       icon=":material/download:")
    g = core.GBDT_PATH
    st.markdown("**GBDT 28 cột (HistGradientBoosting, scikit-learn 1.4.2)**")
    st.download_button(f"Tải {g.name}", g.read_bytes(), g.name, "application/octet-stream",
                       icon=":material/download:")
with c2:
    st.markdown("**Dữ liệu mẫu đã xử lý**")
    s = core.SAMPLE_PATH
    st.caption(f"{core.fmt_num(len(core.sample()))} giao dịch val HI-Medium, 28 đặc trưng, nhãn, kiểu hình, "
               f"điểm của từng mô hình ({s.stat().st_size / 1e6:.1f} MB, Parquet).".replace(".", ",", 1))
    st.download_button(f"Tải {s.name}", s.read_bytes(), s.name, "application/octet-stream",
                       icon=":material/download:")
    r = core.BUNDLE / "results.json"
    st.markdown("**Kết quả đo (bảng xếp hạng, P2, đường học, ngưỡng)**")
    st.download_button("Tải results.json", r.read_bytes(), "results.json", "application/json",
                       icon=":material/download:")

st.subheader("Tái lập toàn bộ từ dữ liệu gốc")
st.markdown(
    "Dữ liệu gốc: **IBM Transactions for Anti Money Laundering (AMLWorld)** trên Kaggle "
    "(`ealtman2019/ibm-transactions-for-anti-money-laundering-aml`), tệp `HI-Small_Trans.csv` và "
    "`HI-Medium_Trans.csv`. Các lệnh dưới đây chạy từ thư mục `Finguard/` (chi tiết trong tài liệu hướng dẫn):")
st.code("""# 1. Tiền xử lý (chia 70/15/15 theo thời gian) và dựng 3 mô-đun đặc trưng kịch bản
bash scripts/run_aml_medium.sh HI      # = python -m src.data.preprocessing --source amlworld --raw-path ../../HI-Medium_Trans.csv --patterns-path ../../HI-Medium_Patterns.txt --min-daily-volume 100000 --output data/processed_hi_medium
python -X utf8 scripts/build_scenarios.py --processed-dir data/processed_hi_medium --raw-csv ../../HI-Medium_Trans.csv --scenarios mule_chain,cross_bank_fx,cycle

# 2. Huấn luyện DRRE R7 (~5 giờ/epoch trên CPU 8 nhân), chạy tiếp được sau khi ngắt
bash scripts/run_r7.sh

# 3. Bảng xếp hạng + P2 không rò rỉ
python -X utf8 scripts/leaderboard.py --processed-dir data/processed_hi_medium --drre R7=models/drre_r7_hi_medium.pt --out-dir reports/r7/leaderboard
python -X utf8 scripts/p2_stacking.py --processed-dir data/processed_hi_medium --scores-dir reports/r7/leaderboard/processed_hi_medium_val/scores --base gbdt_28 --drre R7

# 4. Dựng lại gói của web này
python -X utf8 scripts/build_prototype_bundle.py""", language="bash")
