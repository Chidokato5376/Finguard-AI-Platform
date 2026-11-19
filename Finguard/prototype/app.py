"""FinGuard AML — prototype vòng 2 (AI-Quantum Challenge 2026).

Chấm điểm rủi ro rửa tiền cho giao dịch liên ngân hàng trên AMLWorld HI-Medium (IBM, 32 triệu
giao dịch): GBDT 28 đặc trưng kịch bản + DRRE (bộ nhớ hành vi + đồ thị giao dịch chỉ dùng quá
khứ), trộn thứ hạng; QAPE phân bổ ca trực. Mọi con số trên web lấy từ gói prototype/bundle/,
dựng bằng scripts/build_prototype_bundle.py từ chính các lần huấn luyện/đánh giá của đội.

Chạy (từ thư mục Finguard/):
    streamlit run prototype/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

st.set_page_config(page_title="FinGuard AML", page_icon=":material/shield:", layout="wide")

import core  # noqa: E402

if not core.SAMPLE_PATH.exists():
    st.error("Thiếu gói dữ liệu prototype/bundle/. Dựng lại bằng: "
             "`python -X utf8 scripts/build_prototype_bundle.py`")
    st.stop()

with st.sidebar:
    st.markdown("### :material/shield: FinGuard AML")
    st.caption("Đội CMET · AQ2026-111 · Vòng bán kết AI-Quantum Challenge 2026")
    st.markdown(
        "**Dữ liệu:** AMLWorld HI-Medium (IBM, dữ liệu mô phỏng có nhãn rửa tiền), "
        f"khung {core.results()['window'][0][11:16]}–{core.results()['window'][1][11:16]} "
        "ngày 13/09/2022 thuộc tập **val** (tập test chưa chạm).")
    st.markdown(
        "**Hệ thống:** GBDT 28 đặc trưng kịch bản (95%) + DRRE R7 (5%), trộn theo thứ hạng.")
    st.caption("FinGuard chỉ đề xuất. Quyết định chặn, báo cáo giao dịch đáng ngờ luôn do "
               "chuyên viên tuân thủ đưa ra.")

pages = [
    st.Page("views/tong_quan.py", title="Tổng quan", icon=":material/home:", default=True),
    st.Page("views/canh_bao.py", title="Hàng đợi cảnh báo", icon=":material/notifications_active:"),
    st.Page("views/vi_sao.py", title="Vì sao đáng ngờ", icon=":material/manage_search:"),
    st.Page("views/ca_truc.py", title="Phân bổ ca trực (QAPE)", icon=":material/checklist:"),
    st.Page("views/chay_thu.py", title="Chạy thử mô hình", icon=":material/play_circle:"),
    st.Page("views/ket_qua.py", title="Kết quả đánh giá", icon=":material/leaderboard:"),
    st.Page("views/tai_xuong.py", title="Dữ liệu & checkpoint", icon=":material/download:"),
]
st.navigation(pages, position="top").run()
