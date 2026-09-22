"""FinGuard AI — dashboard "kể chuyện" dành cho người không chuyên.

Mỗi trang trả lời MỘT câu hỏi, bằng ngôn ngữ đời thường:
  1. Câu chuyện                  — vì sao cần hệ thống này
  2. Giao dịch nào đang đáng lo  — hàng đợi cảnh báo (chấm điểm rủi ro)
  3. Vì sao đáng ngờ             — giải thích + đường đi của tiền (mạng lưới)
  4. Ca trực xử lý gì trước      — phân bổ nguồn lực (QAPE) so với làm theo thứ tự đến
  5. Hệ thống hoạt động thế nào  — giải thích cơ chế, giới hạn, chi tiết kỹ thuật

Kiến trúc: các trang chỉ lo hiển thị. Logic nằm trong data_service,
scoring_service, qape_service; story.py diễn đạt kết quả thành câu chữ.
Cột `method` (cách chấm điểm đang dùng) LUÔN được công bố ở thanh bên.

Chạy: streamlit run src/dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Cho phép chạy `streamlit run src/dashboard/app.py` từ thư mục gốc dự án
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

st.set_page_config(page_title="FinGuard AI", page_icon=":material/shield:", layout="wide")

from src.dashboard.data_service import list_available_sources  # noqa: E402
from src.dashboard.story import DATA_NOTES, SOURCE_LABELS, load_scored, method_label  # noqa: E402

sources = list_available_sources()
if not sources:
    st.warning(
        "Chưa có dữ liệu để hiển thị. Chạy bước tiền xử lý trước:\n\n```bash\n"
        "python -m src.data.preprocessing --source synthetic_vn --raw-path data/raw/synthetic_vn.csv "
        "--output data/processed\n```"
    )
    st.stop()

if "source" not in st.session_state:
    st.session_state["source"] = "synthetic_vn" if "synthetic_vn" in sources else sources[0]

with st.sidebar:
    st.markdown("**:material/tune: Tùy chọn hiển thị**")
    st.selectbox("Bộ dữ liệu", sources, format_func=lambda s: SOURCE_LABELS.get(s, s),
                 key="source", disabled=len(sources) == 1)
    st.slider("Xem bao nhiêu giao dịch gần nhất", 100, 800, 800, step=50, key="n_recent")
    scored = load_scored(st.session_state["source"])
    st.caption(f"**Cách chấm điểm đang dùng:** {method_label(scored)}.")
    st.caption(DATA_NOTES.get(st.session_state["source"], ""))
    st.caption("FinGuard chỉ **đề xuất**. Mọi quyết định chặn hay cho qua giao dịch "
               "luôn do nhân viên ngân hàng đưa ra.")

pages = [
    st.Page("app_pages/cau_chuyen.py", title="Câu chuyện", icon=":material/auto_stories:", default=True),
    st.Page("app_pages/giam_sat.py", title="Giao dịch đáng lo", icon=":material/notifications_active:"),
    st.Page("app_pages/vi_sao.py", title="Vì sao đáng ngờ", icon=":material/manage_search:"),
    st.Page("app_pages/uu_tien.py", title="Xử lý gì trước", icon=":material/checklist:"),
    st.Page("app_pages/cach_hoat_dong.py", title="Hệ thống hoạt động thế nào", icon=":material/lightbulb:"),
]
st.navigation(pages, position="top").run()
