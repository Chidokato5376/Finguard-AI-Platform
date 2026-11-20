"""FINGUARD AI — Intelligent Transaction Risk Management (prototype vòng 2, AI-Quantum Challenge 2026).

Ứng dụng cho người dùng cuối, chạy trên các mô hình ĐÃ HUẤN LUYỆN của dự án (không train lại):
  * GBDT 28 đặc trưng kịch bản (prototype/bundle/models/gbdt_28_hi_medium.joblib)
  * DRRE R7 — bộ nhớ hành vi + đồ thị giao dịch chỉ dùng quá khứ (drre_r7_hi_medium.pt)
  * Điểm hệ thống = trộn thứ hạng 95% GBDT + 5% DRRE; giải thích bằng SHAP TreeExplainer
  * QAPE — phân bổ ca trực bằng Greedy / ILP / QUBO (ủ mô phỏng, QAOA)
Dữ liệu trình diễn: AMLWorld HI-Medium (IBM), tập val; tập test chưa chạm.

Chạy (từ thư mục gốc repo):  streamlit run Finguard/prototype/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

st.set_page_config(page_title="FINGUARD AI | Risk Intelligence", page_icon="🛡️", layout="wide",
                   initial_sidebar_state="expanded")

import core  # noqa: E402
import ui  # noqa: E402

ui.setup()
if not core.SAMPLE_PATH.exists():
    st.error("Thiếu gói dữ liệu/mô hình prototype/bundle/. Dựng lại: "
             "`python -X utf8 Finguard/scripts/build_prototype_bundle.py`")
    st.stop()

with st.sidebar:
    st.markdown('<div class="fg-brand">FINGUARD AI</div>'
                '<div class="fg-tagline">Intelligent Transaction Risk Management</div>', unsafe_allow_html=True)
    st.divider()
    w = core.results()["window"]
    st.markdown("**Workspace**")
    st.caption(f"Dữ liệu: AMLWorld HI-Medium (IBM) · tập **validation** · {w[0][11:16]}–{w[1][11:16]} "
               "13/09/2022 · 233.762 giao dịch thật đã chấm điểm.")
    st.caption("Mô hình: GBDT 28 đặc trưng (95%) + DRRE R7 (5%) · giải thích SHAP · QAPE.")
    st.divider()
    st.caption("● DEMO ENVIRONMENT — dữ liệu mô phỏng công khai của IBM, không chứa khách hàng thật.")
    st.caption("Kết quả chỉ hỗ trợ rà soát. Quyết định chặn / báo cáo giao dịch đáng ngờ do chuyên viên đưa ra.")

pages = [
    st.Page("views/overview.py", title="Overview Dashboard", icon="📊", default=True),
    st.Page("views/transaction_analysis.py", title="Transaction Analysis", icon="🔎"),
    st.Page("views/network.py", title="Fraud Network Intelligence", icon="🕸️"),
    st.Page("views/explainable_ai.py", title="Explainable AI", icon="🧠"),
    st.Page("views/alert_center.py", title="Alert Center", icon="🚨"),
    st.Page("views/model_insights.py", title="Model Insights", icon="📈"),
    st.Page("views/quantum.py", title="Quantum Optimization", icon="⚛️"),
    st.Page("views/resources.py", title="Data & Checkpoints", icon="📦"),
]
st.navigation(pages, position="sidebar").run()
