"""FinGuard AI — application shell for the Streamlit demo."""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
st.set_page_config(
    page_title="FINGUARD AI | Risk Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.dashboard.data_service import list_available_sources  # noqa: E402
from src.dashboard.story import DATA_NOTES, SOURCE_LABELS, load_scored, method_label  # noqa: E402

st.markdown("""
<style>
:root { --fg-cyan:#22D3EE; --fg-muted:#91A4BA; --fg-panel:#0E1B2D; }
.stApp { background: radial-gradient(ellipse at 75% -15%, #102944 0%, #07111F 50%); }
[data-testid="stSidebar"] { background:#091626; border-right:1px solid #1B3048; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color:#C5D3E1; }
.block-container { padding-top:1.35rem; max-width:1500px; }
h1,h2,h3 { letter-spacing:-.025em; }
[data-testid="stMetric"] { background:linear-gradient(145deg,#102238,#0B1828); border:1px solid #1B354F; padding:18px 20px; border-radius:14px; }
[data-testid="stMetricLabel"] { color:#9DB2C8; }
[data-testid="stMetricValue"] { color:#F1F7FC; }
[data-testid="stAlert"] { border-radius:12px; }
.fg-brand { font-size:1.4rem; font-weight:800; letter-spacing:.13em; color:#F4FAFF; }
.fg-tagline { color:#91A4BA; font-size:.78rem; margin-top:-.35rem; }
.fg-kicker { color:#22D3EE; text-transform:uppercase; letter-spacing:.16em; font-size:.72rem; font-weight:700; }
.fg-card { background:linear-gradient(145deg,#102238,#0B1828); border:1px solid #1B354F; border-radius:14px; padding:18px; }
.fg-muted { color:#91A4BA; }
[data-testid="stDataFrame"] { border:1px solid #1B354F; border-radius:12px; }
button[kind="primary"] { background:#0891B2; border-color:#22D3EE; }
</style>
""", unsafe_allow_html=True)

sources = list_available_sources()
if not sources:
    st.error("Chưa có tập dữ liệu đã tiền xử lý trong `data/processed/`.")
    st.stop()
if "source" not in st.session_state or st.session_state["source"] not in sources:
    st.session_state["source"] = "synthetic_vn" if "synthetic_vn" in sources else sources[0]

with st.sidebar:
    st.markdown('<div class="fg-brand">FINGUARD AI</div><div class="fg-tagline">Intelligent Transaction Risk Management</div>', unsafe_allow_html=True)
    st.divider()
    st.markdown("**Workspace**")
    st.selectbox("Nguồn dữ liệu", sources, format_func=lambda s: SOURCE_LABELS.get(s, s), key="source", disabled=len(sources) == 1)
    st.slider("Số giao dịch gần nhất", 100, 800, 800, step=50, key="n_recent")
    scored = load_scored(st.session_state["source"])
    method = method_label(scored)
    st.caption(f"Backend hiện tại: **{method}**")
    st.caption(DATA_NOTES.get(st.session_state["source"], ""))
    st.divider()
    st.caption("● DEMO ENVIRONMENT")
    st.caption("Kết quả chỉ hỗ trợ rà soát. Không tự động chặn giao dịch.")

st.markdown('<div class="fg-kicker">Financial Risk Intelligence Platform</div>', unsafe_allow_html=True)
pages = [
    st.Page("app_pages/overview.py", title="Overview Dashboard", icon="📊", default=True),
    st.Page("app_pages/transaction_analysis.py", title="Transaction Analysis", icon="🔎"),
    st.Page("app_pages/giam_sat.py", title="Alert Center", icon="🚨"),
    st.Page("app_pages/vi_sao.py", title="Explainable AI", icon="🧠"),
    st.Page("app_pages/uu_tien.py", title="Fraud Network & Prioritization", icon="🕸️"),
    st.Page("app_pages/model_insights.py", title="Model Insights", icon="📈"),
    st.Page("app_pages/quantum_status.py", title="Quantum Optimization", icon="⚛️"),
    st.Page("app_pages/cach_hoat_dong.py", title="About & Methodology", icon="ℹ️"),
]
st.navigation(pages, position="sidebar").run()

