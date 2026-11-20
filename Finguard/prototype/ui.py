"""Giao diện chung FINGUARD AI: chủ đề Dark Navy / Cyan, định dạng biểu đồ, thành phần lặp lại."""

from __future__ import annotations

import streamlit as st

CYAN = "#22D3EE"
NAVY = "#07111F"
PANEL = "#0E1B2D"
MUTED = "#91A4BA"
RED = "#FB7185"
AMBER = "#FBBF24"
GREEN = "#34D399"
LEVEL_COLORS = {"critical": RED, "medium": AMBER, "low": GREEN}
SEQ = [CYAN, "#818CF8", AMBER, RED, GREEN, "#F472B6", "#A3E635"]

CSS = """
<style>
:root { --fg-cyan:#22D3EE; --fg-muted:#91A4BA; --fg-panel:#0E1B2D; }
.stApp { background: radial-gradient(ellipse at 75% -15%, #102944 0%, #07111F 50%); }
[data-testid="stSidebar"] { background:#091626; border-right:1px solid #1B3048; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color:#C5D3E1; }
.block-container { padding-top:1.4rem; max-width:1500px; }
h1,h2,h3 { letter-spacing:-.02em; }
[data-testid="stMetric"] { background:linear-gradient(145deg,#102238,#0B1828); border:1px solid #1B354F;
  padding:16px 18px; border-radius:14px; }
[data-testid="stMetricLabel"] { color:#9DB2C8; }
[data-testid="stMetricValue"] { color:#F1F7FC; }
[data-testid="stAlert"] { border-radius:12px; }
[data-testid="stDataFrame"] { border:1px solid #1B354F; border-radius:12px; }
.fg-brand { font-size:1.35rem; font-weight:800; letter-spacing:.14em; color:#F4FAFF; }
.fg-tagline { color:#91A4BA; font-size:.78rem; margin-top:-.2rem; }
.fg-kicker { color:#22D3EE; text-transform:uppercase; letter-spacing:.16em; font-size:.72rem; font-weight:700; }
.fg-card { background:linear-gradient(145deg,#102238,#0B1828); border:1px solid #1B354F; border-radius:14px;
  padding:16px 18px; height:100%; }
.fg-card h4 { margin:0 0 .35rem 0; font-size:.95rem; color:#E6F0FA; }
.fg-card p { margin:0; color:#9DB2C8; font-size:.86rem; line-height:1.45; }
.fg-pill { display:inline-block; padding:2px 10px; border-radius:999px; font-size:.75rem; font-weight:700;
  border:1px solid #1B354F; margin-right:6px; }
.fg-src { color:#6F86A0; font-size:.75rem; }
</style>
"""


def setup() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def header(title: str, subtitle: str, kicker: str = "Financial Risk Intelligence Platform") -> None:
    st.markdown(f'<div class="fg-kicker">{kicker}</div>', unsafe_allow_html=True)
    st.title(title)
    st.caption(subtitle)


def card(title: str, body: str) -> None:
    st.markdown(f'<div class="fg-card"><h4>{title}</h4><p>{body}</p></div>', unsafe_allow_html=True)


def source_note(text: str) -> None:
    st.markdown(f'<div class="fg-src">Nguồn số liệu: {text}</div>', unsafe_allow_html=True)


def style(fig, height: int = 340, legend: bool = True):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      height=height, margin=dict(l=8, r=8, t=28, b=8), showlegend=legend,
                      font=dict(color="#C5D3E1"), legend=dict(orientation="h", y=-0.18, title_text=""))
    fig.update_xaxes(gridcolor="#16283D", zerolinecolor="#16283D")
    fig.update_yaxes(gridcolor="#16283D", zerolinecolor="#16283D")
    return fig
