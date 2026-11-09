"""Lớp "kể chuyện" của dashboard: diễn đạt kết quả của hệ thống bằng ngôn ngữ đời thường.

Module này KHÔNG chứa logic mô hình. Nó chỉ đọc kết quả từ data_service,
scoring_service và qape_service, rồi chuyển thành câu chữ, màu sắc và con số
mà một người không có kiến thức về rủi ro, học máy hay lượng tử vẫn hiểu được.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np
import pandas as pd
import streamlit as st

LOW_MAX, MEDIUM_MAX = 30, 70  # khớp config.yaml: risk_scoring.thresholds
FORCE_SELECT_MIN = 90  # QAPE: cảnh báo từ mức này luôn phải được xử lý
MINUTES_PER_COST_UNIT = 10  # giả định quy đổi: 1 đơn vị chi phí QAPE ≈ 10 phút làm việc

LEVELS = {
    "critical": {
        "emoji": "🔴", "name": "Chặn ngay", "color": "red", "hex": "#e74c3c",
        "action": "Tạm giữ giao dịch và gọi điện xác minh chính chủ. Nếu không phải "
                  "chính chủ: khóa tài khoản và báo cơ quan chức năng.",
    },
    "medium": {
        "emoji": "🟡", "name": "Cần xác minh", "color": "orange", "hex": "#f39c12",
        "action": "Cho giao dịch chờ, nhắn tin hoặc gửi mã OTP để khách hàng tự xác nhận.",
    },
    "low": {
        "emoji": "🟢", "name": "Bình thường", "color": "green", "hex": "#27ae60",
        "action": "Không cần làm gì, giao dịch được đi qua.",
    },
}

SOURCE_LABELS = {
    "synthetic_vn": "Mô phỏng bối cảnh Việt Nam",
    "paysim": "PaySim — mô phỏng ví điện tử",
    "ieee_cis": "IEEE-CIS — giao dịch thẻ thật đã ẩn danh",
}

DATA_NOTES = {
    "synthetic_vn": "Dữ liệu minh họa được mô phỏng theo bối cảnh Việt Nam, "
                    "không phải giao dịch của khách hàng thật nào.",
    "paysim": "Dữ liệu mô phỏng giao dịch ví điện tử (PaySim), không phải giao dịch thật.",
    "ieee_cis": "Giao dịch thẻ thật đã được ẩn danh (IEEE-CIS). Bộ dữ liệu này "
                "không có thông tin người nhận.",
}

COUNTRY_NAMES = {"VN": "Việt Nam", "KH": "Campuchia", "PH": "Philippines", "unknown": "một nơi không xác định"}

CHANNEL_NAMES = {
    "domestic": "chuyển khoản trong nước", "e_wallet": "ví điện tử",
    "international": "chuyển tiền quốc tế", "transfer": "chuyển khoản",
    "cash_out": "rút tiền", "payment": "thanh toán", "cash_in": "nạp tiền", "debit": "ghi nợ",
}

SCENARIO_NAMES = {
    "money_mule_chain": "chuyển tiền qua chuỗi tài khoản trung gian",
    "account_takeover": "kẻ gian chiếm quyền tài khoản",
    "investment_romance_scam": "lừa đảo đầu tư / tình cảm",
}


# --------------------------------------------------------------------------
# Định dạng cho người đọc
# --------------------------------------------------------------------------

def level_of(score: float) -> str:
    if score <= LOW_MAX:
        return "low"
    if score <= MEDIUM_MAX:
        return "medium"
    return "critical"


def level_label(level: str) -> str:
    info = LEVELS[level]
    return f"{info['emoji']} {info['name']}"


def mask_account(account) -> str:
    """Hiển thị tài khoản như sao kê ngân hàng (che bớt số): vừa giống thực tế,
    vừa không để lộ tên kịch bản mô phỏng nằm trong mã tài khoản."""
    if account is None or (isinstance(account, float) and math.isnan(account)):
        return "—"
    number = int(hashlib.md5(str(account).encode("utf-8")).hexdigest(), 16) % 1_000_000
    return f"TK ••• {number:06d}"


def fmt_num(x: float, digits: int = 1) -> str:
    return f"{x:,.{digits}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def fmt_money(x, source: str) -> str:
    if x is None or pd.isna(x):
        return "—"
    if source == "ieee_cis":
        return f"${x:,.2f}"
    text = f"{x:,.0f}".replace(",", ".")
    return f"{text} ₫" if source == "synthetic_vn" else text


def time_col_of(df: pd.DataFrame) -> str:
    return "timestamp" if "timestamp" in df.columns else "step"


def fmt_when(row) -> str:
    ts = row.get("timestamp") if hasattr(row, "get") else None
    if ts is not None and not pd.isna(ts):
        t = pd.Timestamp(ts)
        return f"{t:%H:%M} ngày {t:%d/%m/%Y}"
    return f"thời điểm #{int(row['step'])}"


def day_part(ts) -> str:
    h = pd.Timestamp(ts).hour
    if h < 5:
        return "rạng sáng"
    if h < 11:
        return "buổi sáng"
    if h < 13:
        return "buổi trưa"
    if h < 18:
        return "buổi chiều"
    if h < 22:
        return "buổi tối"
    return "đêm khuya"


def method_label(scored: pd.DataFrame) -> str:
    if (scored["method"] == "drre_trained").any():
        return "Mô hình AI đã huấn luyện (DRRE)"
    return "Quy tắc thống kê — so sánh mỗi giao dịch với thói quen của chính khách hàng đó"


# --------------------------------------------------------------------------
# Dữ liệu (đã chấm điểm) + các cột diễn giải
# --------------------------------------------------------------------------

@st.cache_data(show_spinner="Đang chấm điểm các giao dịch...", max_entries=6)
def load_scored(source: str, split: str = "test") -> pd.DataFrame:
    from src.dashboard.data_service import load_processed_split
    from src.dashboard.scoring_service import compute_heuristic_scores, score_batch

    df = load_processed_split(source, split)
    try:
        scored = score_batch(df)
    except Exception:  # noqa: BLE001 — checkpoint không hợp nguồn dữ liệu: quay về quy tắc thống kê
        scored = compute_heuristic_scores(df)
    return _add_story_columns(scored)


def _add_story_columns(df: pd.DataFrame) -> pd.DataFrame:
    tc = time_col_of(df)
    df = df.sort_values(tc, kind="stable").reset_index(drop=True).copy()
    df["tx_id"] = np.arange(len(df))

    # Thói quen chi tiêu: chỉ dùng các giao dịch TRƯỚC giao dịch đang xét
    log_amount = np.log1p(df["amount"].astype(float))
    n_prior = df.groupby("account_id", sort=False, observed=True).cumcount()
    mean_prior = (log_amount.groupby(df["account_id"], observed=True).cumsum() - log_amount) / n_prior.where(n_prior > 0)
    df["n_prior"] = n_prior
    df["typical_amount"] = np.expm1(mean_prior)
    df["amount_ratio"] = df["amount"] / df["typical_amount"]

    has_receiver = df["counterparty_id"].notna()
    df["new_receiver"] = has_receiver & ~df.duplicated(subset=["account_id", "counterparty_id"])
    df["level"] = df["risk_score"].map(level_of)
    df["reason"] = [_short_reason(r) for r in df.itertuples(index=False)]
    return df


def _amount_is_unusual(r) -> bool:
    z = getattr(r, "z_score", None)
    if z is not None and not pd.isna(z):
        return z >= 2
    return r.amount_ratio >= 3


def _short_reason(r) -> str:
    if r.n_prior == 0:
        return "Giao dịch đầu tiên — chưa có thói quen để so sánh"
    if _amount_is_unusual(r) and r.amount_ratio >= 1:
        return f"Số tiền cao gấp {fmt_num(r.amount_ratio)} lần mức thường ngày"
    if r.new_receiver and r.n_prior >= 3:
        return "Chuyển cho người nhận lần đầu"
    return "Giống thói quen thường ngày"


def get_view() -> tuple[str, pd.DataFrame, pd.DataFrame]:
    """(nguồn, toàn bộ dữ liệu đã chấm điểm, N giao dịch gần nhất đang xem)."""
    source = st.session_state["source"]
    scored = load_scored(source)
    view = scored.tail(st.session_state.get("n_recent", 800))
    return source, scored, view


def tx_label(row, source: str) -> str:
    return (f"{LEVELS[row['level']]['emoji']} {fmt_money(row['amount'], source)} · "
            f"{mask_account(row['account_id'])} → {mask_account(row['counterparty_id'])} · {fmt_when(row)}")


# --------------------------------------------------------------------------
# Vì sao đáng ngờ
# --------------------------------------------------------------------------

def explain(scored: pd.DataFrame, row, source: str) -> tuple[list[tuple[str, str, str]], list[str]]:
    """Trả về (lý do tính vào điểm, dấu hiệu nên kiểm tra thêm).

    Mỗi lý do là (icon, tiêu đề, giải thích). Dấu hiệu kiểm tra thêm KHÔNG
    được tính vào điểm — tách riêng để không nói quá về cách hệ thống chấm.
    """
    reasons: list[tuple[str, str, str]] = []
    n = int(row["n_prior"])
    amount = fmt_money(row["amount"], source)
    if n == 0:
        reasons.append((
            ":material/person_add:", "Khách hàng chưa có lịch sử giao dịch",
            "Đây là giao dịch đầu tiên hệ thống thấy của khách hàng này, nên chưa thể so "
            "sánh với thói quen. Hệ thống thận trọng và không kết luận vội.",
        ))
    else:
        typical = fmt_money(row["typical_amount"], source)
        ratio = row["amount_ratio"]
        text = (f"Thường ngày khách hàng này chuyển khoảng **{typical}** (tính từ {n} "
                f"giao dịch trước). Lần này là **{amount}**")
        if ratio >= 1.5:
            text += f" — gấp **{fmt_num(ratio)} lần**."
        elif ratio <= 0.67:
            text += " — nhỏ hơn thường ngày."
        else:
            text += " — tương đương thường ngày."
        if n < 3:
            text += " Khách hàng mới có ít giao dịch nên phép so sánh này chỉ mang tính tham khảo."
        unusual = _amount_is_unusual(row) and ratio >= 1
        reasons.append((
            ":material/trending_up:" if unusual else ":material/check_circle:",
            "Số tiền lớn bất thường so với thói quen" if unusual else "So với thói quen chi tiêu",
            text,
        ))
    if row["new_receiver"] and n >= 1:
        reasons.append((
            ":material/person_search:", "Người nhận lần đầu",
            f"Khách hàng đã có {n} giao dịch nhưng **chưa từng chuyển tiền cho tài khoản này**. "
            "Kẻ gian thường yêu cầu nạn nhân chuyển vào một tài khoản lạ.",
        ))

    signs: list[str] = []
    history = scored[(scored["account_id"] == row["account_id"]) & (scored["tx_id"] < row["tx_id"])]
    device = row.get("device_id")
    if n >= 1 and device is not None and not pd.isna(device) and "device_id" in history:
        if device not in set(history["device_id"].dropna()):
            signs.append(":material/devices: Giao dịch thực hiện từ **một thiết bị chưa từng dùng** trước đây.")
    country = row.get("ip_country")
    if country is not None and not pd.isna(country) and country != "VN":
        signs.append(f":material/public: Truy cập từ **{COUNTRY_NAMES.get(country, country)}**, không phải Việt Nam.")
    ts = row.get("timestamp")
    if ts is not None and not pd.isna(ts) and pd.Timestamp(ts).hour < 5:
        signs.append(f":material/bedtime: Giao dịch lúc **{pd.Timestamp(ts):%H:%M} {day_part(ts)}**, khi hầu hết mọi người đang ngủ.")
    return reasons, signs


# --------------------------------------------------------------------------
# Tiền chảy đi đâu
# --------------------------------------------------------------------------

def _edges(view: pd.DataFrame) -> pd.DataFrame:
    e = view[view["counterparty_id"].notna()]
    return pd.DataFrame({
        "sender": e["account_id"].to_numpy(), "receiver": e["counterparty_id"].to_numpy(),
        "amount": e["amount"].to_numpy(), "time": e[time_col_of(e)].to_numpy(),
        "level": e["level"].to_numpy(),
    })


MULE_MAX_GAP = pd.Timedelta(minutes=60)  # tài khoản trung gian chuyển tiền đi rất nhanh...
MULE_AMOUNT_RANGE = (0.5, 2.0)  # ...và số tiền chuyển tiếp gần như giữ nguyên


def _index(edges: pd.DataFrame) -> dict:
    return {"out": dict(tuple(edges.groupby("sender"))), "in": dict(tuple(edges.groupby("receiver"))),
            "is_time": np.issubdtype(edges["time"].dtype, np.datetime64)}


def _follow(index: dict, start, t0, amount0, forward: bool, avoid: set, max_hops: int = 4) -> list:
    """Lần theo CHÍNH khoản tiền của giao dịch đang xem (xuôi: tiền đi tiếp; ngược: tiền từ đâu tới).

    Chỉ nối một bước khi nó giống hành vi tài khoản trung gian: xảy ra ngay sau đó
    (trong MULE_MAX_GAP) và số tiền tương đương (MULE_AMOUNT_RANGE). Không có hai
    điều kiện này, mọi tài khoản bình thường nhận rồi tiêu tiền vài giờ sau cũng
    bị nối thành "chuỗi" — nói quá sự thật.
    """
    gap = MULE_MAX_GAP if index["is_time"] else 1  # không có giờ thật: cho phép cùng hoặc bước kế tiếp
    lo, hi = MULE_AMOUNT_RANGE
    path, current, t_bound, amount = [start], start, t0, amount0
    for _ in range(max_hops):
        seen = set(path) | avoid
        if forward:
            nxt = index["out"].get(current)
            if nxt is None:
                break
            nxt = nxt[~nxt["receiver"].isin(seen) & (nxt["time"] >= t_bound) & (nxt["time"] <= t_bound + gap)]
        else:
            nxt = index["in"].get(current)
            if nxt is None:
                break
            nxt = nxt[~nxt["sender"].isin(seen) & (nxt["time"] <= t_bound) & (nxt["time"] >= t_bound - gap)]
        nxt = nxt[(nxt["amount"] >= lo * amount) & (nxt["amount"] <= hi * amount)]
        if nxt.empty:
            break
        e = nxt.sort_values("time").iloc[0 if forward else -1]
        current = e["receiver"] if forward else e["sender"]
        path.append(current)
        t_bound, amount = e["time"], e["amount"]
    return path


def _chain(index: dict, row, time_col: str) -> list:
    """Chuỗi tài khoản mà khoản tiền của giao dịch này đi qua (tối thiểu [người chuyển, người nhận])."""
    focus, receiver = row["account_id"], row["counterparty_id"]
    upstream = _follow(index, focus, row[time_col], row["amount"], forward=False, avoid={receiver})
    downstream = _follow(index, receiver, row[time_col], row["amount"], forward=True, avoid={focus})
    return upstream[::-1] + downstream


@st.cache_resource(show_spinner="Đang nạp kho tri thức kiểu hình rửa tiền...")
def get_typology_retriever(reference_signature: str, _reference: pd.DataFrame):
    """Bộ truy xuất kiểu hình, đã chuẩn hóa theo mặt bằng của chính bộ dữ liệu đang xem.

    Tiền tố `_` của `_reference` là BẮT BUỘC: Streamlit băm mọi tham số để
    làm khóa cache, mà DataFrame thì không băm được — thiếu dấu gạch dưới là
    trang nổ lúc chạy. Khóa cache thật nằm ở `reference_signature`, nên chuỗi
    đó phải đổi khi dữ liệu đổi.

    Lưu ý về mặt phương pháp: tham chiếu z-score ở đây fit trên TOÀN BỘ bộ dữ
    liệu dashboard đang nạp, không phải trên tập train riêng. Với dashboard
    thì đó chính là ngữ nghĩa đúng — "giao dịch này lệch bao nhiêu so với mặt
    bằng chung của lô đang xem". Khi đánh giá mô hình thì vẫn phải fit trên
    train (xem src/knowledge/retriever.py::fit_reference).
    """
    from src.knowledge import TypologyRetriever

    return TypologyRetriever().fit_reference(_reference)


def typology_narrative(scored: pd.DataFrame, row) -> object | None:
    """Lời giải thích theo kiểu hình rửa tiền cho một giao dịch, hoặc None.

    Trả về None khi bộ dữ liệu chưa có đủ đặc trưng mạng lưới (vd. nguồn
    IEEE-CIS không có người nhận nên không có đặc trưng đối tác nào).
    """
    from src.knowledge import SIGNATURE_FEATURES, build_narrative

    if not any(feature in scored.columns for feature in SIGNATURE_FEATURES):
        return None
    signature = f"{len(scored)}-{sorted(set(scored.columns) & set(SIGNATURE_FEATURES))}"
    try:
        retriever = get_typology_retriever(signature, scored)
        return build_narrative(row.to_dict(), retriever, risk_score=float(row["risk_score"]))
    except (ValueError, RuntimeError, KeyError):
        # Kho tri thức là phần BỔ SUNG cho dashboard. Nó hỏng thì trang vẫn
        # phải hiển thị được phần còn lại, nên nuốt lỗi ở đây là có chủ ý.
        return None


@st.cache_data(show_spinner="Đang lần theo các dòng tiền...", max_entries=12)
def chain_members(source: str, n_recent: int) -> dict:
    """tx_id -> chuỗi tài khoản (từ 3 tài khoản trở lên) mà khoản tiền của giao dịch đó bị chuyển tiếp qua.

    Đây là dấu hiệu MẠNG LƯỚI, quan sát bằng quy tắc minh bạch (chuyển tiếp nhanh,
    số tiền gần như giữ nguyên) — KHÔNG làm thay đổi mức độ đáng ngờ đã chấm.
    """
    view = load_scored(source).tail(n_recent)
    edges = _edges(view)
    if edges.empty:
        return {}
    index, tc = _index(edges), time_col_of(view)
    out = {}
    for row in view[view["counterparty_id"].notna()].to_dict("records"):
        chain = _chain(index, row, tc)
        if len(chain) >= 3:
            out[int(row["tx_id"])] = chain
    return out


def money_flow(view: pd.DataFrame, row, max_nodes: int = 40) -> dict:
    """Mạng lưới quanh giao dịch đang xem + các câu mô tả đường đi của khoản tiền đó."""
    edges = _edges(view)
    focus, receiver = row["account_id"], row["counterparty_id"]
    if edges.empty or receiver is None or pd.isna(receiver):
        return {"edges": edges.iloc[0:0], "nodes": [], "chain": [], "sentences": [
            "Bộ dữ liệu này không có thông tin người nhận, nên chưa vẽ được đường đi của tiền."]}

    chain = _chain(_index(edges), row, time_col_of(view))

    nodes, frontier = {focus} | set(chain), {focus}
    for _ in range(2):  # 2 bước quanh tài khoản đang xem (luôn giữ trọn chuỗi tiền)
        touch = edges[edges["sender"].isin(frontier) | edges["receiver"].isin(frontier)]
        frontier = (set(touch["sender"]) | set(touch["receiver"])) - nodes
        nodes |= frontier
        if len(nodes) >= max_nodes:
            break
    local = edges[edges["sender"].isin(nodes) & edges["receiver"].isin(nodes)]

    out_n = edges.loc[edges["sender"] == focus, "receiver"].nunique()
    in_n = edges.loc[edges["receiver"] == focus, "sender"].nunique()
    sentences = [f"Trong các giao dịch đang xem, tài khoản này **chuyển tiền cho {out_n} tài khoản** "
                 f"và **nhận tiền từ {in_n} tài khoản**."]
    if len(chain) >= 3:
        hops = set(zip(chain[:-1], chain[1:]))
        on_chain = edges[[(s, r) in hops for s, r in zip(edges["sender"], edges["receiver"])]]
        span = pd.Timestamp(on_chain["time"].max()) - pd.Timestamp(on_chain["time"].min()) \
            if np.issubdtype(on_chain["time"].dtype, np.datetime64) else None
        within = f" chỉ trong **{max(1, int(span.total_seconds() // 60))} phút**" if span is not None else ""
        sentences.append(
            f"Khoản tiền này được chuyển **nối tiếp qua {len(chain)} tài khoản**{within}, "
            "số tiền gần như giữ nguyên sau mỗi lần: " + " → ".join(mask_account(a) for a in chain) + ".")
        sentences.append(
            "Đây là dấu hiệu điển hình của **tài khoản trung gian** (còn gọi là *money mule*): "
            "kẻ gian chuyển tiền qua nhiều tài khoản liên tiếp để xóa dấu vết.")
    elif in_n >= 3:
        sentences.append("Tài khoản này **nhận tiền từ nhiều nguồn khác nhau** — cần xem có phải tài khoản gom tiền không.")
    else:
        sentences.append("Chưa thấy dấu hiệu chuyển tiền dây chuyền quanh tài khoản này.")
    return {"edges": local, "nodes": sorted(nodes, key=str), "chain": chain, "sentences": sentences}


def flow_figure(flow: dict, focus):
    import networkx as nx
    import plotly.graph_objects as go

    edges = flow["edges"]
    graph = nx.DiGraph()
    graph.add_nodes_from(flow["nodes"])
    graph.add_edges_from(zip(edges["sender"], edges["receiver"]))
    chain = flow["chain"]  # luôn có ít nhất [người chuyển, người nhận] của giao dịch đang xem
    hops = set(zip(chain[:-1], chain[1:]))
    fixed = {a: (i * 2.0, 0.0) for i, a in enumerate(chain)}
    pos = nx.spring_layout(graph, seed=7, pos=fixed or None, fixed=list(fixed) or None, k=1.2)

    rank = {"low": 0, "medium": 1, "critical": 2}
    node_level = {}
    for _, e in edges.iterrows():
        for acc in (e["sender"], e["receiver"]):
            if rank[e["level"]] > rank.get(node_level.get(acc), -1):
                node_level[acc] = e["level"]

    fig = go.Figure()
    for _, e in edges.iterrows():
        (x0, y0), (x1, y1) = pos[e["sender"]], pos[e["receiver"]]
        on_chain = (e["sender"], e["receiver"]) in hops
        fig.add_annotation(
            x=x1, y=y1, ax=x0, ay=y0, xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=3, arrowsize=1.2, standoff=14, startstandoff=14,
            arrowwidth=3 if on_chain else 1.5, arrowcolor="#c0392b" if on_chain else "#b0b7c3",
        )
    names = list(graph.nodes())
    colors = ["#1f3864" if n == focus else LEVELS[node_level[n]]["hex"] if n in node_level else "#b0b7c3"
              for n in names]
    hover = [f"{mask_account(n)}<br>Gửi {graph.out_degree(n)} lần · nhận {graph.in_degree(n)} lần" for n in names]
    fig.add_trace(go.Scatter(
        x=[pos[n][0] for n in names], y=[pos[n][1] for n in names], mode="markers+text",
        marker=dict(size=[30 if n == focus else 18 for n in names], color=colors,
                    line=dict(width=[4 if n == focus else 1 for n in names], color="white")),
        text=[mask_account(n) for n in names], textposition="bottom center",
        hovertext=hover, hoverinfo="text",
    ))
    fig.update_layout(showlegend=False, height=420, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis=dict(visible=False), yaxis=dict(visible=False),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    return fig


# --------------------------------------------------------------------------
# Ca trực xử lý gì trước
# --------------------------------------------------------------------------

def plan_shift(alerts_df: pd.DataFrame, top_n: int, hours: float) -> tuple[pd.DataFrame, dict]:
    """Chạy QAPE thật (qape_service) rồi so với cách làm thông thường: xử lý theo thứ tự đến."""
    from src.dashboard.qape_service import run_qape_for_dashboard

    budget = hours * 60 / MINUTES_PER_COST_UNIT
    result = run_qape_for_dashboard(alerts_df, budget=budget, top_n=top_n)
    cands = alerts_df.sort_values("risk_score", ascending=False).head(top_n).reset_index(drop=True)
    alerts = result["alerts"]
    cands["cost"] = [a.cost for a in alerts]
    cands["minutes"] = cands["cost"] * MINUTES_PER_COST_UNIT

    if result["ilp"] is not None:
        index_of = {a.alert_id: i for i, a in enumerate(alerts)}
        picked = {index_of[a] for a in result["ilp"].selected_alert_ids if a in index_of}
    else:
        # Không đủ giờ cho mọi cảnh báo khẩn cấp (bài toán vô nghiệm): không liệt kê
        # quá số giờ thực có — xử lý cảnh báo nặng nhất trước, trong phạm vi thời gian.
        used, picked = 0.0, set()
        for i in cands.index:  # cands đã sắp theo rủi ro giảm dần
            if used + cands.at[i, "cost"] <= budget + 1e-9:
                picked.add(i)
                used += cands.at[i, "cost"]
    cands["handle_now"] = cands.index.isin(picked)
    must_hours = cands.loc[cands["risk_score"] >= FORCE_SELECT_MIN, "minutes"].sum() / 60

    used, fifo = 0.0, set()
    for i in cands.sort_values(time_col_of(cands), kind="stable").index:
        if used + cands.at[i, "cost"] <= budget + 1e-9:
            fifo.add(i)
            used += cands.at[i, "cost"]
    cands["fifo"] = cands.index.isin(fifo)
    return cands, {"budget": budget, "exact": result["ilp"] is not None,
                   "not_enough_time": result["ilp"] is None, "must_hours": must_hours}
