"""Lớp ứng dụng / Inference Interface của FINGUARD AI.

Luồng: USER INPUT → kiểm tra đầu vào → đặc trưng đã tiền xử lý đúng pipeline train → mô hình đã
huấn luyện (GBDT 28 cột + DRRE R7, nạp từ prototype/bundle/models, không train lại) → chấm điểm
rủi ro → giải thích SHAP thật → hiển thị.

Mọi hàm ở đây chỉ gọi lại core.py (nạp mô hình, chấm điểm) và src/ của dự án.
"""

from __future__ import annotations

import time
from functools import lru_cache

import numpy as np
import pandas as pd

import core

ID_COLUMNS = ["account_id", "counterparty_id", "timestamp"]
OPTIONAL_COLUMNS = ["tx_ref", "tx_id", "amount", "currency", "channel", "label", "laundering_pattern",
                    "eval_universe"]


def input_schema() -> pd.DataFrame:
    """Lược đồ đầu vào lấy đúng từ checkpoint (tên + thứ tự 28 đặc trưng)."""
    ref = core.reference()
    meaning = {"account_id": "Tài khoản gửi", "counterparty_id": "Tài khoản nhận",
               "timestamp": "Thời điểm giao dịch"}
    rows = [{"Cột": c, "Kiểu": "thời gian (ISO 8601)" if c == "timestamp" else "chuỗi", "Bắt buộc": "có",
             "Ý nghĩa": meaning[c], "Khoảng thường gặp trên val (P1 … P99)": "—"} for c in ID_COLUMNS]
    for i, f in enumerate(core.features()):
        q = ref["feature_quantiles"][i]
        rows.append({"Cột": f, "Kiểu": "số thực", "Bắt buộc": "có", "Ý nghĩa": core.FEATURE_LABELS.get(f, f),
                     "Khoảng thường gặp trên val (P1 … P99)": f"{q[1]:.3g} … {q[99]:.3g}"})
    return pd.DataFrame(rows)


def validate(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Kiểm tra và chuẩn hóa kiểu. KHÔNG tự điền giá trị đặc trưng (làm vậy sẽ làm sai lệch kết quả
    mô hình): dòng thiếu/sai định dạng được đánh dấu và không đưa vào suy luận."""
    feats = core.features()
    missing = [c for c in ID_COLUMNS + feats if c not in df.columns]
    report = {"rows": int(len(df)), "missing_columns": missing,
              "extra_columns": [c for c in df.columns if c not in ID_COLUMNS + feats + OPTIONAL_COLUMNS]}
    if missing:
        report.update(ok_rows=0, bad_rows=int(len(df)))
        return df.assign(_status="Lỗi: thiếu cột bắt buộc"), report
    out = df.copy()
    problems = pd.Series("", index=out.index, dtype=object)
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    problems[out["timestamp"].isna()] += "thời gian sai định dạng; "
    for c in ("account_id", "counterparty_id"):
        bad = out[c].isna() | (out[c].astype(str).str.strip().isin(["", "nan", "None"]))
        problems[bad] += f"thiếu {c}; "
        out[c] = out[c].astype(str).str.strip()
    for f in feats:
        val = pd.to_numeric(out[f], errors="coerce").astype(float)
        bad = ~np.isfinite(val)
        if bad.any():
            problems[bad] += f"{f} trống hoặc không phải số; "
        out[f] = val.astype(np.float32)
    if "label" in out:
        out["label"] = pd.to_numeric(out["label"], errors="coerce").fillna(0).astype(bool)
    out["_status"] = np.where(problems == "", "Hợp lệ", "Lỗi: " + problems.str.rstrip("; "))
    report["ok_rows"] = int((out["_status"] == "Hợp lệ").sum())
    report["bad_rows"] = report["rows"] - report["ok_rows"]
    return out, report


@lru_cache(maxsize=1)
def shap_explainer():
    """SHAP TreeExplainer trên GBDT 28 cột (thành phần 95% của điểm hệ thống). Giá trị ở thang
    log-odds; nền + tổng đóng góp = đúng đầu ra mô hình (kiểm tra: sai số 1e-14)."""
    import shap

    return shap.TreeExplainer(core.gbdt_model())


def shap_available() -> bool:
    try:
        import shap  # noqa: F401
        return True
    except ImportError:
        return False


def shap_values(df: pd.DataFrame) -> tuple[float, np.ndarray]:
    ex = shap_explainer()
    values = np.asarray(ex.shap_values(df[core.features()].to_numpy(np.float32)))
    return float(np.ravel(ex.expected_value)[0]), values


# Mã kênh của tiền xử lý (data/processed_hi_medium/amlworld_categorical_mappings.json).
CHANNELS = {0: "ACH", 1: "Bitcoin", 2: "Tiền mặt", 3: "Séc", 4: "Thẻ tín dụng", 5: "Tái đầu tư", 6: "Chuyển khoản wire"}
BINARY = {"new_counterparty", "cb_cross_bank", "cy_reciprocity_7d"}


def value_text(feature: str, value: float) -> str:
    if feature == "channel_encoded":
        return CHANNELS.get(int(round(value)), str(value))
    if feature in BINARY:
        return "có" if value >= 0.5 else "không"
    return f"{value:.3g}"


def factor_text(feature: str, value: float, contribution: float) -> str:
    name = core.FEATURE_LABELS.get(feature, feature).split("] ")[-1]
    return f"{name}: {value_text(feature, value)} {'↑' if contribution > 0 else '↓'}"


def key_factors(values: np.ndarray, x: pd.DataFrame, k: int = 3) -> list[str]:
    feats = core.features()
    out = []
    for i in range(len(values)):
        order = [j for j in np.argsort(-values[i])[:k] if values[i, j] > 0]
        out.append("; ".join(factor_text(feats[j], float(x.iloc[i][feats[j]]), float(values[i, j]))
                             for j in order) or "Không có yếu tố tăng rủi ro nổi bật")
    return out


def analyze(df: pd.DataFrame, use_drre: bool = True, explain: bool = True) -> tuple[pd.DataFrame, dict]:
    """Phân tích một lô: kiểm tra → GBDT + DRRE thật → điểm, mức, xác suất → yếu tố SHAP."""
    checked, report = validate(df)
    ok = checked[checked["_status"] == "Hợp lệ"].drop(columns="_status")
    bad = checked[checked["_status"] != "Hợp lệ"]
    if ok.empty:
        return checked.assign(**{"Analysis Status": checked["_status"]}), {**report, "timing": {}}
    try:
        scored, timing = core.run_live(ok, use_drre=use_drre)
    except Exception as exc:  # noqa: BLE001 — báo lỗi suy luận thay vì trả kết quả sai
        return checked.assign(**{"Analysis Status": f"Lỗi suy luận: {exc}"}), {**report, "timing": {}}
    scored["fraud_probability"] = core.probability(scored["live_he_thong"].to_numpy())
    scored["risk_score"] = 100 * scored["live_he_thong"]
    scored["prediction"] = np.where(scored["live_level"] != "low", "Đáng ngờ", "Bình thường")
    if explain and shap_available():
        t = time.perf_counter()
        _, sv = shap_values(scored)
        timing["SHAP"] = time.perf_counter() - t
        scored["key_factors"] = key_factors(sv, scored)
    scored["Analysis Status"] = "Hoàn tất" if use_drre else "Hoàn tất (chỉ GBDT)"
    if not bad.empty:
        scored = pd.concat([scored, bad.assign(**{"Analysis Status": bad["_status"]})], ignore_index=True)
    return scored, {**report, "timing": timing}


def result_table(scored: pd.DataFrame) -> pd.DataFrame:
    """Bảng kết quả: Transaction ID, Prediction, Fraud Probability, Risk Score, Risk Level,
    Key Risk Factors, Analysis Status."""
    n = len(scored)
    if "tx_ref" in scored:
        tx = scored["tx_ref"].astype(str)
    elif "tx_id" in scored:
        tx = scored["tx_id"].map(lambda v: f"TX-{int(v):06d}" if pd.notna(v) else "—")
    else:
        tx = pd.Series([f"ROW-{i + 1:06d}" for i in range(n)], index=scored.index)
    lvl = scored["live_level"] if "live_level" in scored else pd.Series(np.nan, index=scored.index)
    col = lambda c, d: scored[c] if c in scored else pd.Series(d, index=scored.index)  # noqa: E731
    return pd.DataFrame({
        "Transaction ID": tx.values,
        "Thời điểm": pd.to_datetime(scored["timestamp"], errors="coerce").dt.strftime("%d/%m/%Y %H:%M").values,
        "Người gửi": scored["account_id"].astype(str).values,
        "Người nhận": scored["counterparty_id"].astype(str).values,
        "Prediction": col("prediction", "—").fillna("—").values,
        "Fraud Probability": col("fraud_probability", np.nan).values,
        "Risk Score": col("risk_score", np.nan).values,
        "Risk Level": [core.LEVELS[v][1] if v in core.LEVELS else "—" for v in lvl],
        "Key Risk Factors": col("key_factors", "").fillna("").values,
        "Analysis Status": scored["Analysis Status"].values,
    })


@lru_cache(maxsize=1)
def global_shap(n_background: int = 4000) -> pd.DataFrame:
    """Tầm quan trọng SHAP toàn cục: trung bình |SHAP| trên mọi cảnh báo của khung mẫu + một mẫu
    ngẫu nhiên cố định các giao dịch thường (để thấy cả yếu tố làm giảm rủi ro)."""
    df = core.sample()
    alerts = df[df["level"] != "low"]
    normal = df[df["level"] == "low"].sample(n=min(n_background, (df["level"] == "low").sum()), random_state=42)
    rows = pd.concat([alerts, normal])
    _, sv = shap_values(rows)
    feats = core.features()
    is_alert = (rows["level"] != "low").to_numpy()
    return pd.DataFrame({
        "feature": feats, "Đặc trưng": [core.FEATURE_LABELS.get(f, f) for f in feats],
        "mean_abs_shap": np.abs(sv).mean(0),
        "mean_shap_alerts": sv[is_alert].mean(0), "mean_shap_normal": sv[~is_alert].mean(0),
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
# Mạng lưới: cụm tài khoản nối với nhau bởi các giao dịch bị cảnh báo
# --------------------------------------------------------------------------

@lru_cache(maxsize=1)
def edges() -> pd.DataFrame:
    """Giao dịch dùng cho phân tích mạng lưới: khung mẫu 3 giờ + ngữ cảnh ±24h quanh 300 cảnh báo
    cao nhất (toàn bộ val). Điểm hệ thống của mọi cạnh là điểm thật đã chấm với đủ lịch sử."""
    cols = ["timestamp", "account_id", "counterparty_id", "amount", "currency", "label",
            "laundering_pattern", "score_he_thong"]
    parts = [core.sample()[cols]]
    ctx = core.graph_context()
    if not ctx.empty:
        parts.append(ctx[cols])
    e = pd.concat(parts, ignore_index=True).drop_duplicates(
        subset=["timestamp", "account_id", "counterparty_id", "amount"])
    e["account_id"] = e["account_id"].astype(str)
    e["counterparty_id"] = e["counterparty_id"].astype(str)
    e["level"] = core.level_of(e["score_he_thong"].to_numpy())
    return e.sort_values("timestamp").reset_index(drop=True)


@lru_cache(maxsize=1)
def alert_clusters() -> pd.DataFrame:
    """Thành phần liên thông của đồ thị chỉ gồm giao dịch bị cảnh báo (🔴/🟡). Cụm ≥ 3 tài khoản
    là một "mạng lưới đáng ngờ" — KHÔNG khẳng định là đường dây; tỷ lệ nhãn thật chỉ để kiểm chứng."""
    import networkx as nx

    e = edges()
    a = e[e["level"] != "low"]
    g = nx.Graph()
    g.add_edges_from(zip(a["account_id"], a["counterparty_id"]))
    rows = []
    for comp in sorted(nx.connected_components(g), key=len, reverse=True):
        if len(comp) < 3:
            continue
        sub = a[a["account_id"].isin(comp)]
        rows.append({"accounts": len(comp), "transactions": len(sub), "total_amount": float(sub["amount"].sum()),
                     "max_score": float(100 * sub["score_he_thong"].max()),
                     "critical": int((sub["level"] == "critical").sum()),
                     "label_share": float(sub["label"].astype(bool).mean()), "members": sorted(comp)})
    out = pd.DataFrame(rows)
    if not out.empty:
        out.insert(0, "cluster", [f"NET-{i + 1:03d}" for i in range(len(out))])
    return out


def account_risk() -> pd.DataFrame:
    """Rủi ro theo tài khoản: số giao dịch gửi/nhận, điểm cao nhất, số cảnh báo."""
    e = edges()
    out = e.groupby("account_id").agg(sent=("amount", "size"), sent_amount=("amount", "sum"),
                                      max_score=("score_he_thong", "max"),
                                      alerts=("level", lambda s: int((s != "low").sum())))
    inn = e.groupby("counterparty_id").agg(received=("amount", "size"), received_amount=("amount", "sum"))
    out = out.join(inn, how="outer").fillna(0)
    out["max_score"] = 100 * out["max_score"]
    return out.sort_values(["alerts", "max_score"], ascending=False)


# --------------------------------------------------------------------------
# Chỉ số đánh giá đã lưu (không sinh thêm)
# --------------------------------------------------------------------------

def confusion_at_f1(board: dict, model: str) -> dict:
    """Ma trận nhầm lẫn tại ngưỡng F1 tốt nhất, dựng từ số đã lưu trong bảng xếp hạng."""
    m = board["models"][model]
    tp, alerts = int(m["tp_at_f1"]), int(m["alerts_at_f1"])
    pos, n = int(board["n_pos"]), int(board["n_rows"])
    return {"TP": tp, "FP": alerts - tp, "FN": pos - tp, "TN": n - pos - (alerts - tp)}


def confusion_at_level(level: str) -> dict:
    """Ma trận nhầm lẫn của hệ thống tại ngưỡng một mức cảnh báo (số đã lưu khi dựng gói, val)."""
    th = core.results()["thresholds"]
    board = core.results()["leaderboard_hi_medium"]
    n, pos = int(board["n_rows"]), int(board["n_pos"])
    alerts = round(th[f"{level}_top_share"] * n)
    tp = round(th[f"{level}_recall"] * pos)
    return {"TP": tp, "FP": alerts - tp, "FN": pos - tp, "TN": n - pos - (alerts - tp)}
