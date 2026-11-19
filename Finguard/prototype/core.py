"""Lõi của prototype FinGuard AML: nạp gói dữ liệu/mô hình, chấm điểm thật, giải thích.

Các trang (prototype/views/*.py) chỉ lo hiển thị; mọi phép tính nằm ở đây để kiểm thử được
mà không cần chạy Streamlit.

Gói prototype/bundle/ được dựng bởi scripts/build_prototype_bundle.py từ kết quả đợt 2:
  * data/hi_medium_val_window.parquet: 3 giờ liên tục của val HI-Medium (AMLWorld) kèm điểm
    đã chấm với ĐỦ lịch sử (bộ nhớ hành vi + đồ thị quá khứ nối từ đầu train);
  * models/: GBDT 28 cột (joblib) và mọi checkpoint DRRE R2–R7;
  * reference.npz: phân vị điểm trên toàn val để đưa lô mới về cùng thang;
  * results.json: bảng xếp hạng, đường học, P2, ngưỡng cảnh báo.
"""

from __future__ import annotations

import json
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent  # thư mục Finguard/ (chứa src/)
sys.path.insert(0, str(ROOT))

BUNDLE = HERE / "bundle"
SAMPLE_PATH = BUNDLE / "data/hi_medium_val_window.parquet"
GBDT_PATH = BUNDLE / "models/gbdt_28_hi_medium.joblib"
DRRE_R7_PATH = BUNDLE / "models/drre_r7_hi_medium.pt"

# Tên dễ hiểu của 28 đặc trưng (18 đặc trưng nền + 3 mô-đun kịch bản đã chốt ở R1).
FEATURE_LABELS = {
    "amount_norm": "Số tiền (đã chuẩn hóa)",
    "channel_encoded": "Kênh thanh toán",
    "log_n_prior": "Số giao dịch trước đó của người gửi",
    "log_gap": "Thời gian kể từ giao dịch trước",
    "z_amount": "Độ lệch số tiền so với thói quen",
    "log_amount_ratio": "Số tiền so với mức trung bình của người gửi",
    "new_counterparty": "Lần đầu chuyển cho người nhận này",
    "log_n_distinct_cp": "Số người nhận khác nhau",
    "log_n_prior_as_cp": "Số lần người gửi từng nhận tiền",
    "log_n_distinct_senders": "Số người khác nhau từng gửi tiền cho người gửi",
    "hour_cos": "Giờ trong ngày",
    "dow": "Thứ trong tuần",
    "log_amt_out_prior": "Tổng tiền người gửi đã chuyển đi",
    "log_amt_in_prior": "Tổng tiền người gửi đã nhận vào",
    "log_n_in_prior": "Số lần người gửi đã nhận tiền",
    "passthrough_ratio": "Tỷ lệ tiền 'đi qua' (nhận vào rồi chuyển ra)",
    "flow_imbalance": "Lệch giữa tiền vào và tiền ra",
    "log_cp_amt_in_prior": "Tổng tiền người nhận đã nhận trước đó",
    "mc_forward_ratio_24h": "[Chuỗi trung gian] Tỷ lệ tiền chuyển tiếp trong 24 giờ",
    "mc_amt_sim_last_in": "[Chuỗi trung gian] Số tiền giống khoản vừa nhận",
    "mc_log_min_since_in": "[Chuỗi trung gian] Phút kể từ khoản nhận gần nhất",
    "mc_log_age_days_cap3": "[Chuỗi trung gian] Tuổi tài khoản (chặn ở 3 ngày)",
    "cb_cross_bank": "[Liên ngân hàng] Chuyển sang ngân hàng khác",
    "cb_log_n_banks_7d": "[Liên ngân hàng] Số ngân hàng giao dịch trong 7 ngày",
    "cb_log_amount_usd": "[Liên ngân hàng] Số tiền quy đổi USD",
    "cy_reciprocity_7d": "[Vòng tròn] Có giao dịch ngược chiều trong 7 ngày",
    "cy_log_n_reverse_7d": "[Vòng tròn] Số giao dịch ngược chiều trong 7 ngày",
    "cy_log_min_since_reverse": "[Vòng tròn] Phút kể từ giao dịch ngược chiều",
}

PATTERN_LABELS = {
    "FAN-OUT": "Tỏa ra (một tài khoản chia tiền cho nhiều nơi)",
    "FAN-IN": "Gom về (nhiều nơi dồn tiền vào một tài khoản)",
    "GATHER-SCATTER": "Gom rồi tỏa",
    "SCATTER-GATHER": "Tỏa rồi gom",
    "CYCLE": "Vòng tròn (tiền quay về chỗ cũ)",
    "STACK": "Xếp tầng (nhiều lớp trung gian)",
    "BIPARTITE": "Hai phía (nhóm gửi ↔ nhóm nhận)",
    "RANDOM": "Ngẫu nhiên",
    "UNSPECIFIED": "Không rõ kiểu (chỉ có nhãn)",
}

LEVELS = {
    "critical": ("🔴", "Chặn ngay", "Tạm giữ giao dịch, xác minh chủ tài khoản, lập báo cáo giao dịch đáng ngờ (STR)."),
    "medium": ("🟡", "Cần xác minh", "Đưa vào hàng đợi kiểm tra của chuyên viên trong ca."),
    "low": ("🟢", "Bình thường", "Cho giao dịch đi qua, chỉ lưu vết."),
}


# --------------------------------------------------------------------------
# Nạp gói
# --------------------------------------------------------------------------

@lru_cache(maxsize=1)
def results() -> dict:
    return json.loads((BUNDLE / "results.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def reference() -> dict[str, np.ndarray]:
    ref = np.load(BUNDLE / "reference.npz", allow_pickle=False)
    return {k: ref[k] for k in ref.files}


@lru_cache(maxsize=1)
def sample() -> pd.DataFrame:
    df = pd.read_parquet(SAMPLE_PATH)
    df["level"] = level_of(df["score_he_thong"].to_numpy())
    df["tx_id"] = np.arange(len(df))
    return df


def features() -> list[str]:
    return list(results()["features"])


@lru_cache(maxsize=1)
def gbdt_model():
    import joblib

    return joblib.load(GBDT_PATH)


@lru_cache(maxsize=2)
def drre_model(path: str = str(DRRE_R7_PATH)):
    from src.dashboard.scoring_service import try_load_drre_checkpoint

    loaded = try_load_drre_checkpoint(path)
    if loaded is None:
        raise RuntimeError(f"Không nạp được checkpoint DRRE {path}")
    return loaded


# --------------------------------------------------------------------------
# Thang điểm và mức cảnh báo
# --------------------------------------------------------------------------

def to_percentile(values: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    """Đưa điểm thô về phân vị (0..1) theo phân phối trên toàn val HI-Medium."""
    return np.searchsorted(quantiles, np.asarray(values, dtype=np.float64), side="right") / len(quantiles)


def system_score(pct_gbdt: np.ndarray, pct_drre: np.ndarray) -> np.ndarray:
    w = float(results()["w_drre"])
    return (1 - w) * pct_gbdt + w * pct_drre


def level_of(system: np.ndarray) -> np.ndarray:
    th = results()["thresholds"]
    return np.where(system >= th["critical"], "critical",
                    np.where(system >= th["medium"], "medium", "low"))


def probability(system: np.ndarray) -> np.ndarray:
    """Xác suất là rửa tiền ước tính từ điểm hệ thống (hồi quy đẳng trương trên val HI-Medium)."""
    cal = results()["calibration"]
    return np.interp(np.asarray(system, dtype=np.float64), cal["x"], cal["y"])


def score_100(system: float) -> float:
    """Điểm hiển thị 0–100: phân vị của giao dịch trong toàn bộ val."""
    return 100.0 * float(system)


# --------------------------------------------------------------------------
# Chấm điểm thật
# --------------------------------------------------------------------------

def score_gbdt(df: pd.DataFrame) -> np.ndarray:
    return gbdt_model().predict_proba(df[features()].to_numpy(np.float32))[:, 1]


def score_drre_live(df: pd.DataFrame, checkpoint: str = str(DRRE_R7_PATH)) -> np.ndarray:
    """Chấm DRRE thật trên `df` (đã sắp theo thời gian), bộ nhớ hành vi bắt đầu TRỐNG.

    Khác với điểm trong gói (nối trạng thái từ đầu train), đây là "khởi động lạnh": tài khoản
    chưa có lịch sử trong lô sẽ được chấm như tài khoản mới.
    """
    import torch

    from src.drre.past_graph import make_buffer
    from src.drre.train import BehaviorStateStore
    from src.evaluation.stateful_scoring import required_capacity, score_split, stream_columns

    model, engine = drre_model(checkpoint)
    frame = df[stream_columns(model)].reset_index(drop=True)
    store = BehaviorStateStore(model.behavior_memory.hidden_dim, model.behavior_memory.num_slots,
                               capacity=required_capacity([frame]))
    with torch.no_grad():
        return score_split(model, engine, frame, store, window=32768, graph_buffer=make_buffer(model))


def run_live(df: pd.DataFrame, use_drre: bool = True) -> tuple[pd.DataFrame, dict]:
    """Chấm cả hệ thống trên một lô; trả về bảng điểm và thời gian chạy từng bước."""
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    ref = reference()
    timing = {}
    t = time.perf_counter()
    g = score_gbdt(df)
    timing["gbdt_28"] = time.perf_counter() - t
    out = pd.DataFrame({"live_gbdt_28": g, "live_pct_gbdt_28": to_percentile(g, ref["gbdt_28"])})
    if use_drre:
        t = time.perf_counter()
        d = score_drre_live(df)
        timing["R7"] = time.perf_counter() - t
        out["live_R7"] = d
        out["live_pct_R7"] = to_percentile(d, ref["R7"])
        out["live_he_thong"] = system_score(out["live_pct_gbdt_28"].to_numpy(), out["live_pct_R7"].to_numpy())
    else:
        out["live_he_thong"] = out["live_pct_gbdt_28"]
    out["live_level"] = level_of(out["live_he_thong"].to_numpy())
    return pd.concat([df, out], axis=1), timing


# --------------------------------------------------------------------------
# Chỉ số đánh giá
# --------------------------------------------------------------------------

def average_precision(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    y = np.asarray(y, bool)
    return float(average_precision_score(y, s)) if y.any() else float("nan")


def evaluate(df: pd.DataFrame, columns: dict[str, str]) -> pd.DataFrame:
    """Bảng chỉ số cho các cột điểm trên các dòng thuộc vũ trụ chấm điểm có nhãn."""
    from sklearn.metrics import roc_auc_score

    keep = df["eval_universe"].fillna(True).to_numpy(bool) if "eval_universe" in df else np.ones(len(df), bool)
    y = df["label"].fillna(False).astype(bool).to_numpy()[keep]
    rows = []
    for name, col in columns.items():
        s = df[col].to_numpy(np.float64)[keep]
        order = np.argsort(-s, kind="stable")
        k100 = min(100, len(s))
        rows.append({
            "Mô hình": name,
            "AUPRC": average_precision(y, s),
            "ROC-AUC": float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan"),
            "Bắt được trong 100 cảnh báo đầu": int(y[order[:k100]].sum()),
            "Độ chính xác @100": float(y[order[:k100]].mean()) if k100 else float("nan"),
            "Recall @ top 1%": float(y[order[:max(1, len(s) // 100)]].sum() / max(y.sum(), 1)),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Giải thích một cảnh báo
# --------------------------------------------------------------------------

def feature_percentiles(row: pd.Series) -> dict[str, float]:
    ref = reference()
    out = {}
    for i, f in enumerate(features()):
        q = ref["feature_quantiles"][i]
        v = float(row[f])
        out[f] = float(np.searchsorted(q, v, side="right") - 1)  # 0..100
    return out


def gbdt_attribution(row: pd.Series) -> pd.DataFrame:
    """Đóng góp từng đặc trưng vào điểm GBDT của MỘT giao dịch, bằng phép thử "nếu bình thường":
    thay đặc trưng đó bằng trung vị trên val, đo điểm (log-odds) giảm bao nhiêu. Dương = đẩy rủi ro lên.
    """
    ref = reference()
    feats = features()
    x = row[feats].to_numpy(np.float32)[None, :]
    medians = ref["feature_quantiles"][:, 50].astype(np.float32)
    probe = np.repeat(x, len(feats), axis=0)
    probe[np.arange(len(feats)), np.arange(len(feats))] = medians
    model = gbdt_model()
    eps = 1e-6
    base = float(np.clip(model.predict_proba(x)[0, 1], eps, 1 - eps))
    changed = np.clip(model.predict_proba(probe)[:, 1], eps, 1 - eps)
    logit = lambda p: np.log(p / (1 - p))  # noqa: E731
    contrib = logit(base) - logit(changed)
    pct = feature_percentiles(row)
    table = pd.DataFrame({
        "feature": feats,
        "Đặc trưng": [FEATURE_LABELS.get(f, f) for f in feats],
        "Đóng góp (log-odds)": contrib,
        "Phân vị trên val": [pct[f] for f in feats],
        "Giá trị": x[0],
    })
    return table.sort_values("Đóng góp (log-odds)", ascending=False, key=np.abs).reset_index(drop=True)


@lru_cache(maxsize=1)
def graph_context() -> pd.DataFrame:
    path = BUNDLE / "data/graph_context.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["alert_tx"])


def money_flow(df: pd.DataFrame, row: pd.Series, hops: int = 2, max_edges: int = 80) -> pd.DataFrame:
    """Giao dịch quanh một cảnh báo để vẽ đường đi của tiền.

    Với 300 cảnh báo điểm cao nhất: lấy từ ngữ cảnh đồ thị trong gói (toàn bộ val, ±24 giờ, 2 bước).
    Giao dịch khác: tìm trong khung mẫu 3 giờ, mở rộng `hops` bước theo cả hai chiều."""
    ctx = graph_context()
    hit = ctx[ctx["alert_tx"] == int(row["tx_id"])]
    if len(hit):
        out = hit.copy()
        out["is_alert"] = out["is_alert"].astype(bool)
        return out.sort_values("timestamp").head(max_edges * 2)
    accounts = {row["account_id"], row["counterparty_id"]}
    touch = df.iloc[0:0]
    for _ in range(hops):
        touch = df[df["account_id"].isin(accounts) | df["counterparty_id"].isin(accounts)]
        if len(touch) > 4 * max_edges:
            break
        accounts = set(touch["account_id"]) | set(touch["counterparty_id"])
    touch = touch.copy()
    touch["gap"] = (touch["timestamp"] - row["timestamp"]).abs()
    near = touch.sort_values("gap").head(max_edges)
    near["is_alert"] = near["tx_id"] == row["tx_id"]
    if not near["is_alert"].any():
        near = pd.concat([near, df[df["tx_id"] == row["tx_id"]].assign(is_alert=True)])
    return near.sort_values("timestamp")


@lru_cache(maxsize=1)
def default_alert() -> int:
    """Ví dụ mặc định của trang giải thích: cảnh báo có nhãn rửa tiền mà mạng lưới quanh nó
    có nhiều giao dịch rửa tiền nhất (thấy rõ đường dây)."""
    df = sample()
    ctx = graph_context()
    if ctx.empty:
        return int(df.sort_values("score_he_thong").iloc[-1]["tx_id"])
    labeled = set(df.loc[df["label"].astype(bool), "tx_id"].astype(int))
    by = ctx[ctx["alert_tx"].isin(labeled)].groupby("alert_tx")["label"].sum()
    return int(by.idxmax()) if len(by) else int(ctx["alert_tx"].iloc[0])


def dec(x: float, digits: int = 3, sign: bool = False) -> str:
    """Số thập phân kiểu Việt Nam (dấu phẩy)."""
    return (f"{x:+.{digits}f}" if sign else f"{x:.{digits}f}").replace(".", ",")


def fmt_pct(x: float, digits: int = 1) -> str:
    return f"{100 * x:.{digits}f}%".replace(".", ",")


def fmt_num(x: float, digits: int = 0) -> str:
    return f"{x:,.{digits}f}".replace(",", "X").replace(".", ",").replace("X", ".")
