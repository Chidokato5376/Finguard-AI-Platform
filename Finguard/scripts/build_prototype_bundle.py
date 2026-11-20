"""Dựng gói dữ liệu cho prototype web (prototype/bundle/) từ kết quả đợt 2.

Gói gồm:
  data/hi_medium_val_window.parquet  một khung giờ liên tục của val HI-Medium: 28 đặc trưng,
                                     nhãn, kiểu hình, và ĐIỂM ĐÃ CHẤM VỚI ĐỦ LỊCH SỬ
                                     (gbdt_28, R7, gbdt_20, rf_28, he_thong = trộn thứ hạng)
  models/gbdt_28_hi_medium.joblib    GBDT 28 cột, đúng cấu hình bảng xếp hạng (200 cây, seed 42)
  models/drre_r*.pt                  mọi checkpoint DRRE của thang chạy thử R2–R7
  reference.npz                      phân vị điểm gbdt_28/R7 trên toàn val (để chấm lô mới
                                     ra cùng thang), phân vị từng đặc trưng (để giải thích)
  results.json                       bảng xếp hạng HI-Small/HI-Medium, đường học, P2, P5...

Chỉ dùng val và train. KHÔNG đọc tập test (test chạy một lần ở R10).

    python -X utf8 scripts/build_prototype_bundle.py
"""

from __future__ import annotations

import json
import re
import shutil
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.scenarios.store import read_split  # noqa: E402

MEDIUM = ROOT / "data/processed_hi_medium"
CK_R7 = ROOT / "models/drre_r7_hi_medium.pt"
SCORES_M = ROOT / "reports/r7/leaderboard/processed_hi_medium_val/scores"
OUT = ROOT / "prototype/bundle"
WINDOW = ("2022-09-13 09:00", "2022-09-13 12:00")  # 3 giờ liên tục trong val
W_DRRE = 0.05  # trọng số DRRE của trộn thứ hạng, chọn trên nửa đầu val (p2_stacking)
N_QUANTILES = 100_001
CHECKPOINTS = {
    "R2": "drre_r2_hi_small.pt", "R3": "drre_r3_hi_small.pt", "R4": "drre_r4_hi_small.pt",
    "R5": "drre_r5_hi_small.pt", "R6": "drre_r6_hi_small.pt", "R6a": "drre_r6a_hi_small.pt",
    "R6b": "drre_r6b_hi_small.pt", "R7": "drre_r7_hi_medium.pt",
}
CURVE_LOGS = {
    "R2": "r2_hi_small", "R3": "r3_hi_small", "R4": "r4_hi_small", "R5": "r5_hi_small",
    "R6b": "r6b_hi_small", "R7": "r7_chain", "v1 HI-Medium (đợt 1, đồ thị nhìn trước)": "train_hi_medium",
}


def percentile_of(reference_sorted: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Tỷ lệ giao dịch val có điểm NHỎ HƠN HOẶC BẰNG giá trị (0..1)."""
    return np.searchsorted(reference_sorted, values, side="right") / len(reference_sorted)


def calibration(system_u: np.ndarray, y_u: np.ndarray, n_knots: int = 400) -> dict:
    """Xác suất rửa tiền ước tính theo điểm hệ thống (hồi quy đẳng trương trên val).
    Lưu dạng nút (x, y) để nội suy; nút dày ở đuôi trên, nơi có cảnh báo."""
    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(system_u, y_u.astype(float))
    grid = np.unique(np.concatenate([np.linspace(0, 0.99, 100),
                                     1 - np.geomspace(1e-2, 1e-6, n_knots - 100)]))
    return {"x": grid.tolist(), "y": iso.predict(grid).tolist()}


def graph_context(val: pd.DataFrame, system: np.ndarray, sample: pd.DataFrame, n_alerts: int = 300,
                  hours: int = 24, per_alert: int = 150) -> pd.DataFrame:
    """Giao dịch quanh mỗi cảnh báo điểm cao của khung mẫu, lấy từ TOÀN BỘ val trong ±`hours` giờ:
    chạm người gửi/người nhận (bước 1) và chạm các tài khoản đó (bước 2); giữ `per_alert` giao dịch
    gần thời điểm cảnh báo nhất. Dùng để vẽ đường đi của tiền vượt ra ngoài khung 3 giờ."""
    cols = ["timestamp", "account_id", "counterparty_id", "amount", "currency", "label", "laundering_pattern"]
    frame = val[cols].copy()
    frame["score_he_thong"] = system.astype(np.float32)
    frame["laundering_pattern"] = frame["laundering_pattern"].astype(object).fillna("").astype(str)
    for c in ("account_id", "counterparty_id", "currency"):
        frame[c] = frame[c].astype(str)
    frame = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
    ts = frame["timestamp"].to_numpy()
    top = sample.sort_values("score_he_thong", ascending=False).head(n_alerts)
    parts = []
    for r in top.itertuples():
        lo, hi = np.searchsorted(ts, [np.datetime64(r.timestamp - pd.Timedelta(hours=hours)),
                                      np.datetime64(r.timestamp + pd.Timedelta(hours=hours))])
        near = frame.iloc[lo:hi]
        accounts = {r.account_id, r.counterparty_id}
        for _ in range(2):
            hit = near[near["account_id"].isin(accounts) | near["counterparty_id"].isin(accounts)]
            accounts = set(hit["account_id"]) | set(hit["counterparty_id"])
        gap = (hit["timestamp"] - r.timestamp).abs()
        keep = hit.loc[gap.sort_values(kind="stable").index[:per_alert]].copy()
        keep["alert_tx"] = int(r.tx_id)
        keep["is_alert"] = ((keep["account_id"] == r.account_id) & (keep["counterparty_id"] == r.counterparty_id)
                            & (keep["timestamp"] == r.timestamp))
        parts.append(keep)
    return pd.concat(parts, ignore_index=True)


def curve_from_log(name: str) -> list[float]:
    text = (ROOT / "logs" / f"{name}.log").read_text(encoding="utf-8", errors="replace")
    epochs: dict[int, float] = {}
    for k, v in re.findall(r"Epoch (\d+) \| Val AUPRC = ([0-9.]+)", text):
        epochs[int(k)] = float(v)  # lần chạy sau cùng của mỗi epoch thắng
    return [epochs[k] for k in sorted(epochs)]


def main() -> int:
    t0 = time.monotonic()
    (OUT / "data").mkdir(parents=True, exist_ok=True)
    (OUT / "models").mkdir(parents=True, exist_ok=True)

    features = list(torch.load(CK_R7, map_location="cpu", weights_only=True)["feature_columns"])
    base_cols = ["account_id", "counterparty_id", "timestamp", "amount", "currency", "channel",
                 "label", "laundering_pattern", "eval_universe"]
    val = read_split(MEDIUM / "amlworld_val.parquet", features + base_cols)
    scores = {n: np.load(SCORES_M / f"{n}.npy") for n in ("gbdt_28", "R7", "gbdt_20", "rf_28")}
    assert all(len(s) == len(val) for s in scores.values())
    universe = val["eval_universe"].fillna(True).to_numpy(bool)

    # Thang chung: phân vị trên toàn val (vũ trụ chấm điểm). Trộn thứ hạng = 0,95·pct_GBDT + 0,05·pct_DRRE.
    ref_g = np.sort(scores["gbdt_28"][universe])
    ref_d = np.sort(scores["R7"][universe])
    pct_g = percentile_of(ref_g, scores["gbdt_28"])
    pct_d = percentile_of(ref_d, scores["R7"])
    system = (1 - W_DRRE) * pct_g + W_DRRE * pct_d
    y_u = val["label"].fillna(False).astype(bool).to_numpy()[universe]
    sys_u = system[universe]
    ref_sys = np.sort(sys_u)

    # Ngưỡng 3 mức chọn trên val: "Chặn ngay" = ngưỡng F1 tốt nhất; "Cần xác minh" = đủ bắt 70% vụ.
    order = np.argsort(-sys_u, kind="stable")
    tp = np.cumsum(y_u[order])
    k = np.arange(1, len(order) + 1)
    f1 = 2 * tp / (k + y_u.sum())
    k_crit = int(np.argmax(f1)) + 1
    k_med = int(np.searchsorted(tp, 0.70 * y_u.sum())) + 1
    thresholds = {
        "critical": float(sys_u[order[k_crit - 1]]), "medium": float(sys_u[order[k_med - 1]]),
        "critical_top_share": k_crit / len(order), "medium_top_share": k_med / len(order),
        "critical_precision": float(tp[k_crit - 1] / k_crit), "critical_recall": float(tp[k_crit - 1] / y_u.sum()),
        "medium_precision": float(tp[k_med - 1] / k_med), "medium_recall": float(tp[k_med - 1] / y_u.sum()),
    }
    print("Ngưỡng:", json.dumps(thresholds, indent=1))

    # Khung giờ mẫu (thứ tự thời gian), kèm điểm đã chấm với đủ lịch sử.
    ts = val["timestamp"]
    keep = ((ts >= WINDOW[0]) & (ts < WINDOW[1])).to_numpy()
    sample = val.loc[keep].copy()
    for n, s in scores.items():
        sample[f"score_{n}"] = s[keep].astype(np.float32)
    sample["pct_gbdt_28"] = pct_g[keep].astype(np.float32)
    sample["pct_R7"] = pct_d[keep].astype(np.float32)
    sample["score_he_thong"] = system[keep].astype(np.float32)
    sample = sample.sort_values("timestamp", kind="stable").reset_index(drop=True)
    # Cột phân loại mang theo TOÀN BỘ từ điển của val (hàng triệu tài khoản) — đổi sang chuỗi cho gọn.
    for c in ("account_id", "counterparty_id", "currency", "channel"):
        sample[c] = sample[c].astype(str)
    sample["laundering_pattern"] = sample["laundering_pattern"].astype(object).fillna("").astype(str)
    sample.to_parquet(OUT / "data/hi_medium_val_window.parquet", index=False, compression="zstd")
    print(f"Mẫu: {len(sample):,} dòng, {int(sample['label'].sum())} vụ, "
          f"{sample['account_id'].nunique():,} tài khoản gửi")
    context = graph_context(val, system, sample.assign(tx_id=np.arange(len(sample))))
    context.to_parquet(OUT / "data/graph_context.parquet", index=False, compression="zstd")
    print(f"Ngữ cảnh đồ thị: {len(context):,} giao dịch quanh {context['alert_tx'].nunique()} cảnh báo")
    if "--only-graph" in sys.argv:
        return 0

    # GBDT 28 cột: học lại đúng cấu hình bảng xếp hạng rồi kiểm tra trùng điểm đã lưu.
    from sklearn.ensemble import HistGradientBoostingClassifier

    train = read_split(MEDIUM / "amlworld_train.parquet", features + ["label", "eval_universe"])
    fit = train["eval_universe"].fillna(True).to_numpy(bool)
    model = HistGradientBoostingClassifier(max_iter=200, class_weight="balanced",
                                           early_stopping=False, random_state=42)
    model.fit(train.loc[fit, features].to_numpy(np.float32),
              train.loc[fit, "label"].astype(bool).to_numpy())
    del train
    again = model.predict_proba(sample[features].to_numpy(np.float32))[:, 1]
    gap = float(np.max(np.abs(again - sample["score_gbdt_28"].to_numpy(np.float64))))
    print(f"GBDT học lại so với điểm bảng xếp hạng: lệch lớn nhất {gap:.2e}")
    joblib.dump(model, OUT / "models/gbdt_28_hi_medium.joblib", compress=3)

    # Tầm quan trọng (hoán vị trên khung mẫu, đo bằng AUPRC) để giải thích.
    from sklearn.inspection import permutation_importance

    su = sample["eval_universe"].fillna(True).to_numpy(bool)
    imp = permutation_importance(model, sample.loc[su, features].to_numpy(np.float32),
                                 sample.loc[su, "label"].astype(bool).to_numpy(),
                                 scoring="average_precision", n_repeats=3, random_state=42, n_jobs=1)
    importance = dict(zip(features, imp.importances_mean.astype(float).tolist()))

    # Phân vị đặc trưng trên val (101 mốc) để nói "cao hơn x% giao dịch".
    feat_q = np.stack([np.nanpercentile(val.loc[universe, f].to_numpy(np.float64), np.arange(101))
                       for f in features]).astype(np.float32)
    grid = np.linspace(0, 1, N_QUANTILES)
    np.savez_compressed(
        OUT / "reference.npz",
        gbdt_28=np.quantile(ref_g, grid).astype(np.float64), R7=np.quantile(ref_d, grid).astype(np.float64),
        he_thong=np.quantile(ref_sys, grid).astype(np.float64), features=np.array(features),
        feature_quantiles=feat_q)

    for name, file in CHECKPOINTS.items():
        shutil.copy2(ROOT / "models" / file, OUT / "models" / file)

    def board(path: Path) -> dict:
        return json.loads(path.read_text(encoding="utf-8"))

    results = {
        "window": WINDOW, "w_drre": W_DRRE, "thresholds": thresholds, "features": features,
        "calibration": calibration(sys_u, y_u),
        "importance": importance,
        "leaderboard_hi_small": board(ROOT / "reports/leaderboard/processed_aml_v2_val/leaderboard.json"),
        "leaderboard_hi_medium": board(ROOT / "reports/r7/leaderboard/processed_hi_medium_val/leaderboard.json"),
        "p2_hi_small": board(ROOT / "reports/gate1/p2_stacking_R5.json"),
        "p2_hi_medium": board(ROOT / "reports/r7/leaderboard/processed_hi_medium_val/p2_stacking_R7.json"),
        "curves": {name: curve_from_log(log) for name, log in CURVE_LOGS.items()},
        "checkpoints": {n: f for n, f in CHECKPOINTS.items()},
    }
    (OUT / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1, default=str),
                                      encoding="utf-8")
    total = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(f"Xong trong {time.monotonic() - t0:.0f} s — gói {total / 1e6:.1f} MB ở {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
