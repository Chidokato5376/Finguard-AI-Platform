"""Đánh giá một checkpoint trên tập TEST — tái lập bằng một lệnh.

Vì sao có file này: `src/drre/train.py` chỉ chấm trên tập val (để CHỌN
checkpoint). Mọi số liệu test trước đây được chạy bằng script rời ngoài
repo, nên không ai trong nhóm tái lập được. File này đưa bước đó vào dự
án, ghi kết quả ra `reports/eval/` kèm manifest truy vết.

Nguyên tắc: tập test chỉ được chạm MỘT LẦN sau khi đã chốt model bằng tập
val. Nếu bạn chạy file này nhiều lần để thử cấu hình, bạn đang dùng test
làm val — và số liệu sẽ lạc quan hơn thực tế.

Ví dụ:

    python scripts/evaluate.py --source ieee_cis --checkpoint models/drre_ieee.pt
    python scripts/evaluate.py --source ieee_cis --checkpoint models/drre_ieee.pt \\
        --compare-checkpoint models/drre_ieee_no_behavior.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Console Windows mặc định là cp1252, không encode được tiếng Việt — nếu
# không đổi, script chạy xong mọi phép tính rồi chết ở bước in bảng.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from src.augmentation.protocol import (  # noqa: E402
    bootstrap_metric_diff,
    expected_calibration_error,
    false_positive_rate_at_budget,
    pr_auc,
    recall_at_budget,
)
from src.evaluation.metrics import (  # noqa: E402
    best_f1_threshold,
    confusion_at_budget,
    confusion_at_threshold,
)
from src.tracking import RunManifest, dataframe_summary, file_fingerprint  # noqa: E402

logger = logging.getLogger("evaluate")

#: Số giao dịch nghi nhất được "cảnh báo" — khớp với ràng buộc năng lực
#: điều tra trong QAPE. Đây là con số dễ hiểu nhất cho người không chuyên:
#: "trong N giao dịch nghi nhất, bắt được bao nhiêu vụ".
DEFAULT_ALERT_BUDGETS = (500, 1000, 2000)

#: Tỷ lệ dòng mỗi cây Random Forest được lấy mẫu. Đặt nhỏ hơn 1 có chủ đích:
#: tập train AMLWorld có 3,55 triệu dòng còn máy của nhóm chỉ có 7,4 GB RAM,
#: nên 100 cây dựng trên toàn bộ dòng sẽ hết bộ nhớ. `max_samples` là tham số
#: sẵn có của sklearn chứ không phải mẹo tự chế, và vẫn giữ đúng tinh thần
#: bagging — mỗi cây thấy một mẫu bootstrap khác nhau.
RF_MAX_SAMPLES = 0.25

#: Số cây Random Forest dựng song song. KHÔNG để -1: mỗi tiến trình con cấp
#: phát mảng trọng số dài bằng cả tập train, nên trên HI-Medium (19,6 triệu
#: dòng) 16 luồng cộng dồn làm MemoryError dù máy 32 GB. Kết quả không phụ
#: thuộc tham số này — random_state cố định cho ra đúng cùng các cây.
RF_N_JOBS = 4

#: Số mẫu mỗi cây của Isolation Forest. 256 là giá trị gốc trong bài báo
#: Liu et al. 2008: thuật toán cô lập điểm bất thường nhanh hơn khi mẫu NHỎ,
#: nên tăng con số này vừa chậm vừa kém.
IFOREST_MAX_SAMPLES = 256


# --------------------------------------------------------------------------
# Baseline cổ điển, dùng ĐÚNG đầu vào mà DRRE nhận
# --------------------------------------------------------------------------

def _causal_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Đặc trưng lịch sử tính NHÂN QUẢ — chỉ dùng giao dịch TRƯỚC đó của
    cùng tài khoản, nên không rò rỉ tương lai.

    Đây là baseline quan trọng nhất: nó cho thấy thông tin lịch sử CÓ giá
    trị hay không, độc lập với việc DRRE có khai thác được hay chưa.
    """
    time_col = "step" if "step" in df.columns else "timestamp"
    out = df.sort_values(time_col, kind="stable").copy()
    log_amount = np.log1p(out["amount"].astype(float))
    grouped = log_amount.groupby(out["account_id"], observed=True)

    n_prior = out.groupby("account_id", sort=False, observed=True).cumcount()
    mean_prior = grouped.transform(lambda s: s.shift(1).expanding().mean())
    std_prior = grouped.transform(lambda s: s.shift(1).expanding().std())

    t = (pd.to_datetime(out[time_col]).astype("int64") / 1e9
         if time_col == "timestamp" else out[time_col].astype(float))
    gap = t.groupby(out["account_id"], observed=True).diff()

    features = pd.DataFrame({
        "amount_norm": out["amount_norm"].astype(float),
        "channel_encoded": out["channel_encoded"].astype(float),
        "n_prior": n_prior.astype(float),
        "z_amount": ((log_amount - mean_prior) / std_prior.replace(0, np.nan)).fillna(0.0).clip(-5, 5),
        "gap_since_prev": np.log1p(gap.fillna(gap.median() if gap.notna().any() else 0.0).clip(lower=0)),
        "hour": (t % 86400) / 3600.0 if time_col == "timestamp" else out[time_col].astype(float) % 24,
    }, index=out.index)
    return features.reindex(df.index).fillna(0.0)


def _baseline_scores(
    df_eval: pd.DataFrame, df_train: pd.DataFrame, seed: int,
    train_mask: np.ndarray | None = None, source: str | None = None,
) -> dict[str, np.ndarray]:
    """Điểm của các baseline cổ điển, huấn luyện trên tập TRAIN rồi chấm
    tập đang đánh giá.

    Đây là cách so sánh công bằng duy nhất: DRRE học từ `*_train.parquet`
    và được chấm trên test, nên baseline phải làm đúng như vậy. Nếu huấn
    luyện baseline bằng kiểm định chéo trên chính tập test thì baseline
    được thấy một phần dữ liệu test — số liệu sẽ lạc quan hơn thực tế, và
    đó chính là lý do các con số GBDT đo bằng script rời trước đây
    (0,111 / 0,121) cao hơn số đo ở đây.
    """
    from sklearn.ensemble import (
        HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier,
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier

    y_train = df_train["label"].fillna(False).astype(int).to_numpy()
    base_cols = ["amount_norm", "channel_encoded"]
    # Baseline phải học trên ĐÚNG tập dòng mà DRRE tính mất mát (vũ trụ chấm
    # điểm), nhưng đặc trưng lịch sử vẫn tính trên CẢ dòng dữ liệu — giống
    # DRRE dùng mọi dòng để dựng lịch sử/đồ thị. Nếu lọc dòng trước khi tính
    # đặc trưng thì lịch sử tài khoản bị cắt vụn và so sánh thành không công bằng.
    fit_rows = (
        np.ones(len(df_train), dtype=bool) if train_mask is None
        else np.asarray(train_mask, dtype=bool)
    )

    rng = np.random.default_rng(seed)
    scores: dict[str, np.ndarray] = {
        "ngau_nhien": rng.random(len(df_eval)),
        "chi_so_tien": df_eval["amount"].astype(float).to_numpy(),
    }
    if y_train[fit_rows].sum() == 0:
        logger.warning("Tập train không có nhãn dương — bỏ qua baseline có học.")
        return scores

    feature_sets = {
        "cung_dau_vao": (
            df_train[base_cols].astype(float).to_numpy(),
            df_eval[base_cols].astype(float).to_numpy(),
        ),
        "them_lich_su": (
            _causal_history_features(df_train).to_numpy(),
            _causal_history_features(df_eval).to_numpy(),
        ),
    }

    # Bộ đặc trưng ĐÃ LƯU trong parquet — đúng những cột mà DRRE nhận cho
    # nguồn này (src.data.features.features_for_source). Đây là bộ so sánh
    # công bằng duy nhất, vì hai bộ ở trên thiệt thòi theo hai cách:
    #   - `cung_dau_vao` chỉ có 2 cột;
    #   - `them_lich_su` tự tính lại lịch sử TRONG TỪNG TẬP, nên mọi tài
    #     khoản đều bị coi như mới ở đầu tập; còn cột đã lưu được tính trên
    #     TOÀN TUYẾN thời gian trước khi chia tập, giống hệt cái DRRE thấy.
    # Không có bộ này thì mọi so sánh DRRE–baseline là so hai lượng thông
    # tin khác nhau, không phải so hai kiến trúc.
    stored: list[str] = []
    if source is not None:
        from src.data.features import features_for_source

        stored = [
            col for col in features_for_source(source)
            if col in df_train.columns and col in df_eval.columns
        ]
    if len(stored) > len(base_cols):
        feature_sets["dac_trung_day_du"] = (
            df_train[stored].astype(float).to_numpy(),
            df_eval[stored].astype(float).to_numpy(),
        )
        logger.info("Baseline dùng %d đặc trưng đã lưu: %s", len(stored), stored)

    x_tr, x_ev = feature_sets["cung_dau_vao"]
    scaler = StandardScaler().fit(x_tr[fit_rows])
    logistic = LogisticRegression(max_iter=1000, class_weight="balanced")
    logistic.fit(scaler.transform(x_tr[fit_rows]), y_train[fit_rows])
    scores["logistic_cung_dau_vao"] = logistic.predict_proba(scaler.transform(x_ev))[:, 1]

    for tag, (x_tr, x_ev) in feature_sets.items():
        clf = HistGradientBoostingClassifier(max_iter=200, random_state=seed)
        clf.fit(x_tr[fit_rows], y_train[fit_rows])
        scores[f"gbdt_{tag}"] = clf.predict_proba(x_ev)[:, 1]

    # Bộ đặc trưng giàu nhất đang có — dùng cho các mô hình thêm bên dưới.
    rich_tag = "dac_trung_day_du" if "dac_trung_day_du" in feature_sets else "them_lich_su"
    logger.info("Cây quyết định / Random Forest / Isolation Forest dùng bộ '%s'", rich_tag)

    # GBDT CÓ cân bằng lớp. Bản không cân bằng ở vòng lặp trên được giữ
    # nguyên để tái lập các con số cũ, nhưng nó bị thiệt: Decision Tree và
    # Random Forest bên dưới đều dùng class_weight="balanced", nên nếu chỉ có
    # bản không cân bằng thì phép so sánh giữa các họ cây là không công bằng.
    # Cặp gbdt_{rich} / gbdt_can_bang cũng chính là phép đo trực tiếp "xử lý
    # mất cân bằng đáng giá bao nhiêu" trên cùng một mô hình, cùng một bộ đặc
    # trưng, chỉ khác đúng một tham số.
    x_tr, x_ev = feature_sets[rich_tag]
    gbdt_balanced = HistGradientBoostingClassifier(
        max_iter=200, class_weight="balanced", random_state=seed)
    gbdt_balanced.fit(x_tr[fit_rows], y_train[fit_rows])
    scores["gbdt_can_bang"] = gbdt_balanced.predict_proba(x_ev)[:, 1]

    # ------------------------------------------------------------------
    # Cây quyết định đơn và Random Forest
    # ------------------------------------------------------------------
    # Hai mô hình này KHÁC GBDT về bản chất dù cùng dùng cây: GBDT là
    # boosting (mỗi cây sửa sai cho cây trước, cây nông), Random Forest là
    # bagging (các cây độc lập trên mẫu bootstrap, cây sâu). Có cả hai mới
    # trả lời được "nền tảng phân loại rủi ro cổ điển làm được tới đâu".
    # Cây đơn còn có giá trị riêng: nó ĐỌC ĐƯỢC — in ra là thấy luật.
    tree = DecisionTreeClassifier(
        max_depth=8, min_samples_leaf=50, class_weight="balanced", random_state=seed)
    tree.fit(x_tr[fit_rows], y_train[fit_rows])
    scores["cay_quyet_dinh"] = tree.predict_proba(x_ev)[:, 1]

    forest = RandomForestClassifier(
        n_estimators=100, max_depth=12, min_samples_leaf=50,
        max_samples=RF_MAX_SAMPLES, class_weight="balanced_subsample",
        n_jobs=RF_N_JOBS, random_state=seed)
    forest.fit(x_tr[fit_rows], y_train[fit_rows])
    scores["random_forest"] = forest.predict_proba(x_ev)[:, 1]

    # ------------------------------------------------------------------
    # Isolation Forest — mốc KHÔNG dùng nhãn
    # ------------------------------------------------------------------
    # Đây là mốc quan trọng nhất trong nhóm này, vì nó không nhìn nhãn một
    # lần nào. Điểm nghẽn đã đo được của dự án là SỐ NHÃN DƯƠNG quá ít
    # (PaySim 1.687, AMLWorld 2.856) và chính nó gây ra bất ổn theo seed tới
    # 13,3 lần. Một phương pháp không cần nhãn thì miễn nhiễm với điểm nghẽn
    # đó — nếu nó ổn định trong khi DRRE dao động, bản thân điều đó đã là
    # một kết luận đáng báo cáo.
    # score_samples trả về càng CAO càng bình thường, nên phải đảo dấu để
    # cùng quy ước "điểm cao = đáng ngờ" với mọi phương pháp khác.
    iforest = IsolationForest(
        n_estimators=100, max_samples=IFOREST_MAX_SAMPLES,
        random_state=seed, n_jobs=-1)
    iforest.fit(x_tr[fit_rows])
    scores["isolation_forest"] = -iforest.score_samples(x_ev)

    # ------------------------------------------------------------------
    # Bảng tra theo hình thức thanh toán
    # ------------------------------------------------------------------
    # Mốc một cột, cố tình thô sơ. Trên AMLWorld, `ach` chứa 86,6% nhãn
    # dương trong 11,8% số dòng (tỷ lệ cao hơn các hình thức khác 20-48
    # lần). Nếu mô hình ba nhánh 56.201 tham số không vượt được một bảng tra
    # 7 dòng thì đó là thông tin quan trọng nhất của cả đợt đánh giá — và
    # không có mốc này thì không ai phát hiện ra.
    if "channel_encoded" in df_train.columns and "channel_encoded" in df_eval.columns:
        rate_by_channel = (
            pd.DataFrame({
                "channel": df_train["channel_encoded"].to_numpy()[fit_rows],
                "y": y_train[fit_rows],
            })
            .groupby("channel")["y"].mean()
        )
        scores["bang_tra_kenh"] = (
            df_eval["channel_encoded"].map(rate_by_channel).fillna(0.0).to_numpy(dtype=float)
        )

    return scores


# --------------------------------------------------------------------------
# Chỉ số cho một bộ điểm
# --------------------------------------------------------------------------

def metrics_for(
    y: np.ndarray, scores: np.ndarray, budgets: tuple[int, ...] = DEFAULT_ALERT_BUDGETS
) -> dict[str, float]:
    from sklearn.metrics import roc_auc_score

    scores = np.asarray(scores, dtype=float)
    out: dict[str, float] = {
        "auprc": pr_auc(y, scores),
        "roc_auc": float(roc_auc_score(y, scores)),
        "ece": expected_calibration_error(
            y, np.clip(scores / 100.0 if scores.max() > 1.0 else scores, 0.0, 1.0)),
    }
    for budget in budgets:
        out[f"recall@{budget}"] = recall_at_budget(y, scores, budget)
        out[f"fpr@{budget}"] = false_positive_rate_at_budget(y, scores, budget)
        top = np.argsort(-scores, kind="stable")[:budget]
        out[f"caught@{budget}"] = float(y[top].sum())
    return out


# --------------------------------------------------------------------------
# Chỉ số theo từng giai đoạn thời gian
# --------------------------------------------------------------------------

def period_bounds(n_rows: int, n_periods: int) -> list[tuple[int, int]]:
    """Chia [0, n_rows) thành n_periods khoảng LIÊN TIẾP gần bằng nhau về SỐ DÒNG.

    Chia theo số dòng chứ không theo độ rộng thời gian là có chủ ý: PaySim có
    hai chế độ rất khác nhau (lượng giao dịch/step sụp giảm sau step ~400 trong
    khi tỷ lệ gian lận giữ nguyên), nên chia đều theo thời gian sẽ dồn gần hết
    dữ liệu vào giai đoạn đầu và các giai đoạn sau không còn đủ mẫu để đo.
    """
    if n_periods < 1:
        raise ValueError(f"n_periods phải >= 1, nhận {n_periods}")
    edges = [round(i * n_rows / n_periods) for i in range(n_periods + 1)]
    return [(lo, hi) for lo, hi in zip(edges, edges[1:]) if hi > lo]


def metrics_by_period(
    y: np.ndarray,
    score_map: dict[str, np.ndarray],
    n_periods: int,
    time_values: np.ndarray | None = None,
) -> list[dict]:
    """Chỉ số của từng phương pháp trên từng giai đoạn thời gian liên tiếp.

    QUAN TRỌNG khi đọc kết quả: AUPRC thô KHÔNG so sánh được giữa các giai
    đoạn, vì mốc ngẫu nhiên của mỗi giai đoạn là tỷ lệ nhãn dương của chính
    nó. Giai đoạn nào nhiều gian lận hơn sẽ có AUPRC cao hơn dù model không
    hề tốt hơn. Vì vậy hàm này trả về cả ``lift = AUPRC / tỷ lệ nhãn dương``
    — đó mới là con số so sánh được giữa các giai đoạn.

    Giai đoạn không có nhãn dương (hoặc toàn nhãn dương) thì AUPRC/ROC-AUC
    không định nghĩa được: trả về None kèm ``skipped`` thay vì một con số giả.
    """
    from sklearn.metrics import roc_auc_score

    out: list[dict] = []
    for idx, (lo, hi) in enumerate(period_bounds(len(y), n_periods), start=1):
        y_p = y[lo:hi]
        n_pos = int(y_p.sum())
        rec: dict = {
            "period": idx, "row_from": lo, "row_to": hi, "n_rows": hi - lo,
            "n_pos": n_pos, "pos_rate": float(y_p.mean()),
        }
        if time_values is not None:
            rec["time_from"] = _as_plain(time_values[lo])
            rec["time_to"] = _as_plain(time_values[hi - 1])
        if n_pos == 0 or n_pos == len(y_p):
            rec["skipped"] = ("giai đoạn không có nhãn dương" if n_pos == 0
                              else "giai đoạn toàn nhãn dương")
            rec["methods"] = {}
            out.append(rec)
            continue
        rec["methods"] = {
            name: {
                "auprc": pr_auc(y_p, np.asarray(scores, dtype=float)[lo:hi]),
                "lift": pr_auc(y_p, np.asarray(scores, dtype=float)[lo:hi]) / float(y_p.mean()),
                "roc_auc": float(roc_auc_score(y_p, np.asarray(scores, dtype=float)[lo:hi])),
            }
            for name, scores in score_map.items()
        }
        out.append(rec)
    return out


def _as_plain(value):
    """Giá trị thời gian về dạng JSON ghi được (np.int64/Timestamp -> int/str)."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.datetime64, pd.Timestamp)):
        # Cắt tới phút: dấu thời gian nano giây làm bảng rộng gấp ba mà không
        # thêm thông tin nào cho việc đọc xu hướng theo giai đoạn.
        return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M")
    return str(value)


def _by_period_table(periods: list[dict], method_names: list[str]) -> str:
    """Bảng markdown: mỗi dòng một giai đoạn, ô ghi ``AUPRC (×lift)``."""
    header = ["Giai đoạn", "Thời gian", "Dòng", "Gian lận", "Mốc ngẫu nhiên"] + method_names
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for rec in periods:
        span = (f"{rec['time_from']}–{rec['time_to']}"
                if "time_from" in rec else f"{rec['row_from']}–{rec['row_to']}")
        cells = [str(rec["period"]), span, f"{rec['n_rows']:,}", f"{rec['n_pos']:,}",
                 f"{rec['pos_rate']:.5f}"]
        for name in method_names:
            m = rec["methods"].get(name)
            cells.append(f"{m['auprc']:.4f} (×{m['lift']:.0f})" if m else
                         rec.get("skipped", "—"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Recall theo từng kiểu hình rửa tiền
# --------------------------------------------------------------------------

def recall_by_typology(
    patterns: np.ndarray,
    y: np.ndarray,
    score_map: dict[str, np.ndarray],
    budget: int,
) -> list[dict]:
    """Trong `budget` giao dịch nghi nhất, mỗi KIỂU HÌNH rửa tiền bắt được bao nhiêu.

    Vì sao phép đo này quan trọng hơn một con số AUPRC tổng: tám kiểu hình
    của AMLWorld (CYCLE, FAN-OUT, GATHER-SCATTER, STACK...) đều được định
    nghĩa bằng HÌNH DẠNG CỦA MẠNG LƯỚI, không phải bằng thuộc tính của một
    giao dịch đơn lẻ. Một mô hình chỉ nhìn từng giao dịch về nguyên tắc
    không thể phân biệt được chúng, còn một mô hình có nhánh đồ thị thì có
    thể. Nên bảng này là bằng chứng TRỰC TIẾP cho câu hỏi "nhánh đồ thị có
    đóng góp gì không" — và nó bền với nhiễu seed hơn AUPRC, vì nó so sánh
    thứ hạng tương đối giữa các kiểu hình chứ không so một con số tuyệt đối.

    Args:
        patterns: nhãn kiểu hình của từng dòng (NaN cho dòng không phải rửa
            tiền). Lấy từ cột `laundering_pattern`, xem
            `src.data.loaders.attach_laundering_pattern`.
        y: nhãn nhị phân, cùng độ dài và cùng thứ tự với `patterns`.
        score_map: {tên phương pháp: điểm}, cùng độ dài.
        budget: số cảnh báo được xét (top-N theo điểm).

    Returns:
        Một bản ghi cho mỗi kiểu hình, kèm `n_pos` và `methods`
        {tên: {caught, recall}}. Kiểu hình không có nhãn dương nào bị bỏ qua.
    """
    top_sets = {
        name: set(np.argsort(-np.asarray(s, dtype=float), kind="stable")[:budget].tolist())
        for name, s in score_map.items()
    }
    out: list[dict] = []
    is_pos = y.astype(bool)
    labels = pd.Series(patterns)[is_pos]
    for kind, idx in labels.groupby(labels, dropna=True).groups.items():
        positions = np.asarray(idx, dtype=int)
        rec: dict = {"typology": str(kind), "n_pos": int(len(positions)), "methods": {}}
        for name, top in top_sets.items():
            caught = sum(1 for p in positions.tolist() if p in top)
            rec["methods"][name] = {
                "caught": caught,
                "recall": caught / len(positions) if len(positions) else 0.0,
            }
        out.append(rec)
    return sorted(out, key=lambda r: -r["n_pos"])


def _by_typology_table(records: list[dict], method_names: list[str], budget: int) -> str:
    """Bảng markdown: mỗi dòng một kiểu hình, ô ghi ``số bắt được (recall%)``."""
    header = [f"Kiểu hình (top {budget:,})", "Nhãn dương"] + method_names
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for rec in records:
        cells = [rec["typology"], f"{rec['n_pos']:,}"]
        for name in method_names:
            m = rec["methods"].get(name)
            cells.append(f"{m['caught']} ({100 * m['recall']:.0f}%)" if m else "—")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Chấm DRRE có mang trạng thái Behavior Memory
# --------------------------------------------------------------------------

#: Tập nào phải quét trước để nạp trạng thái. `drift` nằm sau cả test theo thời
#: gian nhưng KHÔNG được quét qua test: tập test chỉ được chạm một lần.
WARM_UP_SPLITS = {"val": ["train"], "test": ["train", "val"], "drift": ["train", "val"]}


def _carries_state(checkpoint: str) -> bool:
    import torch

    if not Path(checkpoint).exists():
        return False
    return bool(torch.load(checkpoint, map_location="cpu", weights_only=True).get("carry_state", False))


def _score_stateful(
    df: pd.DataFrame, checkpoint: str, args: argparse.Namespace, time_col: str
) -> pd.DataFrame:
    """Chấm `df` sau khi quét các tập trước đó, đúng như lúc huấn luyện và chấm val.

    Xem src/evaluation/stateful_scoring.py. Trả về khung cùng các cột mà phần
    sau của main() đọc từ score_batch.
    """
    from src.dashboard.scoring_service import try_load_drre_checkpoint
    from src.drre.train import BehaviorStateStore
    from src.evaluation.stateful_scoring import (
        fingerprint, required_capacity, score_split, stream_columns, time_sorted, warm_up_store,
    )

    loaded = try_load_drre_checkpoint(checkpoint)
    if loaded is None:
        raise SystemExit(f"Không nạp được checkpoint {checkpoint} để chấm có mang trạng thái.")
    model, risk_engine = loaded
    paths = [Path(args.processed_dir) / f"{args.source}_{name}.parquet"
             for name in WARM_UP_SPLITS[args.split]]
    columns = stream_columns(model, time_col)
    stream = time_sorted(pd.concat(
        [pd.read_parquet(path, columns=columns) for path in paths], ignore_index=True))

    memory = model.behavior_memory
    store = BehaviorStateStore(
        memory.hidden_dim, memory.num_slots, capacity=required_capacity([stream, df]))
    cache = Path(checkpoint).with_suffix(f".warm_{args.split}.pt")
    window = args.warm_window
    warm_up_store(model, risk_engine, stream, store, cache,
                  fingerprint(checkpoint, paths, window), window=window)
    del stream
    logger.info("Làm nóng xong: gặp lại %.1f%% tài khoản, %d lần đẩy khỏi bộ đệm.",
                100 * store.hit_rate, store.evictions)
    return pd.DataFrame({
        "risk_score": score_split(model, risk_engine, df, store, window=window),
        "label": df["label"].to_numpy(),
        "method": "drre_trained_mang_trang_thai",
    })


# --------------------------------------------------------------------------
# Ma trận nhầm lẫn và theo kênh thanh toán
# --------------------------------------------------------------------------

BEST_F1_KEY = "f1_tot_nhat"
VAL_THRESHOLD_KEY = "nguong_tu_val"


def confusion_report(
    y: np.ndarray,
    score_map: dict[str, np.ndarray],
    budgets: tuple[int, ...],
    split: str,
    val_thresholds: dict[str, float] | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    """Ma trận nhầm lẫn của mỗi phương pháp ở từng điểm vận hành.

    Ngưỡng F1 tốt nhất chỉ được CHỌN trên val. Trên test, ngưỡng lấy nguyên từ
    báo cáo val (`--thresholds-from`); tự chọn trên test là dùng đáp án để đo.
    """
    report: dict[str, dict[str, dict[str, float]]] = {}
    for name, scores in score_map.items():
        row = {f"top{budget}": confusion_at_budget(y, scores, budget) for budget in budgets}
        if val_thresholds and name in val_thresholds:
            row[VAL_THRESHOLD_KEY] = confusion_at_threshold(y, scores, val_thresholds[name])
        elif split != "test":
            row[BEST_F1_KEY] = confusion_at_threshold(y, scores, best_f1_threshold(y, scores))
        report[name] = row
    return report


def _operating_threshold(row: dict[str, dict[str, float]]) -> float | None:
    chosen = row.get(VAL_THRESHOLD_KEY) or row.get(BEST_F1_KEY)
    return None if chosen is None else float(chosen["threshold"])


def metrics_by_channel(
    channels: np.ndarray,
    y: np.ndarray,
    score_map: dict[str, np.ndarray],
    confusion: dict[str, dict[str, dict[str, float]]],
    budget: int,
) -> list[dict]:
    """Mỗi kênh: AUPRC trong kênh, số cảnh báo chiếm trong ngân sách chung, số vụ bắt được."""
    channels = np.asarray(channels, dtype=object)
    positive = np.asarray(y, dtype=bool)
    records: list[dict] = []
    for channel in sorted(set(channels.tolist())):
        mask = channels == channel
        n_pos = int(positive[mask].sum())
        record: dict = {"channel": channel, "n": int(mask.sum()), "n_pos": n_pos, "methods": {}}
        for name, scores in score_map.items():
            top = np.zeros(len(scores), dtype=bool)
            top[np.argsort(-scores, kind="stable")[:budget]] = True
            entry = {"in_top": int(top[mask].sum()), "tp_top": int((top & mask & positive).sum())}
            if n_pos:
                entry["auprc"] = pr_auc(y[mask], scores[mask])
            threshold = _operating_threshold(confusion[name])
            if threshold is not None:
                entry["tp_nguong"] = int(((scores >= threshold) & mask & positive).sum())
                entry["fp_nguong"] = int(((scores >= threshold) & mask & ~positive).sum())
            record["methods"][name] = entry
        records.append(record)
    return records


def _confusion_table(report: dict[str, dict[str, dict[str, float]]]) -> str:
    lines = ["| Phương pháp | Điểm vận hành | TP | FP | FN | TN | Precision | Recall | F1 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, row in report.items():
        for point, c in row.items():
            lines.append(f"| {name} | {point} | {c['tp']:,} | {c['fp']:,} | {c['fn']:,} | "
                         f"{c['tn']:,} | {c['precision']:.3f} | {c['recall']:.3f} | {c['f1']:.3f} |")
    return "\n".join(lines)


def _by_channel_table(records: list[dict], names: list[str], budget: int) -> str:
    head = "".join(f" {n}: AUPRC / trong top {budget:,} / bắt được ở ngưỡng |" for n in names)
    lines = [f"| Kênh | Giao dịch | Nhãn dương |{head}", "|---|---|---|" + "---|" * len(names)]
    for r in records:
        cells = []
        for name in names:
            m = r["methods"][name]
            auprc = f"{m['auprc']:.4f}" if "auprc" in m else "—"
            cells.append(f" {auprc} / {m['in_top']:,} / {m.get('tp_nguong', '—')} |")
        lines.append(f"| {r['channel']} | {r['n']:,} | {r['n_pos']:,} |" + "".join(cells))
    return "\n".join(lines)


def _markdown_table(rows: dict[str, dict[str, float]], budgets: tuple[int, ...]) -> str:
    header = ["Phương pháp", "AUPRC", "ROC-AUC"] + [f"Bắt được / {b}" for b in budgets]
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for name, m in sorted(rows.items(), key=lambda kv: kv[1]["auprc"]):
        cells = [name, f"{m['auprc']:.4f}", f"{m['roc_auc']:.3f}"]
        cells += [f"{int(m[f'caught@{b}'])}" for b in budgets]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Đánh giá checkpoint DRRE trên tập test")
    parser.add_argument(
        "--source", required=True,
        choices=["paysim", "ieee_cis", "synthetic_vn", "amlworld", "transxion"])
    parser.add_argument("--split", default="test", choices=["val", "test", "drift"])
    parser.add_argument(
        "--universe", default="clean", choices=["clean", "all"],
        help="'clean' (mặc định) chỉ tính chỉ số trên các dòng có eval_universe=True — "
             "với PaySim là giờ 9-23 và loại transfer/cash_out, vì một bảng tra theo giờ "
             "đạt AUPRC 0,2949 trên vũ trụ đầy đủ (gấp đôi DRRE) nên số đo ở đó vô nghĩa. "
             "'all' tính trên mọi dòng — dùng để tái lập các con số cũ, KHÔNG để báo cáo.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--compare-checkpoint", default=None,
                        help="Checkpoint thứ hai để so sánh (vd. bản ablation), kèm "
                             "khoảng tin cậy bootstrap cho hiệu số AUPRC")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--budgets", default=",".join(str(b) for b in DEFAULT_ALERT_BUDGETS))
    parser.add_argument("--bootstrap", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-baselines", action="store_true",
                        help="Bỏ qua baseline cổ điển (nhanh hơn, nhưng mất mốc so sánh)")
    parser.add_argument("--compare-baseline", default=None, metavar="TEN",
                        help="Tên MỘT baseline để so với checkpoint chính, kèm khoảng tin "
                             "cậy bootstrap ghép cặp cho hiệu số AUPRC. Đây là phép đo trả "
                             "lời câu hỏi 'DRRE có hơn phương pháp thông thường không' — "
                             "chênh lệch AUPRC thô giữa hai dòng trong bảng KHÔNG trả lời "
                             "được, vì nó không cho biết khoảng cách đó có vượt nhiễu lấy "
                             "mẫu hay không. Tên lấy đúng như cột 'Phương pháp', vd. "
                             "gbdt_them_lich_su. Không dùng cùng --no-baselines.")
    parser.add_argument("--score-batch-size", type=int, default=8192,
                        help="Chấm điểm theo lô bấy nhiêu dòng. Bắt buộc với tập lớn: "
                             "chấm 450.000 dòng PaySim trong một lô làm tiến trình chết "
                             "vì hết bộ nhớ. Đặt 0 để chấm cả tập trong một lô.")
    parser.add_argument("--by-period", type=int, default=0, metavar="N",
                        help="Ngoài chỉ số trên cả tập, đo thêm trên N giai đoạn thời gian "
                             "liên tiếp (chia đều theo SỐ DÒNG). Dùng để xem hiệu quả có "
                             "giữ được qua thời gian hay không. Mỗi giai đoạn có mốc ngẫu "
                             "nhiên riêng nên bảng ghi kèm lift = AUPRC / tỷ lệ nhãn dương "
                             "— chỉ lift mới so sánh được giữa các giai đoạn. 0 = tắt.")
    parser.add_argument(
        "--by-typology", action="store_true",
        help="Thêm bảng recall theo từng KIỂU HÌNH rửa tiền (CYCLE, FAN-OUT, "
             "GATHER-SCATTER...) ở ngân sách cảnh báo lớn nhất. Cần cột "
             "`laundering_pattern` — tiền xử lý với --patterns-path. Đây là bằng "
             "chứng trực tiếp nhất cho đóng góp của nhánh đồ thị, vì tám kiểu hình "
             "đều được định nghĩa bằng hình dạng mạng lưới chứ không phải bằng "
             "thuộc tính của một giao dịch đơn lẻ.")
    parser.add_argument(
        "--stateful", default="auto", choices=["auto", "on", "off"],
        help="Chấm DRRE có mang trạng thái Behavior Memory từ các tập trước đó. 'auto' bật "
             "cho checkpoint huấn luyện với --carry-state — chấm các checkpoint đó bằng bộ "
             "nhớ trống là đo một mô hình khác với mô hình đã học. 'off' chỉ để tái lập số cũ.")
    parser.add_argument("--warm-window", type=int, default=32768,
                        help="Cửa sổ khi quét và chấm có mang trạng thái — đúng kích thước đã "
                             "dùng khi huấn luyện.")
    parser.add_argument("--thresholds-from", default=None, metavar="BAO_CAO_VAL",
                        help="Báo cáo evaluate trên val để lấy ngưỡng F1 tốt nhất. Bắt buộc "
                             "dùng khi đo test nếu muốn ma trận nhầm lẫn ở ngưỡng đã chốt.")
    parser.add_argument("--keep-awake", action="store_true",
                        help="Xin Windows không ngủ khi rảnh trong lúc chạy.")
    parser.add_argument("--out-dir", default="reports/eval")
    args = parser.parse_args(argv)

    if args.keep_awake and sys.platform == "win32":
        import ctypes

        # ES_CONTINUOUS | ES_SYSTEM_REQUIRED: tự hết khi tiến trình thoát.
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)

    if args.compare_baseline and args.no_baselines:
        parser.error("--compare-baseline cần baseline được tính, nên không dùng cùng --no-baselines")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    budgets = tuple(int(b) for b in args.budgets.split(","))
    started = time.monotonic()

    from src.dashboard.scoring_service import score_batch

    path = Path(args.processed_dir) / f"{args.source}_{args.split}.parquet"
    if not path.exists():
        parser.error(f"Không tìm thấy {path} — chạy src.data.preprocessing trước.")
    df = pd.read_parquet(path)
    # Chuẩn hóa thứ tự theo thời gian NGAY khi nạp. Bắt buộc: chấm theo lô
    # sẽ tự sắp lại theo thời gian, nên nếu thứ tự ở đây khác thì điểm và
    # nhãn lệch nhau mà không có lỗi nào được báo.
    _time_col = "step" if "step" in df.columns else "timestamp"
    df = df.sort_values(_time_col, kind="stable").reset_index(drop=True)
    y_all = df["label"].fillna(False).astype(int).to_numpy()

    # Vũ trụ chấm điểm: điểm vẫn được tính cho MỌI dòng (đồ thị và lịch sử cần
    # chúng), nhưng chỉ số chỉ tính trên các dòng trong vũ trụ — đúng như mất
    # mát BCE lúc huấn luyện. Xem src/data/preprocessing.py::add_eval_universe.
    keep = np.ones(len(df), dtype=bool)
    if args.universe == "clean":
        if "eval_universe" in df.columns:
            keep = df["eval_universe"].fillna(True).to_numpy(dtype=bool)
        else:
            logger.warning(
                "%s không có cột eval_universe (tiền xử lý trước 2026-10-17) — đang tính "
                "trên MỌI dòng. Chạy lại src.data.preprocessing để có vũ trụ chấm điểm.", path)
    y = y_all[keep]
    if y.sum() == 0:
        parser.error(
            f"{path} không có nhãn dương nào trong vũ trụ '{args.universe}' — không đánh giá được.")

    logger.info("%s %s: %d dòng, %d nhãn dương (tỷ lệ %.4f%%)",
                args.source, args.split, len(df), int(y_all.sum()), 100 * y_all.mean())
    if not keep.all():
        logger.info(
            "Vũ trụ chấm điểm '%s': %d/%d dòng (%.1f%%), %d/%d nhãn dương, tỷ lệ nền %.5f "
            "— mọi chỉ số dưới đây tính trên tập này.",
            args.universe, int(keep.sum()), len(df), 100 * keep.mean(),
            int(y.sum()), int(y_all.sum()), y.mean())
    if args.split == "test":
        logger.warning(
            "Đang chấm trên tập TEST. Chỉ nên chạm MỘT LẦN sau khi đã chốt model bằng val — "
            "chạy lại nhiều lần để thử cấu hình là đang dùng test làm val.")

    results: dict[str, dict[str, float]] = {}
    score_map: dict[str, np.ndarray] = {}

    chunk = args.score_batch_size or None

    def score_checkpoint(checkpoint: str) -> pd.DataFrame:
        if args.stateful == "on" or (args.stateful == "auto" and _carries_state(checkpoint)):
            return _score_stateful(df, checkpoint, args, _time_col)
        return score_batch(df, checkpoint, batch_size=chunk)

    scored = score_checkpoint(args.checkpoint)
    main_scores = scored["risk_score"].to_numpy(dtype=float)
    # Chặn lệch điểm–nhãn: khung đã chấm phải giữ đúng thứ tự nhãn ban đầu.
    if "label" in scored.columns:
        scored_labels = scored["label"].fillna(False).astype(int).to_numpy()
        if not np.array_equal(scored_labels, y_all):
            parser.error(
                "Thứ tự dòng sau khi chấm điểm KHÁC thứ tự nhãn — mọi chỉ số sẽ sai. "
                "Kiểm tra lại cột thời gian của nguồn dữ liệu này.")
    method = str(scored["method"].iloc[0])
    label = f"{Path(args.checkpoint).name} ({method})"
    results[label] = metrics_for(y, main_scores[keep], budgets)
    score_map[label] = main_scores
    if not method.startswith("drre_trained"):
        logger.warning(
            "Checkpoint không nạp được nên đã rơi về quy tắc thống kê (method=%s). "
            "Số liệu dưới đây KHÔNG phải của DRRE.", method)

    compare_scores = None
    if args.compare_checkpoint:
        scored_b = score_checkpoint(args.compare_checkpoint)
        compare_scores = scored_b["risk_score"].to_numpy(dtype=float)
        compare_label = f"{Path(args.compare_checkpoint).name} ({scored_b['method'].iloc[0]})"
        results[compare_label] = metrics_for(y, compare_scores[keep], budgets)
        score_map[compare_label] = compare_scores

    if not args.no_baselines:
        train_path = Path(args.processed_dir) / f"{args.source}_train.parquet"
        if not train_path.exists():
            logger.warning(
                "Không tìm thấy %s — bỏ qua baseline có học. Baseline phải học từ tập "
                "train để so sánh công bằng với DRRE.", train_path)
        else:
            logger.info("Đang chạy baseline cổ điển (học từ %s, chấm trên %s)...",
                        train_path.name, path.name)
            df_train = pd.read_parquet(train_path)
            train_mask = None
            if args.universe == "clean" and "eval_universe" in df_train.columns:
                train_mask = df_train["eval_universe"].fillna(True).to_numpy(dtype=bool)
            for name, scores in _baseline_scores(
                    df, df_train, args.seed, train_mask, source=args.source).items():
                results[name] = metrics_for(y, scores[keep], budgets)
                score_map[name] = scores

    baseline_bootstrap = None
    if args.compare_baseline:
        if args.compare_baseline not in score_map:
            parser.error(
                f"Không có phương pháp tên {args.compare_baseline!r}. Các tên dùng được: "
                + ", ".join(sorted(score_map)))
        diff, lo, hi = bootstrap_metric_diff(
            y, main_scores[keep], score_map[args.compare_baseline][keep], pr_auc,
            n_boot=args.bootstrap, seed=args.seed)
        baseline_bootstrap = {
            "baseline": args.compare_baseline, "diff_auprc": diff,
            "ci95": [lo, hi], "n_boot": args.bootstrap,
        }
        verdict = ("khác biệt có ý nghĩa thống kê" if (lo > 0 or hi < 0)
                   else "KHÔNG kết luận được (khoảng tin cậy chứa 0)")
        logger.info("Hiệu số AUPRC (%s − %s) = %+.4f, KTC 95%% [%+.4f; %+.4f] — %s",
                    Path(args.checkpoint).name, args.compare_baseline, diff, lo, hi, verdict)

    bootstrap_result = None
    if compare_scores is not None:
        diff, lo, hi = bootstrap_metric_diff(
            y, main_scores[keep], compare_scores[keep], pr_auc,
            n_boot=args.bootstrap, seed=args.seed)
        bootstrap_result = {"diff_auprc": diff, "ci95": [lo, hi], "n_boot": args.bootstrap}
        verdict = ("khác biệt có ý nghĩa thống kê" if (lo > 0 or hi < 0)
                   else "KHÔNG kết luận được (khoảng tin cậy chứa 0)")
        logger.info("Hiệu số AUPRC (%s − %s) = %+.4f, KTC 95%% [%+.4f; %+.4f] — %s",
                    Path(args.checkpoint).name, Path(args.compare_checkpoint).name,
                    diff, lo, hi, verdict)

    table = _markdown_table(results, budgets)
    print("\n" + table + "\n")
    print(f"Mốc ngẫu nhiên (tỷ lệ nhãn dương): AUPRC = {y.mean():.4f}\n")

    periods = None
    period_table = None
    if args.by_period:
        # Thứ tự dòng của df đã được sắp theo thời gian ở trên và đã kiểm tra
        # khớp nhãn, nên cắt theo VỊ TRÍ ở đây chính là cắt theo thời gian.
        periods = metrics_by_period(
            y, {name: s[keep] for name, s in score_map.items()}, args.by_period,
            time_values=df.loc[keep, _time_col].to_numpy())
        method_names = [name for name, _ in sorted(results.items(), key=lambda kv: kv[1]["auprc"])]
        period_table = _by_period_table(periods, method_names)
        print(f"Theo {len(periods)} giai đoạn thời gian — ô ghi AUPRC (×lift so với mốc "
              f"ngẫu nhiên CỦA GIAI ĐOẠN đó):")
        print("\n" + period_table + "\n")
        thin = [r["period"] for r in periods if r.get("n_pos", 0) < 30]
        if thin:
            print(f"CẢNH BÁO: giai đoạn {thin} có dưới 30 nhãn dương — AUPRC ở đó nhiễu "
                  f"rất mạnh, KHÔNG dùng để kết luận xu hướng.\n")

    typology = None
    typology_table = None
    if args.by_typology:
        if "laundering_pattern" not in df.columns:
            logger.warning(
                "%s không có cột `laundering_pattern` — bỏ qua bảng theo kiểu hình. "
                "Chạy lại tiền xử lý với --patterns-path <bộ>_Patterns.txt.", path)
        else:
            budget = max(budgets)
            typology = recall_by_typology(
                df.loc[keep, "laundering_pattern"].astype(object).to_numpy(),
                y, {name: s[keep] for name, s in score_map.items()}, budget)
            method_names = [
                name for name, _ in sorted(results.items(), key=lambda kv: -kv[1]["auprc"])]
            typology_table = _by_typology_table(typology, method_names, budget)
            print(f"Recall theo từng kiểu hình rửa tiền, trong {budget:,} giao dịch nghi nhất:")
            print("\n" + typology_table + "\n")
            thin_kinds = [r["typology"] for r in typology if r["n_pos"] < 20]
            if thin_kinds:
                print(f"CẢNH BÁO: kiểu hình {thin_kinds} có dưới 20 nhãn dương — recall ở "
                      f"đó nhiễu mạnh, KHÔNG dùng để xếp hạng kiểu hình.\n")

    val_thresholds = None
    if args.thresholds_from:
        with open(args.thresholds_from, encoding="utf-8") as handle:
            saved = json.load(handle)["results"]["confusion"]
        val_thresholds = {name: row[BEST_F1_KEY]["threshold"] for name, row in saved.items()
                          if BEST_F1_KEY in row}
    elif args.split == "test":
        logger.warning("Đo test mà không có --thresholds-from: chỉ có ma trận nhầm lẫn theo ngân "
                       "sách cảnh báo, không chọn ngưỡng trên test.")
    kept_scores = {name: s[keep] for name, s in score_map.items()}
    confusion = confusion_report(y, kept_scores, budgets, args.split, val_thresholds)
    confusion_table = _confusion_table(confusion)
    print("Ma trận nhầm lẫn ở từng điểm vận hành:")
    print("\n" + confusion_table + "\n")

    by_channel = None
    by_channel_table = None
    if "channel" in df.columns:
        by_channel = metrics_by_channel(
            df.loc[keep, "channel"].astype(object).to_numpy(), y, kept_scores, confusion, max(budgets))
        shown = [label] + ([args.compare_baseline] if args.compare_baseline else [])
        by_channel_table = _by_channel_table(by_channel, shown, max(budgets))
        print("Theo kênh thanh toán:")
        print("\n" + by_channel_table + "\n")

    manifest = RunManifest(
        kind=f"evaluate_{args.source}_{args.split}",
        seed=args.seed,
        hyperparameters={
            "checkpoint": args.checkpoint,
            "compare_checkpoint": args.compare_checkpoint,
            "budgets": list(budgets),
            "bootstrap": args.bootstrap,
            "score_batch_size": chunk,
            "baselines": not args.no_baselines,
            "compare_baseline": args.compare_baseline,
            "by_period": args.by_period,
            "by_typology": args.by_typology,
            "universe": args.universe,
            "stateful": args.stateful,
            "warm_window": args.warm_window,
            "thresholds_from": args.thresholds_from,
            "scored_rows": int(keep.sum()),
            "total_rows": len(df),
        },
        data={"eval_file": {**file_fingerprint(path), **dataframe_summary(df)}},
        results={
            "random_baseline_auprc": float(y.mean()),
            "metrics": results,
            "bootstrap": bootstrap_result,
            "bootstrap_vs_baseline": baseline_bootstrap,
            "markdown_table": table,
            "by_period": periods,
            "by_period_markdown_table": period_table,
            "by_typology": typology,
            "by_typology_markdown_table": typology_table,
            "confusion": confusion,
            "confusion_markdown_table": confusion_table,
            "by_channel": by_channel,
            "by_channel_markdown_table": by_channel_table,
        },
    )
    manifest.finish(started)
    out_path = manifest.write(args.out_dir)
    logger.info("Đã ghi kết quả: %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
