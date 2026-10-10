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
from src.tracking import RunManifest, dataframe_summary, file_fingerprint  # noqa: E402

logger = logging.getLogger("evaluate")

#: Số giao dịch nghi nhất được "cảnh báo" — khớp với ràng buộc năng lực
#: điều tra trong QAPE. Đây là con số dễ hiểu nhất cho người không chuyên:
#: "trong N giao dịch nghi nhất, bắt được bao nhiêu vụ".
DEFAULT_ALERT_BUDGETS = (500, 1000, 2000)


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
    grouped = log_amount.groupby(out["account_id"])

    n_prior = out.groupby("account_id", sort=False).cumcount()
    mean_prior = grouped.transform(lambda s: s.shift(1).expanding().mean())
    std_prior = grouped.transform(lambda s: s.shift(1).expanding().std())

    t = (pd.to_datetime(out[time_col]).astype("int64") / 1e9
         if time_col == "timestamp" else out[time_col].astype(float))
    gap = t.groupby(out["account_id"]).diff()

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
    df_eval: pd.DataFrame, df_train: pd.DataFrame, seed: int
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
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    y_train = df_train["label"].fillna(False).astype(int).to_numpy()
    base_cols = ["amount_norm", "channel_encoded"]

    rng = np.random.default_rng(seed)
    scores: dict[str, np.ndarray] = {
        "ngau_nhien": rng.random(len(df_eval)),
        "chi_so_tien": df_eval["amount"].astype(float).to_numpy(),
    }
    if y_train.sum() == 0:
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

    x_tr, x_ev = feature_sets["cung_dau_vao"]
    scaler = StandardScaler().fit(x_tr)
    logistic = LogisticRegression(max_iter=1000, class_weight="balanced")
    logistic.fit(scaler.transform(x_tr), y_train)
    scores["logistic_cung_dau_vao"] = logistic.predict_proba(scaler.transform(x_ev))[:, 1]

    for tag, (x_tr, x_ev) in feature_sets.items():
        clf = HistGradientBoostingClassifier(max_iter=200, random_state=seed)
        clf.fit(x_tr, y_train)
        scores[f"gbdt_{tag}"] = clf.predict_proba(x_ev)[:, 1]
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
    parser.add_argument("--source", required=True, choices=["paysim", "ieee_cis", "synthetic_vn"])
    parser.add_argument("--split", default="test", choices=["val", "test"])
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
    parser.add_argument("--out-dir", default="reports/eval")
    args = parser.parse_args(argv)

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
    y = df["label"].fillna(False).astype(int).to_numpy()
    if y.sum() == 0:
        parser.error(f"{path} không có nhãn dương nào — không đánh giá được.")

    logger.info("%s %s: %d dòng, %d nhãn dương (tỷ lệ %.4f%%)",
                args.source, args.split, len(df), int(y.sum()), 100 * y.mean())
    if args.split == "test":
        logger.warning(
            "Đang chấm trên tập TEST. Chỉ nên chạm MỘT LẦN sau khi đã chốt model bằng val — "
            "chạy lại nhiều lần để thử cấu hình là đang dùng test làm val.")

    results: dict[str, dict[str, float]] = {}
    score_map: dict[str, np.ndarray] = {}

    chunk = args.score_batch_size or None
    scored = score_batch(df, args.checkpoint, batch_size=chunk)
    main_scores = scored["risk_score"].to_numpy(dtype=float)
    # Chặn lệch điểm–nhãn: khung đã chấm phải giữ đúng thứ tự nhãn ban đầu.
    if "label" in scored.columns:
        scored_labels = scored["label"].fillna(False).astype(int).to_numpy()
        if not np.array_equal(scored_labels, y):
            parser.error(
                "Thứ tự dòng sau khi chấm điểm KHÁC thứ tự nhãn — mọi chỉ số sẽ sai. "
                "Kiểm tra lại cột thời gian của nguồn dữ liệu này.")
    method = str(scored["method"].iloc[0])
    label = f"{Path(args.checkpoint).name} ({method})"
    results[label] = metrics_for(y, main_scores, budgets)
    score_map[label] = main_scores
    if method != "drre_trained":
        logger.warning(
            "Checkpoint không nạp được nên đã rơi về quy tắc thống kê (method=%s). "
            "Số liệu dưới đây KHÔNG phải của DRRE.", method)

    compare_scores = None
    if args.compare_checkpoint:
        scored_b = score_batch(df, args.compare_checkpoint, batch_size=chunk)
        compare_scores = scored_b["risk_score"].to_numpy(dtype=float)
        compare_label = f"{Path(args.compare_checkpoint).name} ({scored_b['method'].iloc[0]})"
        results[compare_label] = metrics_for(y, compare_scores, budgets)
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
            for name, scores in _baseline_scores(df, df_train, args.seed).items():
                results[name] = metrics_for(y, scores, budgets)
                score_map[name] = scores

    baseline_bootstrap = None
    if args.compare_baseline:
        if args.compare_baseline not in score_map:
            parser.error(
                f"Không có phương pháp tên {args.compare_baseline!r}. Các tên dùng được: "
                + ", ".join(sorted(score_map)))
        diff, lo, hi = bootstrap_metric_diff(
            y, main_scores, score_map[args.compare_baseline], pr_auc,
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
            y, main_scores, compare_scores, pr_auc, n_boot=args.bootstrap, seed=args.seed)
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
            y, score_map, args.by_period, time_values=df[_time_col].to_numpy())
        method_names = [name for name, _ in sorted(results.items(), key=lambda kv: kv[1]["auprc"])]
        period_table = _by_period_table(periods, method_names)
        print(f"Theo {len(periods)} giai đoạn thời gian — ô ghi AUPRC (×lift so với mốc "
              f"ngẫu nhiên CỦA GIAI ĐOẠN đó):")
        print("\n" + period_table + "\n")
        thin = [r["period"] for r in periods if r.get("n_pos", 0) < 30]
        if thin:
            print(f"CẢNH BÁO: giai đoạn {thin} có dưới 30 nhãn dương — AUPRC ở đó nhiễu "
                  f"rất mạnh, KHÔNG dùng để kết luận xu hướng.\n")

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
        },
    )
    manifest.finish(started)
    out_path = manifest.write(args.out_dir)
    logger.info("Đã ghi kết quả: %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
