"""Bảng xếp hạng mọi mô hình trên CÙNG một tập dòng, theo TỪNG phép đo.

Vì sao cần script riêng: mỗi báo cáo evaluate.py chỉ so một cặp checkpoint, các báo cáo
khác ngày có thể khác cách đo (vd. trước 09/11 chấm DRRE không mang trạng thái, trước
14/11 baseline chỉ có 20 cột gốc). Ở đây mọi mô hình được chấm lại theo một quy tắc:

  * cùng tập (val mặc định), cùng vũ trụ chấm điểm (eval_universe: bỏ Wire,
    Reinvestment), cùng thứ tự dòng;
  * DRRE chấm CÓ MANG TRẠNG THÁI (làm nóng trên train, như lúc huấn luyện);
  * mô hình cổ điển học trên train với ĐÚNG bộ cột của DRRE (28 cột: 18 gốc + kịch bản),
    cộng một GBDT 20 cột gốc làm mốc của đợt 1;
  * mỗi phép đo có thứ hạng riêng (1 = tốt nhất); hiệu số AUPRC so với hạng 1 kèm KTC 95%
    bootstrap GHÉP CẶP (Poisson, cùng trọng số cho mọi mô hình trong một vòng) — mô hình
    có KTC chứa 0 là "chưa tách được khỏi hạng 1".

Điểm từng mô hình được lưu ở <out>/scores/*.npy (theo thứ tự dòng GỐC của tệp) kèm dấu
vân tay; chạy lại chỉ tính phần thay đổi.

    python -X utf8 scripts/leaderboard.py --processed-dir data/processed_aml_v2 \\
        --drre R2=models/drre_r2_hi_small.pt --drre R3=models/drre_r3_hi_small.pt \\
        --drre R4=models/drre_r4_hi_small.pt --drre R5=models/drre_r5_hi_small.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from sklearn.metrics import roc_auc_score  # noqa: E402

from src.data.features import features_for_source  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

BUDGETS = (500, 1000, 2000)
RF_N_JOBS = 4
RF_MAX_SAMPLES = 500_000
#: Phép đo đưa vào hạng trung bình (cao hơn là tốt hơn với tất cả).
RANKED = ("auprc", "roc_auc", "caught@500", "caught@1000", "caught@2000", "f1_best")


# --------------------------------------------------------------------------
# Phép đo
# --------------------------------------------------------------------------

def weighted_ap(y: np.ndarray, order: np.ndarray, boundary: np.ndarray,
                w: np.ndarray | None = None) -> float:
    """Average precision theo đúng định nghĩa của sklearn (gộp các điểm bằng nhau),
    có trọng số mẫu — dùng cho bootstrap Poisson mà không phải sắp xếp lại.

    `order` là chỉ số sắp điểm giảm dần; `boundary[i]` = True nếu vị trí i là phần tử
    cuối của một nhóm điểm bằng nhau (theo thứ tự `order`).
    """
    ys = y[order].astype(np.float64)
    ws = np.ones(len(ys)) if w is None else w[order].astype(np.float64)
    tp = np.cumsum(ws * ys)[boundary]
    allc = np.cumsum(ws)[boundary]
    total_pos = tp[-1] if len(tp) else 0.0
    if total_pos <= 0:
        return 0.0
    precision = tp / np.maximum(allc, 1e-12)
    recall = tp / total_pos
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def ranking_prep(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-scores, kind="stable")
    s = scores[order]
    boundary = np.r_[s[1:] != s[:-1], True]
    return order, boundary


def best_f1(y: np.ndarray, order: np.ndarray, boundary: np.ndarray) -> dict:
    ys = y[order].astype(np.float64)
    tp = np.cumsum(ys)[boundary]
    alerts = (np.flatnonzero(boundary) + 1).astype(np.float64)
    precision, recall = tp / alerts, tp / max(tp[-1], 1.0)
    f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    k = int(np.argmax(f1))
    return {"f1_best": float(f1[k]), "precision_at_f1": float(precision[k]),
            "recall_at_f1": float(recall[k]), "alerts_at_f1": int(alerts[k]),
            "tp_at_f1": int(tp[k])}


def metrics(y: np.ndarray, scores: np.ndarray, patterns: np.ndarray) -> dict:
    order, boundary = ranking_prep(scores)
    out = {"auprc": weighted_ap(y, order, boundary), "roc_auc": float(roc_auc_score(y, scores))}
    for k in BUDGETS:
        out[f"caught@{k}"] = int(y[order[:k]].sum())
    out.update(best_f1(y, order, boundary))
    top = np.zeros(len(y), dtype=bool)
    top[order[:max(BUDGETS)]] = True
    out["typology_recall@2000"] = {
        str(kind): float(top[(patterns == kind) & y].sum() / max(((patterns == kind) & y).sum(), 1))
        for kind in sorted(set(patterns[y].tolist())) if kind}
    return out


def paired_bootstrap(y: np.ndarray, score_map: dict[str, np.ndarray], leader: str,
                     rounds: int, seed: int) -> dict[str, dict]:
    """KTC 95% của AUPRC(mô hình) − AUPRC(leader), bootstrap Poisson ghép cặp."""
    prep = {name: ranking_prep(s) for name, s in score_map.items()}
    rng = np.random.default_rng(seed)
    diffs = {name: [] for name in score_map if name != leader}
    for _ in range(rounds):
        w = rng.poisson(1.0, len(y)).astype(np.float64)
        ref = weighted_ap(y, *prep[leader], w)
        for name in diffs:
            diffs[name].append(weighted_ap(y, *prep[name], w) - ref)
    return {name: {"lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5))}
            for name, d in diffs.items()}


def rank_desc(values: dict[str, float]) -> dict[str, int]:
    """Hạng 1 = giá trị lớn nhất; bằng nhau thì cùng hạng (kiểu thi đấu: 1, 1, 3)."""
    ordered = sorted(values.values(), reverse=True)
    return {name: 1 + ordered.index(v) for name, v in values.items()}


# --------------------------------------------------------------------------
# Điểm của từng mô hình (có bộ nhớ đệm)
# --------------------------------------------------------------------------

def _stamp(path: Path) -> list:
    st = os.stat(path)
    return [path.name, st.st_size, st.st_mtime_ns]


class ScoreCache:
    def __init__(self, folder: Path):
        self.folder = folder
        folder.mkdir(parents=True, exist_ok=True)

    def get(self, name: str, key: dict) -> np.ndarray | None:
        meta, data = self.folder / f"{name}.json", self.folder / f"{name}.npy"
        if meta.exists() and data.exists() and json.loads(meta.read_text("utf-8")) == key:
            return np.load(data)
        return None

    def put(self, name: str, key: dict, scores: np.ndarray) -> None:
        np.save(self.folder / f"{name}.npy", scores)
        (self.folder / f"{name}.json").write_text(json.dumps(key), encoding="utf-8")


def score_drre(checkpoint: str, train_path: Path, eval_path: Path, window: int) -> np.ndarray:
    """Điểm DRRE có mang trạng thái, trả về theo thứ tự dòng GỐC của tệp eval."""
    import torch

    from src.dashboard.scoring_service import try_load_drre_checkpoint
    from src.data.scenarios.store import read_split
    from src.drre.past_graph import make_buffer
    from src.drre.train import BehaviorStateStore
    from src.evaluation.stateful_scoring import (
        fingerprint, required_capacity, score_split, stream_columns, warm_up_store,
    )

    loaded = try_load_drre_checkpoint(checkpoint)
    if loaded is None:
        raise SystemExit(f"Không nạp được {checkpoint}")
    model, engine = loaded
    model.eval()
    columns = stream_columns(model, "timestamp")
    stream = read_split(train_path, columns).sort_values(
        "timestamp", kind="stable").reset_index(drop=True)
    frame = read_split(eval_path, columns)
    order = np.argsort(frame["timestamp"].to_numpy(), kind="stable")
    frame = frame.iloc[order].reset_index(drop=True)
    store = BehaviorStateStore(model.behavior_memory.hidden_dim, model.behavior_memory.num_slots,
                               capacity=required_capacity([stream, frame]))
    graph_buffer = make_buffer(model)
    with torch.no_grad():
        warm_up_store(model, engine, stream, store,
                      Path(checkpoint).with_suffix(f".warm_{eval_path.stem.rsplit('_', 1)[-1]}.pt"),
                      fingerprint(checkpoint, [train_path], window), window=window,
                      graph_buffer=graph_buffer)
        del stream
        sorted_scores = score_split(model, engine, frame, store, window=window,
                                    graph_buffer=graph_buffer)
    scores = np.empty(len(sorted_scores))
    scores[order] = sorted_scores
    return scores


def classical_scores(name: str, columns: list[str], train: pd.DataFrame, fit_rows: np.ndarray,
                     evaluate: pd.DataFrame, seed: int) -> np.ndarray:
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    x_fit = train.loc[fit_rows, columns].to_numpy(np.float32)
    y_fit = train.loc[fit_rows, "label"].astype(bool).to_numpy()
    x_eval = evaluate[columns].to_numpy(np.float32)
    kind = name.split("_")[0]
    if kind == "gbdt":
        model = HistGradientBoostingClassifier(max_iter=200, class_weight="balanced",
                                               early_stopping=False, random_state=seed)
    elif kind == "rf":
        model = RandomForestClassifier(
            n_estimators=100, max_depth=12, min_samples_leaf=50, max_samples=RF_MAX_SAMPLES,
            class_weight="balanced_subsample", n_jobs=RF_N_JOBS, random_state=seed)
    elif kind == "logistic":
        model = make_pipeline(StandardScaler(),
                              LogisticRegression(max_iter=1000, class_weight="balanced"))
    else:
        raise ValueError(name)
    model.fit(x_fit, y_fit)
    return model.predict_proba(x_eval)[:, 1]


# --------------------------------------------------------------------------
# Bảng
# --------------------------------------------------------------------------

def markdown(board: dict) -> str:
    rows = board["models"]
    names = sorted(rows, key=lambda n: board["ranks"]["auprc"][n])
    lines = [
        f"# Bảng xếp hạng — {board['split']} ({board['data']})",
        "",
        f"{board['n_rows']:,} dòng trong vũ trụ chấm điểm, {board['n_pos']:,} nhãn dương "
        f"(tỷ lệ nền {board['base_rate']:.5f}). Mỗi ô: giá trị (hạng). Hạng TB = trung bình hạng "
        f"của {', '.join(RANKED)}. Δ AUPRC so với hạng 1 kèm KTC 95% bootstrap ghép cặp "
        f"({board['bootstrap']} vòng); KTC chứa 0 = chưa tách được khỏi hạng 1.",
        "",
        "| # | Mô hình | Số cột | AUPRC | ROC-AUC | Bắt@500 | Bắt@1000 | Bắt@2000 | F1 tốt nhất "
        "(P / R) | Hạng TB | Δ AUPRC vs hạng 1 [KTC 95%] |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for i, n in enumerate(names, 1):
        m, r = rows[n], {k: board["ranks"][k][n] for k in RANKED}
        ci = board["vs_leader"].get(n)
        delta = ("—" if ci is None else
                 f"{m['auprc'] - rows[board['leader']]['auprc']:+.3f} [{ci['lo']:+.3f}; {ci['hi']:+.3f}]")
        lines.append(
            f"| {i} | {n} | {board['n_features'].get(n, '—')} | {m['auprc']:.4f} ({r['auprc']}) "
            f"| {m['roc_auc']:.4f} ({r['roc_auc']}) | {m['caught@500']} ({r['caught@500']}) "
            f"| {m['caught@1000']} ({r['caught@1000']}) | {m['caught@2000']} ({r['caught@2000']}) "
            f"| {m['f1_best']:.3f} ({m['precision_at_f1']:.0%} / {m['recall_at_f1']:.0%}) "
            f"({r['f1_best']}) | {board['mean_rank'][n]:.2f} | {delta} |")
    kinds = sorted({k for n in names for k in rows[n]["typology_recall@2000"]},
                   key=lambda k: -board["typology_n"].get(k, 0))
    lines += ["", "## Recall theo kiểu hình trong 2.000 cảnh báo đầu — giá trị (hạng)", "",
              "| Mô hình | " + " | ".join(f"{k} (n={board['typology_n'].get(k, 0)})" for k in kinds)
              + " | Hạng TB kiểu hình |",
              "|---|" + "---|" * (len(kinds) + 1)]
    for n in names:
        cells = [f"{rows[n]['typology_recall@2000'].get(k, 0):.0%} "
                 f"({board['typology_ranks'][k][n]})" for k in kinds]
        lines.append(f"| {n} | " + " | ".join(cells) + f" | {board['typology_mean_rank'][n]:.2f} |")
    lines += ["", "Ghi chú: " + " ".join(board["notes"])]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--source", default="amlworld")
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--drre", action="append", default=[], metavar="TEN=CHECKPOINT")
    parser.add_argument("--features-from", default=None,
                        help="checkpoint lấy bộ cột cho mô hình cổ điển (mặc định: DRRE cuối)")
    parser.add_argument("--no-classical", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=300)
    parser.add_argument("--window", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="reports/leaderboard")
    parser.add_argument("--keep-awake", action="store_true")
    args = parser.parse_args()

    if args.keep_awake and sys.platform == "win32":
        import ctypes

        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    if args.split == "test":
        logger.warning("Đang xếp hạng trên tập TEST — chỉ làm MỘT LẦN khi đã chốt mô hình.")

    import torch

    from src.data.scenarios.store import read_split

    started = time.monotonic()
    folder = Path(args.processed_dir)
    train_path = folder / f"{args.source}_train.parquet"
    eval_path = folder / f"{args.source}_{args.split}.parquet"
    out = Path(args.out_dir) / f"{folder.name}_{args.split}"
    cache = ScoreCache(out / "scores")

    drre = {}
    for item in args.drre:
        name, _, path = item.partition("=")
        if not Path(path).exists():
            logger.warning("Bỏ qua %s: chưa có %s", name, path)
            continue
        drre[name] = path
    ref = args.features_from or (list(drre.values())[-1] if drre else None)
    base20 = features_for_source(args.source)
    full = (list(torch.load(ref, map_location="cpu", weights_only=True)["feature_columns"])
            if ref else base20)
    n_features = {n: len(torch.load(p, map_location="cpu", weights_only=True)["feature_columns"])
                  for n, p in drre.items()}

    frame = read_split(eval_path, sorted(set(base20 + full)) + ["label", "eval_universe",
                                                                "laundering_pattern"])
    keep = frame["eval_universe"].fillna(True).to_numpy(bool)
    y = frame["label"].fillna(False).astype(bool).to_numpy()[keep]
    patterns = frame["laundering_pattern"].astype(object).fillna("").to_numpy(str)[keep]
    data_stamp = [_stamp(train_path), _stamp(eval_path)]

    score_map: dict[str, np.ndarray] = {}
    for name, path in drre.items():
        key = {"checkpoint": _stamp(Path(path)), "data": data_stamp, "window": args.window}
        scores = cache.get(name, key)
        if scores is None:
            logger.info("Chấm %s (%s) có mang trạng thái...", name, path)
            scores = score_drre(path, train_path, eval_path, args.window)
            cache.put(name, key, scores)
        score_map[name] = scores

    if not args.no_classical:
        n_full = len(full)
        plan = {f"gbdt_{n_full}": full, f"rf_{n_full}": full, f"logistic_{n_full}": full,
                "gbdt_20": base20}
        train = None
        for name, columns in plan.items():
            key = {"columns": columns, "data": data_stamp, "seed": args.seed, "v": 1}
            scores = cache.get(name, key)
            if scores is None:
                if train is None:
                    train = read_split(train_path, sorted(set(base20 + full)) +
                                       ["label", "eval_universe"])
                    fit_rows = train["eval_universe"].fillna(True).to_numpy(bool)
                logger.info("Học %s trên %d cột...", name, len(columns))
                scores = classical_scores(name, columns, train, fit_rows, frame, args.seed)
                cache.put(name, key, scores)
            score_map[name] = scores
            n_features[name] = len(columns)
        del train
        score_map["ngau_nhien"] = np.random.default_rng(args.seed).random(len(frame))
        n_features["ngau_nhien"] = 0

    rows = {n: metrics(y, s[keep], patterns) for n, s in score_map.items()}
    ranks = {m: rank_desc({n: rows[n][m] for n in rows}) for m in RANKED}
    mean_rank = {n: float(np.mean([ranks[m][n] for m in RANKED])) for n in rows}
    leader = min(rows, key=lambda n: ranks["auprc"][n])
    logger.info("Bootstrap ghép cặp %d vòng so với %s...", args.bootstrap, leader)
    vs_leader = paired_bootstrap(y, {n: s[keep] for n, s in score_map.items()}, leader,
                                 args.bootstrap, args.seed)
    kinds = sorted({k for r in rows.values() for k in r["typology_recall@2000"]})
    typology_ranks = {k: rank_desc({n: rows[n]["typology_recall@2000"].get(k, 0.0) for n in rows})
                      for k in kinds}
    board = {
        "split": args.split, "data": str(folder), "n_rows": int(keep.sum()),
        "n_pos": int(y.sum()), "base_rate": float(y.mean()), "bootstrap": args.bootstrap,
        "leader": leader, "models": rows, "ranks": ranks, "mean_rank": mean_rank,
        "vs_leader": vs_leader, "n_features": n_features,
        "drre_checkpoints": drre, "classical_feature_columns": full,
        "typology_n": {k: int(((patterns == k) & y).sum()) for k in kinds},
        "typology_ranks": typology_ranks,
        "typology_mean_rank": {n: float(np.mean([typology_ranks[k][n] for k in kinds]))
                               for n in rows},
        "notes": [
            "Một seed (42).",
            "F1 tốt nhất chọn ngưỡng trên CHÍNH tập này nên lạc quan; chỉ dùng để so tương đối.",
            "Checkpoint DRRE được chọn theo val nên val hơi lạc quan cho DRRE; số chốt là "
            "test, chạy một lần.",
            "Mô hình cổ điển học trên vũ trụ chấm điểm của train, cùng bộ cột với DRRE.",
        ],
        "seconds": round(time.monotonic() - started, 1),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "leaderboard.json").write_text(json.dumps(board, indent=2, ensure_ascii=False),
                                          encoding="utf-8")
    (out / "leaderboard.md").write_text(markdown(board), encoding="utf-8")
    logger.info("Đã ghi %s", out / "leaderboard.md")
    print(markdown(board))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
