"""DRRE đóng góp gì cho mô hình cổ điển — phép đo P2 và P3 của lộ trình đợt 2.

P2: GBDT có và không có điểm/embedding của DRRE, kèm khoảng tin cậy bootstrap.
P3: xáo trộn cả nhóm đặc trưng để xem nhóm DRRE gánh bao nhiêu.

Embedding được trích bằng đường chấm CÓ MANG TRẠNG THÁI, ghi vào memmap trên đĩa.
Tiến trình chết giữa chừng thì chạy lại đúng lệnh này: phần đã quét không phải
quét lại. Tập làm nóng là train; chỉ đo trên val.
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

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from sklearn.decomposition import PCA  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.metrics import average_precision_score as ap  # noqa: E402

from src.dashboard.scoring_service import try_load_drre_checkpoint  # noqa: E402
from src.drre.train import BehaviorStateStore  # noqa: E402
from src.evaluation.stateful_scoring import (  # noqa: E402
    DEFAULT_WINDOW, fingerprint, required_capacity, score_split, stream_columns, time_sorted,
    warm_up_store,
)
from src.tracking import RunManifest, dataframe_summary, file_fingerprint  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PCA_DIM = 16
PCA_SAMPLE = 500_000
BOOTSTRAP_ROUNDS = 150

GROUPS = {
    "giao_dich": ["amount_norm", "channel_encoded", "is_round_amount", "new_counterparty"],
    "thoi_quen": ["log_n_prior", "log_gap", "z_amount", "log_amount_ratio"],
    "thoi_diem": ["hour_sin", "hour_cos", "dow"],
    "doi_tac_mang_luoi": [
        "log_n_distinct_cp", "log_n_prior_as_cp", "log_n_distinct_senders", "log_amt_out_prior",
        "log_amt_in_prior", "log_n_in_prior", "passthrough_ratio", "flow_imbalance",
        "log_cp_amt_in_prior",
    ],
}


def _memmaps(folder: Path, name: str, rows: int, dim: int, create: bool):
    mode = "w+" if create else "r+"
    embedding = np.memmap(folder / f"{name}_e.f16", dtype=np.float16, mode=mode, shape=(rows, dim))
    logit = np.memmap(folder / f"{name}_s.f32", dtype=np.float32, mode=mode, shape=(rows,))
    return embedding, logit


def _writer(embedding: np.memmap, logit: np.memmap):
    def on_window(start: int, stop: int, e, s) -> None:
        embedding[start:stop] = e.numpy().astype(np.float16)
        logit[start:stop] = s.numpy()
        embedding.flush()
        logit.flush()
    return on_window


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--source", default="amlworld")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep-awake", action="store_true")
    parser.add_argument("--out-dir", default="reports/eval")
    args = parser.parse_args()

    if args.keep_awake and sys.platform == "win32":
        import ctypes

        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)

    started = time.time()
    loaded = try_load_drre_checkpoint(args.checkpoint)
    if loaded is None:
        raise SystemExit(f"Không nạp được {args.checkpoint}")
    model, engine = loaded
    features = list(model.feature_columns)
    dim = model.fusion.phi_1.out_features

    processed = Path(args.processed_dir)
    train_path = processed / f"{args.source}_train.parquet"
    val_path = processed / f"{args.source}_val.parquet"
    columns = stream_columns(model) + ["label", "eval_universe"]
    train = time_sorted(pd.read_parquet(train_path, columns=columns))
    val = time_sorted(pd.read_parquet(val_path, columns=columns))
    logger.info("train %d dòng, val %d dòng", len(train), len(val))

    folder = Path(args.checkpoint).with_suffix(".contrib")
    folder.mkdir(exist_ok=True)
    cache = Path(args.checkpoint).with_suffix(".warm_val.pt")
    fresh = not (folder / "train_e.f16").exists()
    if fresh and cache.exists():
        raise SystemExit(
            f"{cache} đã có nhưng chưa có embedding trong {folder}: quét làm nóng đã xong mà không "
            "ghi embedding. Chạy script này TRƯỚC evaluate.py, hoặc xóa file cache để quét lại.")

    e_train, s_train = _memmaps(folder, "train", len(train), dim, create=fresh)
    store = BehaviorStateStore(
        model.behavior_memory.hidden_dim, model.behavior_memory.num_slots,
        capacity=required_capacity([train, val]))
    window = DEFAULT_WINDOW
    warm_up_store(model, engine, train, store, cache,
                  fingerprint(args.checkpoint, [train_path], window), window=window,
                  on_window=_writer(e_train, s_train))
    logger.info("Quét train xong: gặp lại %.1f%% tài khoản, %d lần đẩy khỏi bộ đệm",
                100 * store.hit_rate, store.evictions)

    e_val, s_val = _memmaps(folder, "val", len(val), dim, create=True)
    score_split(model, engine, val, store, window=window, on_window=_writer(e_val, s_val))
    del store

    u_train = train["eval_universe"].astype(bool).to_numpy()
    u_val = val["eval_universe"].astype(bool).to_numpy()
    x_train = train.loc[u_train, features].to_numpy(np.float32)
    y_train = train.loc[u_train, "label"].astype(bool).to_numpy()
    x_val = val.loc[u_val, features].to_numpy(np.float32)
    y_val = val.loc[u_val, "label"].astype(bool).to_numpy()
    val_summary = dataframe_summary(val)
    del train, val

    idx_train, idx_val = np.flatnonzero(u_train), np.flatnonzero(u_val)
    logit_train = np.asarray(s_train)[idx_train].astype(np.float32)[:, None]
    logit_val = np.asarray(s_val)[idx_val].astype(np.float32)[:, None]

    rng = np.random.default_rng(args.seed)
    sample = np.sort(rng.choice(idx_train, min(PCA_SAMPLE, len(idx_train)), replace=False))
    pca = PCA(PCA_DIM, random_state=args.seed).fit(np.asarray(e_train[sample], dtype=np.float32))

    def project(source: np.memmap, index: np.ndarray) -> np.ndarray:
        out = np.empty((len(index), PCA_DIM), np.float32)
        for start in range(0, len(index), 1_000_000):
            part = index[start:start + 1_000_000]
            out[start:start + len(part)] = pca.transform(np.asarray(source[part], dtype=np.float32))
        return out

    p_train, p_val = project(e_train, idx_train), project(e_val, idx_val)
    results: dict = {
        "base_rate": float(y_val.mean()),
        "drre_alone_auprc": float(ap(y_val, logit_val.ravel())),
        "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
    }
    logger.info("DRRE đứng riêng (có mang trạng thái): AUPRC %.4f", results["drre_alone_auprc"])

    variants = {
        "gbdt": (x_train, x_val),
        "gbdt+diem_drre": (np.c_[x_train, logit_train], np.c_[x_val, logit_val]),
        "gbdt+emb_drre": (np.c_[x_train, p_train], np.c_[x_val, p_val]),
        "gbdt+emb_drre+diem": (np.c_[x_train, p_train, logit_train], np.c_[x_val, p_val, logit_val]),
    }
    scores: dict[str, np.ndarray] = {}
    fitted = {}
    for name, (fit_x, eval_x) in variants.items():
        clf = HistGradientBoostingClassifier(
            max_iter=200, class_weight="balanced", random_state=args.seed).fit(fit_x, y_train)
        scores[name], fitted[name] = clf.predict_proba(eval_x)[:, 1], clf
        results[name] = {"auprc": float(ap(y_val, scores[name])), "n_iter": int(clf.n_iter_)}
        order = np.argsort(-scores[name], kind="stable")
        results[name].update({f"caught@{k}": int(y_val[order[:k]].sum()) for k in (500, 1000, 2000)})
        logger.info("%s: AUPRC %.4f", name, results[name]["auprc"])

    positive, negative = np.flatnonzero(y_val), np.flatnonzero(~y_val)

    def paired_ci(a: np.ndarray, b: np.ndarray) -> dict:
        diffs = []
        for _ in range(BOOTSTRAP_ROUNDS):
            i = np.concatenate([rng.choice(positive, len(positive)), rng.choice(negative, len(negative))])
            diffs.append(ap(y_val[i], a[i]) - ap(y_val[i], b[i]))
        return {"lo": float(np.percentile(diffs, 2.5)), "hi": float(np.percentile(diffs, 97.5)),
                "share_positive": float(np.mean(np.array(diffs) > 0))}

    for name in list(variants)[1:]:
        results[name]["minus_gbdt"] = results[name]["auprc"] - results["gbdt"]["auprc"]
        results[name]["ci95_minus_gbdt"] = paired_ci(scores[name], scores["gbdt"])
        logger.info("%s − gbdt: %+.4f, KTC %s", name, results[name]["minus_gbdt"],
                    results[name]["ci95_minus_gbdt"])

    subset = np.concatenate([positive, rng.choice(negative, 400_000, replace=False)])
    matrix = variants["gbdt+emb_drre+diem"][1][subset]
    labels = y_val[subset]
    model_all = fitted["gbdt+emb_drre+diem"]
    base = ap(labels, model_all.predict_proba(matrix)[:, 1])
    columns_of = {g: [features.index(c) for c in cs if c in features] for g, cs in GROUPS.items()}
    columns_of["DRRE"] = list(range(len(features), len(features) + PCA_DIM + 1))
    drops = {}
    for group, index in columns_of.items():
        values = []
        for _ in range(3):
            shuffled = matrix.copy()
            shuffled[:, index] = shuffled[rng.permutation(len(shuffled))][:, index]
            values.append(base - ap(labels, model_all.predict_proba(shuffled)[:, 1]))
        drops[group] = float(np.mean(values))
    results["group_permutation"] = {"base_auprc_on_sample": float(base), "drop": drops}
    logger.info("Xáo trộn cả nhóm, AUPRC mất đi: %s", drops)

    manifest = RunManifest(
        kind=f"drre_contribution_{args.source}_val",
        seed=args.seed,
        hyperparameters={"checkpoint": args.checkpoint, "window": window, "pca_dim": PCA_DIM,
                         "bootstrap": BOOTSTRAP_ROUNDS, "stateful": True},
        data={"train_file": file_fingerprint(train_path),
              "eval_file": {**file_fingerprint(val_path), **val_summary}},
        results=results,
    )
    manifest.finish(started)
    logger.info("Đã ghi kết quả: %s", manifest.write(args.out_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
