"""Đo mức sụp của embedding DRRE trên val (chẩn đoán cho Giai đoạn 2C).

Embedding e_t được trích bằng đường chấm CÓ MANG TRẠNG THÁI (làm nóng trên
train, rồi chấm val), dùng lại bộ nhớ làm nóng mà scripts/evaluate.py đã lưu
cạnh checkpoint (<checkpoint>.warm_val.pt) nếu dấu vân tay khớp.

Chỉ số:
  - participation_ratio = (Σλ)² / Σλ²  — số chiều "thực sự dùng" (1..dim)
  - effective_rank = exp(entropy của λ/Σλ)  (Roy & Vetterli, 2007)
  - dims_90 / dims_99 — số thành phần chính cần để giữ 90% / 99% phương sai
Tính trên mẫu ngẫu nhiên của val, riêng cho cả mẫu và cho nhãn dương.

    python -X utf8 scripts/embedding_health.py --checkpoint models/drre_r3_hi_small.pt \\
        --processed-dir data/processed_aml_v2 --out reports/r3/embedding_r3.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from src.dashboard.scoring_service import try_load_drre_checkpoint  # noqa: E402
from src.data.scenarios.store import read_split  # noqa: E402
from src.drre.past_graph import make_buffer  # noqa: E402
from src.drre.train import BehaviorStateStore  # noqa: E402
from src.evaluation.stateful_scoring import (  # noqa: E402
    DEFAULT_WINDOW, fingerprint, required_capacity, score_split, stream_columns, time_sorted,
    warm_up_store,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

SAMPLE_RATE = 0.15


def spectrum_stats(x: np.ndarray) -> dict[str, float]:
    """Thống kê phổ hiệp phương sai của các hàng trong `x`."""
    x = x.astype(np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    cov = x.T @ x / max(len(x) - 1, 1)
    lam = np.clip(np.linalg.eigvalsh(cov)[::-1], 0.0, None)
    total = lam.sum()
    if total <= 0:
        return {"rows": int(len(x)), "participation_ratio": 0.0, "effective_rank": 0.0,
                "dims_90": 0, "dims_99": 0, "top1_share": 1.0}
    p = lam / total
    nz = p[p > 0]
    cum = np.cumsum(p)
    return {
        "rows": int(len(x)),
        "participation_ratio": float(total ** 2 / (lam ** 2).sum()),
        "effective_rank": float(np.exp(-(nz * np.log(nz)).sum())),
        "dims_90": int(np.searchsorted(cum, 0.90) + 1),
        "dims_99": int(np.searchsorted(cum, 0.99) + 1),
        "top1_share": float(p[0]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--source", default="amlworld")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    loaded = try_load_drre_checkpoint(args.checkpoint)
    if loaded is None:
        raise SystemExit(f"Không nạp được {args.checkpoint}")
    model, risk_engine = loaded
    model.eval()
    folder = Path(args.processed_dir)
    train_path = folder / f"{args.source}_train.parquet"
    val_path = folder / f"{args.source}_val.parquet"

    time_col = "timestamp"
    columns = stream_columns(model, time_col)
    stream = time_sorted(read_split(train_path, columns))
    val = time_sorted(read_split(val_path, columns + ["label"]))
    store = BehaviorStateStore(model.behavior_memory.hidden_dim, model.behavior_memory.num_slots,
                               capacity=required_capacity([stream, val]))
    cache = Path(args.checkpoint).with_suffix(".warm_val.pt")
    graph_buffer = make_buffer(model)
    warm_up_store(model, risk_engine, stream, store, cache,
                  fingerprint(args.checkpoint, [train_path], args.window), window=args.window,
                  graph_buffer=graph_buffer)
    del stream

    rng = np.random.default_rng(args.seed)
    labels = val["label"].fillna(False).astype(bool).to_numpy()
    keep = rng.random(len(val)) < SAMPLE_RATE
    keep |= labels  # giữ mọi nhãn dương
    sample_rows: list[np.ndarray] = []
    sample_pos: list[np.ndarray] = []

    def on_window(start: int, stop: int, embedding, _logit) -> None:
        e = embedding.numpy()
        sel = keep[start:stop]
        sample_rows.append(e[sel].astype(np.float32))
        sample_pos.append(labels[start:stop][sel])

    score_split(model, risk_engine, val, store, window=args.window, on_window=on_window,
                graph_buffer=graph_buffer)
    e = np.concatenate(sample_rows)
    pos = np.concatenate(sample_pos)
    report = {
        "checkpoint": args.checkpoint,
        "arch": type(model).__name__,
        "embedding_dim": int(model.embedding_dim),
        "all": spectrum_stats(e),
        "positives": spectrum_stats(e[pos]),
        "mean_norm_pos": float(np.linalg.norm(e[pos], axis=1).mean()),
        "mean_norm_neg": float(np.linalg.norm(e[~pos], axis=1).mean()),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Embedding %s: %s", args.checkpoint, json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
