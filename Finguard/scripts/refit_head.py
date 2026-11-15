"""Giai đoạn 2C, pha 2: đóng băng bộ mã hóa DRRE, học lại ĐẦU CHẤM ĐIỂM trên mọi vụ cùng lúc.

=== VÌ SAO (lỗi 6 của lộ trình đợt 2) ===

Trong vòng huấn luyện chung, đầu chấm điểm chỉ học qua BCE của TỪNG cửa sổ: mỗi lần cập
nhật thấy ~26 vụ rửa tiền (2.856 vụ / 109 cửa sổ), cửa sổ chạy theo đúng thứ tự thời gian
(bắt buộc khi mang trạng thái), nên đầu chấm điểm luôn bị kéo theo đoạn dữ liệu vừa xem.

Ở đây: trích embedding e_t (có mang trạng thái, đúng như lúc chấm) cho TOÀN BỘ train, rồi
học một đầu chấm điểm mới (cùng kiến trúc) trên mọi dòng, xáo trộn, nhiều epoch.

=== CHỌN KHÔNG DÙNG VAL ===

15% CUỐI train theo thời gian là tập giữ lại (giao thức đợt 2, mục 2): mọi lựa chọn — hàm mất
mát (BCE trọng số đầy đủ / căn bậc hai / focal), số epoch — đều theo AUPRC trên tập giữ lại.
Val không được nhìn tới; val chỉ dùng để đo checkpoint kết quả như mọi bậc khác.

Kết quả: checkpoint mới = bộ mã hóa cũ + đầu chấm điểm mới (khóa "head_refit" ghi lại lựa
chọn), chấm được ở mọi đường như checkpoint thường.

    python -X utf8 scripts/refit_head.py --checkpoint models/drre_r5_hi_small.pt \\
        --processed-dir data/processed_aml_v2 --out-checkpoint models/drre_r6a_hi_small.pt
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from sklearn.metrics import average_precision_score as ap  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

LOSSES = ("bce", "bce_sqrt", "focal")
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.25


def head_loss(kind: str, logits: torch.Tensor, labels: torch.Tensor,
              pos_weight: float) -> torch.Tensor:
    """BCE trọng số đầy đủ (neg/pos), BCE trọng số căn bậc hai, hoặc focal loss (Lin 2017)."""
    if kind == "bce":
        return nn.functional.binary_cross_entropy_with_logits(
            logits, labels, pos_weight=torch.tensor(pos_weight))
    if kind == "bce_sqrt":
        return nn.functional.binary_cross_entropy_with_logits(
            logits, labels, pos_weight=torch.tensor(pos_weight ** 0.5))
    if kind == "focal":
        ce = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        p_t = torch.exp(-ce)
        alpha = FOCAL_ALPHA * labels + (1 - FOCAL_ALPHA) * (1 - labels)
        return (alpha * (1 - p_t) ** FOCAL_GAMMA * ce).mean()
    raise ValueError(kind)


@torch.no_grad()
def _logits(head, e: torch.Tensor, batch: int = 262_144) -> np.ndarray:
    return torch.cat([head.logits(e[i:i + batch]) for i in range(0, len(e), batch)]).numpy()


def fit_head(head_factory, e_fit: torch.Tensor, y_fit: torch.Tensor, e_hold: torch.Tensor,
             y_hold: np.ndarray, kind: str, epochs: int, batch: int, lr: float, patience: int,
             seed: int) -> tuple[nn.Module, dict]:
    """Học một đầu chấm điểm mới; giữ trạng thái có AUPRC tập giữ lại cao nhất."""
    torch.manual_seed(seed)
    head = head_factory()
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    pos = float(y_fit.sum())
    pos_weight = (len(y_fit) - pos) / max(pos, 1.0)
    generator = torch.Generator().manual_seed(seed)
    best, best_state, history, stale = -1.0, None, [], 0
    for epoch in range(1, epochs + 1):
        head.train()
        order = torch.randperm(len(e_fit), generator=generator)
        total = 0.0
        for i in range(0, len(order), batch):
            idx = order[i:i + batch]
            loss = head_loss(kind, head.logits(e_fit[idx]), y_fit[idx], pos_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss) * len(idx)
        head.eval()
        score = float(ap(y_hold, _logits(head, e_hold)))
        history.append({"epoch": epoch, "loss": total / len(order), "holdout_auprc": score})
        logger.info("%s epoch %d: loss %.4f, AUPRC tập giữ lại %.4f", kind, epoch,
                    total / len(order), score)
        if score > best:
            best, best_state, stale = score, copy.deepcopy(head.state_dict()), 0
        else:
            stale += 1
            if stale >= patience:
                break
    head.load_state_dict(best_state)
    return head, {"loss": kind, "holdout_auprc": best, "pos_weight": pos_weight,
                  "best_epoch": max(history, key=lambda h: h["holdout_auprc"])["epoch"],
                  "history": history}


def extract_train_embeddings(checkpoint: str, train_path: Path, folder: Path, window: int):
    """Embedding (fp16, memmap) và nhãn của train theo thứ tự thời gian, có mang trạng thái.

    Dùng tệp làm nóng riêng (<checkpoint>.warm_head.pt) để quét chạy tiếp được nếu bị ngắt.
    """
    from src.dashboard.scoring_service import try_load_drre_checkpoint
    from src.data.scenarios.store import read_split
    from src.drre.past_graph import make_buffer
    from src.drre.train import BehaviorStateStore
    from src.evaluation.stateful_scoring import (
        fingerprint, required_capacity, stream_columns, time_sorted, warm_up_store,
    )

    model, engine = try_load_drre_checkpoint(checkpoint)
    model.eval()
    stream = time_sorted(read_split(train_path, stream_columns(model, "timestamp")
                                    + ["label", "eval_universe"]))
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "train_e.f16"
    cache = Path(checkpoint).with_suffix(".warm_head.pt")
    fresh = not path.exists()
    if fresh and cache.exists():
        cache.unlink()  # có tiến độ quét mà không có embedding -> quét lại từ đầu
    emb = np.memmap(path, dtype=np.float16, mode="w+" if fresh else "r+",
                    shape=(len(stream), model.embedding_dim))

    def on_window(start, stop, e, _logit):
        emb[start:stop] = e.numpy().astype(np.float16)
        emb.flush()

    store = BehaviorStateStore(model.behavior_memory.hidden_dim, model.behavior_memory.num_slots,
                               capacity=required_capacity([stream]))
    with torch.no_grad():
        warm_up_store(model, engine, stream, store, cache,
                      fingerprint(checkpoint, [train_path], window), window=window,
                      on_window=on_window, graph_buffer=make_buffer(model))
    labels = stream["label"].fillna(False).astype(bool).to_numpy()
    universe = stream["eval_universe"].fillna(True).to_numpy(bool)
    return emb, labels, universe, model, engine


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--out-checkpoint", required=True)
    parser.add_argument("--source", default="amlworld")
    parser.add_argument("--holdout", type=float, default=0.15,
                        help="phần CUỐI train (theo thời gian) làm tập giữ lại")
    parser.add_argument("--losses", default=",".join(LOSSES))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch", type=int, default=16384)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--window", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report", default=None)
    parser.add_argument("--keep-awake", action="store_true")
    args = parser.parse_args()

    if args.keep_awake and sys.platform == "win32":
        import ctypes

        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    from src.scoring.risk_scoring import build_risk_engine

    started = time.monotonic()
    train_path = Path(args.processed_dir) / f"{args.source}_train.parquet"
    folder = Path(args.out_checkpoint).with_suffix(".head")
    emb, labels, universe, model, engine = extract_train_embeddings(
        args.checkpoint, train_path, folder, args.window)

    cut = int(len(labels) * (1 - args.holdout))
    fit_idx = np.flatnonzero(universe[:cut])
    hold_idx = cut + np.flatnonzero(universe[cut:])
    e_fit = torch.from_numpy(np.asarray(emb[fit_idx], dtype=np.float32))
    y_fit = torch.from_numpy(labels[fit_idx].astype(np.float32))
    e_hold = torch.from_numpy(np.asarray(emb[hold_idx], dtype=np.float32))
    y_hold = labels[hold_idx]
    logger.info("Học trên %d dòng (%d vụ), giữ lại %d dòng (%d vụ) — 15%% cuối train",
                len(fit_idx), int(y_fit.sum()), len(hold_idx), int(y_hold.sum()))

    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    original = float(ap(y_hold, _logits(engine, e_hold)))
    logger.info("Đầu chấm điểm cũ (học chung): AUPRC tập giữ lại %.4f", original)

    candidates = {}
    for kind in [k.strip() for k in args.losses.split(",") if k.strip()]:
        head, info = fit_head(lambda: build_risk_engine(saved["model_config"]), e_fit, y_fit,
                              e_hold, y_hold, kind, args.epochs, args.batch, args.lr,
                              args.patience, args.seed)
        candidates[kind] = (head, info)
    chosen = max(candidates, key=lambda k: candidates[k][1]["holdout_auprc"])
    head, info = candidates[chosen]
    report = {
        "checkpoint": args.checkpoint, "out_checkpoint": args.out_checkpoint,
        "holdout_fraction": args.holdout, "fit_rows": len(fit_idx),
        "fit_positives": int(y_fit.sum()), "holdout_rows": len(hold_idx),
        "holdout_positives": int(y_hold.sum()),
        "original_head_holdout_auprc": original, "chosen": chosen,
        "candidates": {k: v[1] for k, v in candidates.items()},
        "seconds": round(time.monotonic() - started, 1),
    }
    out = dict(saved)
    out["risk_engine_state_dict"] = head.state_dict()
    out["head_refit"] = {k: report[k] for k in (
        "checkpoint", "holdout_fraction", "chosen", "original_head_holdout_auprc")}
    out["head_refit"]["holdout_auprc"] = info["holdout_auprc"]
    torch.save(out, args.out_checkpoint)
    report_path = Path(args.report or f"reports/{Path(args.out_checkpoint).stem}_head_refit.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Chọn %s (AUPRC tập giữ lại %.4f, đầu cũ %.4f) -> %s; báo cáo %s", chosen,
                info["holdout_auprc"], original, args.out_checkpoint, report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
