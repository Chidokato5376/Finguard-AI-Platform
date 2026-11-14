"""Chấm DRRE CÓ MANG TRẠNG THÁI Behavior Memory — đúng như lúc huấn luyện.

=== VÌ SAO CÓ FILE NÀY ===

`src/dashboard/scoring_service.py::score_batch` khởi tạo (h, m) về 0 ở MỖI lô.
Hợp lý cho dashboard, nhưng khi `scripts/evaluate.py` dùng nó để chấm checkpoint
huấn luyện với `--carry-state` thì nhánh Behavior Memory gần như bị tắt: mô hình
được HỌC với bộ nhớ dài suốt tuyến thời gian, rồi bị CHẤM với bộ nhớ trống. Mọi
số DRRE trong reports/eval/ trước ngày 2026-11-09 đều đo theo cách đó.

File này chạy đúng đường của `src/drre/train.py::evaluate_auprc`: quét tập trước
đó theo thứ tự thời gian để nạp trạng thái ("làm nóng"), rồi chấm tập cần đo.

=== CHẠY TIẾP ĐƯỢC ===

Quét 22 triệu dòng mất gần một giờ và tiến trình có thể chết giữa chừng (máy
ngủ, gập máy). Trạng thái được ghi nguyên tử mỗi `checkpoint_every` cửa sổ cùng
dấu vân tay của checkpoint và dữ liệu; dấu vân tay lệch thì quét lại từ đầu.
Trọng số cố định trong lúc quét, nên chạy tiếp cho kết quả giống hệt quét liền
(tests/test_stateful_scoring.py).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import torch

from src.dashboard.graph_utils import build_account_graph
from src.drre.past_graph import PastGraphBuffer
from src.drre.train import BehaviorStateStore, eval_chunk_bounds, run_encoder_over_window

logger = logging.getLogger(__name__)

#: 2: thêm bộ đệm đồ thị quá khứ (Giai đoạn 2B1) vào tệp làm nóng.
CACHE_FORMAT = 2
#: Cửa sổ giữa hai lần lưu. 100 cửa sổ ≈ 8 phút trên HI-Medium, mỗi lần ghi ≈ 3 GB.
DEFAULT_CHECKPOINT_EVERY = 100
#: Đúng kích thước cửa sổ đã dùng để huấn luyện và chấm val ở scripts/train_aml_medium.sh.
DEFAULT_WINDOW = 32768

OnWindow = Callable[[int, int, torch.Tensor, torch.Tensor], None]


def stream_columns(model, time_col: str = "timestamp") -> list[str]:
    """Các cột cần để quét: đặc trưng của checkpoint + định danh + thời gian."""
    return list(model.feature_columns) + ["account_id", "counterparty_id", time_col]


def fingerprint(checkpoint_path: str | Path, data_paths: Sequence[str | Path], window: int) -> dict:
    def stamp(path: str | Path) -> list:
        stat = os.stat(path)
        return [str(Path(path).name), stat.st_size, stat.st_mtime_ns]

    return {
        "format": CACHE_FORMAT, "window": int(window),
        "checkpoint": stamp(checkpoint_path), "data": [stamp(p) for p in data_paths],
    }


def _scan_window(
    model, df: pd.DataFrame, start: int, stop: int, store: BehaviorStateStore, device: torch.device,
    graph_buffer: PastGraphBuffer | None = None,
) -> torch.Tensor:
    window = df.iloc[start:stop].reset_index(drop=True)
    if graph_buffer is not None:
        graphs = graph_buffer.window_graphs(window)
        graph_buffer.append(window)
        out = run_encoder_over_window(
            model, window, None, None, None, device,
            feature_columns=list(model.feature_columns), state_store=store, graphs=graphs)
        return out["e"]
    node_features, edge_index, account_to_idx = build_account_graph(
        window, list(model.feature_columns), use_graph=getattr(model, "use_graph", True))
    out = run_encoder_over_window(
        model, window, node_features.to(device), edge_index.to(device), account_to_idx, device,
        feature_columns=list(model.feature_columns), state_store=store)
    return out["e"]


@torch.no_grad()
def warm_up_store(
    model,
    risk_engine,
    stream: pd.DataFrame,
    store: BehaviorStateStore,
    cache_path: str | Path | None,
    cache_id: dict,
    window: int = DEFAULT_WINDOW,
    checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY,
    on_window: OnWindow | None = None,
    device: torch.device | None = None,
    graph_buffer: PastGraphBuffer | None = None,
) -> None:
    """Quét `stream` (đã sắp theo thời gian) để nạp trạng thái vào `store`.

    Mô hình dùng đồ thị quá khứ cần `graph_buffer` (src.drre.past_graph.make_buffer);
    bộ đệm được lưu cùng tệp làm nóng và phải được truyền tiếp cho score_split.

    `on_window(start, stop, embedding, logit)` nhận kết quả từng cửa sổ, để
    xuất embedding. Hàm gọi lại phải ghi xong trước khi hàm này lưu tiến độ:
    khi chạy tiếp, các cửa sổ đã lưu sẽ không được gọi lại.
    """
    device = device or torch.device("cpu")
    bounds = eval_chunk_bounds(len(stream), window)
    next_window = 0
    cache_path = Path(cache_path) if cache_path else None

    if cache_path and cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        if cached.get("id") == cache_id:
            store.load_state_dict(cached["store"])
            if graph_buffer is not None:
                graph_buffer.load_state_dict(cached["graph"])
            next_window = int(cached["next_window"])
            logger.info("Chạy tiếp quét làm nóng từ cửa sổ %d/%d (%s)",
                        next_window, len(bounds), cache_path)
        else:
            logger.warning("%s không khớp checkpoint/dữ liệu hiện tại — quét lại từ đầu.", cache_path)

    def save(done: int) -> None:
        if cache_path is None:
            return
        tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
        torch.save({"id": cache_id, "next_window": done, "store": store.state_dict(),
                    "graph": None if graph_buffer is None else graph_buffer.state_dict()}, tmp)
        os.replace(tmp, cache_path)

    for i in range(next_window, len(bounds)):
        start, stop = bounds[i]
        embedding = _scan_window(model, stream, start, stop, store, device, graph_buffer)
        if on_window is not None:
            on_window(start, stop, embedding, risk_engine.logits(embedding))
        done = i + 1
        if done % 50 == 0:
            logger.info("Làm nóng: cửa sổ %d/%d | giữ lại %.1f%% tài khoản", done, len(bounds),
                        100 * store.hit_rate)
        if done % checkpoint_every == 0 and done < len(bounds):
            save(done)
    if next_window < len(bounds):
        save(len(bounds))


@torch.no_grad()
def score_split(
    model,
    risk_engine,
    df: pd.DataFrame,
    store: BehaviorStateStore,
    window: int = DEFAULT_WINDOW,
    on_window: OnWindow | None = None,
    device: torch.device | None = None,
    graph_buffer: PastGraphBuffer | None = None,
) -> np.ndarray:
    """Điểm 0–100 cho `df` (đã sắp theo thời gian), tiếp tục trạng thái trong `store`."""
    device = device or torch.device("cpu")
    parts = []
    for start, stop in eval_chunk_bounds(len(df), window):
        embedding = _scan_window(model, df, start, stop, store, device, graph_buffer)
        if on_window is not None:
            on_window(start, stop, embedding, risk_engine.logits(embedding))
        parts.append(risk_engine(embedding).cpu().numpy())
    return np.concatenate(parts).astype(np.float64)


def required_capacity(frames: Sequence[pd.DataFrame]) -> int:
    """Đủ chỗ cho mọi tài khoản gửi: bộ đệm không phải đẩy ai ra, giống lúc huấn luyện."""
    accounts: set = set()
    for frame in frames:
        accounts.update(pd.unique(frame["account_id"].astype(object)))
    return max(len(accounts), 1)


def time_sorted(df: pd.DataFrame) -> pd.DataFrame:
    time_col = "step" if "step" in df.columns else "timestamp"
    return df.sort_values(time_col, kind="stable").reset_index(drop=True)
