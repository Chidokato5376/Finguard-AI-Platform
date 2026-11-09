"""Training loop cho DRRE (Module 3′) — huấn luyện self-supervised +
contrastive, tùy chọn fine-tune bán giám sát (L_BCE) nếu có nhãn.

=== THIẾT KẾ MINI-BATCH THEO CỬA SỔ THỜI GIAN ===

Dữ liệu được chia thành các cửa sổ (mini-batch) gồm B giao dịch liên tiếp
theo thời gian (giống cách Dashboard đã batch dữ liệu — xem
src/dashboard/graph_utils.py). Trong mỗi cửa sổ:

  1. Graph Encoder dựng MỘT đồ thị dùng chung cho cả cửa sổ (giả định đồ
     thị quan hệ tài khoản ổn định trong khung thời gian ngắn — hợp lý
     với cửa sổ vài trăm giao dịch, KHÔNG hợp lý nếu áp dụng cho toàn bộ
     dataset).
  2. Behavior Memory chạy TUẦN TỰ theo từng account_id ring trong cửa sổ,
     khởi tạo lại (h=0, m=0) ở đầu mỗi cửa sổ — đây là điểm ĐƠN GIẢN HÓA
     có chủ đích cho MVP: không mang trạng thái Behavior Memory qua giữa
     các cửa sổ/epoch. Nhất quán với giới hạn cold-start đã ghi nhận ở
     Dashboard (src/dashboard/scoring_service.py). Cải tiến sau này: dùng
     một state store bền vững theo account_id xuyên suốt toàn bộ dataset
     (truncated BPTT thật) — nằm ngoài phạm vi MVP 6 tuần.
  3. L_pred cần cặp (giao dịch t, giao dịch t+1) CỦA CÙNG TÀI KHOẢN — chỉ
     tính được cho tài khoản xuất hiện ≥2 lần trong CÙNG một cửa sổ. Cửa
     sổ càng nhỏ, tỷ lệ tài khoản đủ điều kiện càng thấp — nếu log cảnh
     báo "0 cặp L_pred" xuất hiện thường xuyên, tăng --batch-size.
  4. L_contrast dùng in-batch negatives (giao dịch của tài khoản KHÁC
     trong cùng cửa sổ) — không cần negative sampling riêng.

=== HIỆU NĂNG (cập nhật 2026-10-08) ===

Encoder đã được VECTOR HÓA THEO BƯỚC THỜI GIAN (run_encoder_over_window):
ở bước t, mọi tài khoản có ít nhất t+1 giao dịch trong cửa sổ chạy trong
MỘT lần forward, nên số lần forward bằng độ dài chuỗi dài nhất chứ không
bằng số giao dịch. Bản tuần tự cũ được giữ lại làm chuẩn đối chiếu
(run_encoder_over_window_sequential) và chỉ dùng trong kiểm thử.

Đo thực tế trên máy 6 nhân (một bước huấn luyện đầy đủ, gồm cả backward):

    PaySim  cửa sổ 2048:  8,33s -> 0,29s  (29x)
    PaySim  cửa sổ 8192: 35,40s -> 0,84s  (42x)
    IEEE    cửa sổ 2048:  7,72s -> 0,28s  (28x)

Hệ quả: 5 epoch trên 2,1 triệu dòng đi từ ~12 giờ xuống ~20 phút, nên
huấn luyện trên toàn bộ PaySim (~6,3 triệu dòng) là khả thi.

Lấy mẫu negative cũng đã bỏ vòng lặp O(n²) (xem build_contrastive_triplets).
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import time
from collections import deque
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn

from src.dashboard.graph_utils import build_account_graph
from src.drre.losses import DRRECompositeLoss, EMATargetEncoder, prediction_loss
from src.drre.model import DRRE
from src.scoring.risk_scoring import RiskScoringEngine
from src.tracking import DEFAULT_RUNS_DIR, RunManifest, dataframe_summary, file_fingerprint

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = ["amount_norm", "channel_encoded"]

def features_for(
    source: str,
    override: Sequence[str] | None = None,
    available: Sequence[str] | None = None,
) -> list[str]:
    """Danh sách đặc trưng đầu vào cho một nguồn, có thể ghi đè từ dòng lệnh.

    NGUỒN DUY NHẤT của danh sách này là src/data/features.py — nơi sinh ra
    chính các cột đó. Hai danh sách ở hai file sẽ lệch nhau sau vài lần sửa,
    nên ở đây chỉ gọi lại, không khai báo lần hai.

    Dùng `--features amount_norm,channel_encoded` để tái lập các lần chạy cũ
    (trước 2026-10-17 mô hình chỉ nhận đúng 2 đặc trưng này).
    """
    if override:
        return list(override)
    from src.data.features import features_for_source

    wanted = features_for_source(source)
    if available is None:
        return wanted

    # Giao với các cột CÓ THẬT trong dữ liệu. Lý do: nhóm đặc trưng dòng tiền
    # (src/data/graph_features.py) chỉ xuất hiện trong parquet được tiền xử lý
    # từ 2026-10-24 trở đi. Nếu không giao ở đây, mọi parquet cũ sẽ chết bằng
    # KeyError giữa chừng thay vì chạy được với bộ đặc trưng cũ.
    have = [c for c in wanted if c in available]
    missing = [c for c in wanted if c not in available]
    if missing:
        logger.warning(
            "Dữ liệu thiếu %d/%d đặc trưng (%s) — chạy với %d đặc trưng còn lại. "
            "Tiền xử lý lại để có đủ nhóm đặc trưng dòng tiền.",
            len(missing), len(wanted), missing, len(have),
        )
    return have


def scored_mask(df: pd.DataFrame) -> np.ndarray:
    """Mặt nạ các dòng ĐƯỢC tính vào mất mát BCE và vào chỉ số đánh giá.

    Lấy từ cột `eval_universe` do preprocessing.add_eval_universe sinh ra.
    Nguồn/tập dữ liệu cũ chưa có cột này thì coi như chấm điểm MỌI dòng —
    giữ đúng hành vi trước đây, không âm thầm đổi kết quả của checkpoint cũ.
    """
    if "eval_universe" not in df.columns:
        return np.ones(len(df), dtype=bool)
    return df["eval_universe"].fillna(True).to_numpy(dtype=bool)


#: Số tài khoản tối đa được giữ trạng thái Behavior Memory giữa các cửa sổ.
#: Mỗi tài khoản tốn (hidden_dim + num_slots*hidden_dim) số thực = 320 số với
#: cấu hình mặc định ≈ 1,3 KB, nên 150.000 tài khoản ≈ 192 MB cho mỗi store.
#: Không giữ được toàn bộ: PaySim có 8,4 triệu tài khoản (≈ 10,8 GB).
DEFAULT_STATE_CACHE = 150_000


class BehaviorStateStore:
    """Giữ trạng thái (h, m) của Behavior Memory GIỮA các cửa sổ, theo account_id.

    === VÌ SAO CẦN ===

    Trước 2026-10-17, `run_encoder_over_window` gọi `init_state` ở MỖI cửa sổ,
    nên "bộ nhớ hành vi" chỉ dài bằng một cửa sổ. Đo thực tế: với cửa sổ
    32.768, chỉ 2,1% giao dịch gian lận của PaySim có giao dịch trước đó của
    cùng tài khoản NẰM TRONG cùng cửa sổ (8192 → 0,7%). Nói cách khác,
    Behavior Memory bị vô hiệu hóa bởi cách chia lô, không phải do thiết kế
    sai — và kết quả ablation âm trước đây (bỏ Behavior Memory thì AUPRC tăng,
    KTC 95% [−0,016; −0,003]) KHÔNG phải phán quyết về ý tưởng.

    === HAI HỆ QUẢ PHẢI BIẾT ===

    1. Mang trạng thái qua cửa sổ CHỈ đúng khi các cửa sổ được xử lý THEO
       THỨ TỰ THỜI GIAN. Nếu xáo trộn thứ tự cửa sổ (mặc định khi huấn luyện),
       một cửa sổ ở thời điểm T sẽ nhận trạng thái của thời điểm T+k — rò rỉ
       tương lai. Vì vậy train() TẮT xáo trộn khi bật tính năng này, và đánh
       đổi đó được ghi vào manifest.
    2. Trạng thái nạp vào đã `detach()`, nên gradient KHÔNG truyền về các cửa
       sổ trước (truncated BPTT). Đây là lựa chọn có ý thức: giữ đồ thị đạo
       hàm qua hàng trăm cửa sổ là bất khả thi về bộ nhớ.

    Bộ nhớ có giới hạn cứng `capacity`, thay thế theo kiểu vào trước ra trước.
    Tài khoản bị loại khỏi bộ đệm sẽ quay về trạng thái khởi tạo — chấp nhận
    được, vì tài khoản bị loại là tài khoản lâu không xuất hiện.
    """

    def __init__(self, hidden_dim: int, num_slots: int, capacity: int = DEFAULT_STATE_CACHE):
        self.capacity = int(capacity)
        self._h = torch.zeros(self.capacity, hidden_dim)
        self._m = torch.zeros(self.capacity, num_slots, hidden_dim)
        self._slot_of: dict[object, int] = {}
        self._owner: list[object | None] = [None] * self.capacity
        self._cursor = 0
        self.hits = 0
        self.lookups = 0
        self.evictions = 0

    def reset(self) -> None:
        """Xóa toàn bộ trạng thái — PHẢI gọi ở đầu mỗi epoch, vì epoch mới
        chạy lại dòng dữ liệu từ thời điểm đầu; giữ trạng thái của cuối epoch
        trước là đưa thông tin tương lai vào cửa sổ đầu tiên."""
        self._slot_of.clear()
        self._owner = [None] * self.capacity
        self._cursor = 0
        self._h.zero_()
        self._m.zero_()
        self.hits = self.lookups = self.evictions = 0

    def state_dict(self) -> dict:
        """Toàn bộ trạng thái để lưu xuống đĩa — dùng khi quét dài cần chạy tiếp."""
        return {
            "capacity": self.capacity, "h": self._h, "m": self._m,
            "slot_of": self._slot_of, "owner": self._owner, "cursor": self._cursor,
            "hits": self.hits, "lookups": self.lookups, "evictions": self.evictions,
        }

    def load_state_dict(self, state: dict) -> None:
        if int(state["capacity"]) != self.capacity:
            raise ValueError(
                f"Bộ đệm đã lưu có capacity={state['capacity']}, bộ đệm hiện tại "
                f"capacity={self.capacity} — không nạp được.")
        self._h, self._m = state["h"], state["m"]
        self._slot_of, self._owner = state["slot_of"], state["owner"]
        self._cursor = int(state["cursor"])
        self.hits, self.lookups, self.evictions = (
            int(state["hits"]), int(state["lookups"]), int(state["evictions"]))

    @property
    def hit_rate(self) -> float:
        return self.hits / self.lookups if self.lookups else 0.0

    def _allocate(self, account: object) -> int:
        slot = self._cursor
        self._cursor = (self._cursor + 1) % self.capacity
        previous = self._owner[slot]
        if previous is not None:
            del self._slot_of[previous]
            self.evictions += 1
        self._owner[slot] = account
        self._slot_of[account] = slot
        return slot

    def load(
        self, accounts: Sequence[object], h_state: torch.Tensor, m_state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Ghi đè trạng thái đã lưu vào h_state/m_state cho các tài khoản đã gặp."""
        rows, slots = [], []
        for i, account in enumerate(accounts):
            slot = self._slot_of.get(account)
            if slot is not None:
                rows.append(i)
                slots.append(slot)
        self.lookups += len(accounts)
        self.hits += len(rows)
        if not rows:
            return h_state, m_state
        idx = torch.as_tensor(rows, dtype=torch.long, device=h_state.device)
        sl = torch.as_tensor(slots, dtype=torch.long)
        return (
            h_state.index_copy(0, idx, self._h[sl].to(h_state.device)),
            m_state.index_copy(0, idx, self._m[sl].to(m_state.device)),
        )

    def save(
        self, accounts: Sequence[object], h_state: torch.Tensor, m_state: torch.Tensor
    ) -> None:
        """Lưu trạng thái CUỐI CÙNG của cửa sổ cho từng tài khoản."""
        slots = np.empty(len(accounts), dtype=np.int64)
        for i, account in enumerate(accounts):
            slot = self._slot_of.get(account)
            slots[i] = self._allocate(account) if slot is None else slot
        sl = torch.as_tensor(slots, dtype=torch.long)
        self._h.index_copy_(0, sl, h_state.detach().to("cpu", torch.float32))
        self._m.index_copy_(0, sl, m_state.detach().to("cpu", torch.float32))

#: Số vòng lấy mẫu lại khi mẫu negative trùng tài khoản với anchor — xem
#: build_contrastive_triplets. Xác suất trùng ≈ 1/số tài khoản trong cửa sổ,
#: nên vài vòng là hết, trừ khi cửa sổ bị một tài khoản chiếm gần hết.
MAX_NEGATIVE_REJECTION_ROUNDS = 8

#: Độ rộng cửa sổ trung bình trượt dùng khi chọn checkpoint (selection="smoothed").
#: Phải là số lẻ để cửa sổ có tâm. Xem select_epoch.
SMOOTH_WINDOW = 3

#: Các cách chọn checkpoint tốt nhất theo val AUPRC. Xem select_epoch.
SELECTION_MODES = ("smoothed", "argmax")

#: Phiên bản định dạng tệp trạng thái chạy tiếp (xem resume_state_path).
#: Tăng số này khi đổi cấu trúc tệp — tệp cũ sẽ bị từ chối thay vì nạp sai.
RESUME_FORMAT = 1

#: Ghi log tiến độ sau mỗi bấy nhiêu cửa sổ huấn luyện. Một epoch HI-Medium
#: có ~680 cửa sổ trong vài giờ — không có dòng này thì không biết còn bao lâu.
PROGRESS_EVERY = 25


# --------------------------------------------------------------------------
# Chạy tiếp theo từng epoch (chia một lần huấn luyện dài thành nhiều phiên)
# --------------------------------------------------------------------------
#
# Một epoch HI-Medium (22 triệu dòng) mất vài giờ trên CPU, nên 5–10 epoch
# không thể chạy trong một phiên làm việc. Ranh giới epoch là điểm dừng SẠCH
# duy nhất: BehaviorStateStore được reset ở đầu mỗi epoch, nên giữa hai epoch
# không có trạng thái tài khoản nào cần mang theo (tiết kiệm ~5 GB). Thứ cần
# lưu chỉ là trọng số, optimizer, mạng EMA, trạng thái RNG và lịch sử chọn
# checkpoint — đủ để phiên sau cho kết quả GIỐNG HỆT chạy liền một mạch
# (xem test_resume_matches_uninterrupted_run).


def resume_state_path(checkpoint_path: str | Path) -> Path:
    """Tệp trạng thái chạy tiếp đi kèm checkpoint: models/x.pt -> models/x.resume.pt."""
    p = Path(checkpoint_path)
    return p.with_name(p.stem + ".resume.pt")


def _save_resume_state(path: Path, state: dict) -> None:
    """Ghi nguyên tử: ghi ra tệp tạm rồi đổi tên, để mất điện giữa lúc ghi
    không làm hỏng trạng thái tốt của epoch trước."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def _resume_mismatches(saved: dict, current: dict) -> list[str]:
    """Các khóa có giá trị khác nhau giữa hai bộ siêu tham số/dấu vân tay dữ
    liệu. Chạy tiếp với cấu hình khác là ghép hai thí nghiệm khác nhau thành
    một manifest — phải từ chối, không được lặng lẽ chấp nhận."""
    keys = sorted(set(saved) | set(current))
    return [f"{k}: {saved.get(k)!r} -> {current.get(k)!r}"
            for k in keys if saved.get(k) != current.get(k)]


# --------------------------------------------------------------------------
# Batching theo cửa sổ thời gian
# --------------------------------------------------------------------------

def make_time_windows(df: pd.DataFrame, batch_size: int, time_col: str) -> list[pd.DataFrame]:
    """Chia dữ liệu (đã sort theo thời gian từ preprocessing.py) thành các
    cửa sổ liên tiếp không chồng lấn, mỗi cửa sổ batch_size giao dịch.

    ⚠ Hàm này VẬT CHẤT HÓA mọi cửa sổ cùng lúc, nên tổng bộ nhớ ≈ 2 lần
    kích thước `df`. Chỉ dùng cho tập nhỏ và cho kiểm thử. Vòng lặp huấn
    luyện dùng iter_time_windows() để không nhân đôi bộ nhớ — xem lý do
    trong docstring của hàm đó.
    """
    df_sorted = df.sort_values(time_col, kind="stable").reset_index(drop=True)
    windows = []
    for start in range(0, len(df_sorted), batch_size):
        window = df_sorted.iloc[start:start + batch_size].reset_index(drop=True)
        if len(window) >= 2:  # cửa sổ 1 dòng không tạo được cặp/negative nào có ích
            windows.append(window)
    return windows


def window_bounds(n_rows: int, batch_size: int) -> list[tuple[int, int]]:
    """Các khoảng [start, stop) của từng cửa sổ — chỉ là số, không phải dữ liệu."""
    bounds = []
    for start in range(0, n_rows, batch_size):
        stop = min(start + batch_size, n_rows)
        if stop - start >= 2:
            bounds.append((start, stop))
    return bounds


def iter_time_windows(
    df: pd.DataFrame, batch_size: int, time_col: str, rng: random.Random | None = None
):
    """Sinh từng cửa sổ MỘT LÚC MỘT, theo thứ tự ngẫu nhiên nếu có `rng`.

    Vì sao cần: bản make_time_windows giữ đồng thời toàn bộ cửa sổ trong
    một list, nên bộ nhớ gấp đôi kích thước tập train. Đo thực tế trên lát
    PaySim 3 triệu dòng (train 2,1 triệu) với máy RAM 7,4 GB: chỉ riêng
    bước dựng cửa sổ đã đẩy bộ nhớ trống xuống dưới 1 GB. Cách này chỉ cắt
    một cửa sổ tại thời điểm cần, nên bộ nhớ thêm vào không đáng kể.

    Trả về bộ (số cửa sổ, generator) để vòng lặp vẫn biết tổng số cửa sổ
    trước khi chạy — dùng cho log tiến độ.
    """
    df_sorted = df.sort_values(time_col, kind="stable").reset_index(drop=True)
    bounds = window_bounds(len(df_sorted), batch_size)
    order = list(range(len(bounds)))
    if rng is not None:
        rng.shuffle(order)

    def generate():
        for i in order:
            start, stop = bounds[i]
            yield df_sorted.iloc[start:stop].reset_index(drop=True)

    return len(bounds), generate()


# --------------------------------------------------------------------------
# Forward pass tuần tự theo account trong một cửa sổ
# --------------------------------------------------------------------------

def run_encoder_over_window_sequential(
    model: DRRE,
    window: pd.DataFrame,
    node_features: torch.Tensor,
    edge_index: torch.Tensor,
    account_to_idx: dict[str, int],
    device: torch.device,
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
) -> dict[str, torch.Tensor]:
    """BẢN THAM CHIẾU — chạy một lần forward cho MỖI giao dịch.

    Giữ lại vì đây là cách diễn đạt trực tiếp nhất ngữ nghĩa Behavior
    Memory, dùng làm chuẩn đối chiếu cho bản vector hóa trong
    tests/test_train_vectorized.py. KHÔNG dùng để huấn luyện: chậm hơn
    28–42 lần (đo thực tế), xem run_encoder_over_window.

    Trả về tensor đã align đúng thứ tự hàng gốc của `window` (RangeIndex
    0..n-1), để có thể ghép với nhãn/thông tin khác của window bằng index.
    """
    n = len(window)
    e_out: list[torch.Tensor | None] = [None] * n
    pred_out: list[torch.Tensor | None] = [None] * n

    # Tính embedding đồ thị MỘT lần cho cả cửa sổ (xem DRRE.forward).
    g_all = model.graph_encoder(node_features, edge_index)

    for account_id, group in window.groupby("account_id", sort=False, observed=True):
        idxs = group.index.tolist()  # đã sort theo group nhưng group giữ thứ tự gốc (đã sort theo time trước đó)
        h, m = model.behavior_memory.init_state(1, device=device)
        if account_id not in account_to_idx:
            # Tài khoản chỉ xuất hiện ở vai trò counterparty trong window,
            # chưa từng là sender -> không có trong đồ thị node -> bỏ qua
            # (không nên xảy ra vì account_id luôn là sender, nhưng kiểm
            # tra phòng thủ để lỗi rõ ràng thay vì KeyError khó hiểu).
            raise KeyError(
                f"account_id '{account_id}' không có trong account_to_idx — "
                "build_account_graph phải bao gồm mọi account_id làm node."
            )
        node_idx = torch.tensor([account_to_idx[account_id]], dtype=torch.long, device=device)

        for row_idx in idxs:
            x_t = torch.tensor(
                window.loc[row_idx, list(feature_columns)].to_numpy(dtype=np.float32),
                device=device,
            ).unsqueeze(0)
            out = model(x_t, h, m, node_features, edge_index, node_idx, g_all=g_all)
            h, m = out["h_t"], out["m_t"]
            e_out[row_idx] = out["e_t"].squeeze(0)
            pred_out[row_idx] = out["e_pred"].squeeze(0)

    return {"e": torch.stack(e_out), "pred": torch.stack(pred_out)}  # type: ignore[arg-type]


def run_encoder_over_window(
    model: DRRE,
    window: pd.DataFrame,
    node_features: torch.Tensor,
    edge_index: torch.Tensor,
    account_to_idx: dict[str, int],
    device: torch.device,
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
    state_store: BehaviorStateStore | None = None,
) -> dict[str, torch.Tensor]:
    """Chạy DRRE cho một cửa sổ theo BƯỚC THỜI GIAN thay vì theo từng dòng.

    Cùng một phép tính như run_encoder_over_window_sequential: mỗi tài
    khoản vẫn đi qua GRU đúng thứ tự thời gian của nó. Khác biệt duy nhất
    là cách xếp lô — ở bước t, TẤT CẢ tài khoản có ít nhất t+1 giao dịch
    trong cửa sổ được chạy trong MỘT lần forward.

    Vì sao đáng làm: số bước bằng độ dài chuỗi DÀI NHẤT trong cửa sổ, chứ
    không bằng số giao dịch. Đo thực tế trên cửa sổ 2048 dòng: PaySim có
    chuỗi dài nhất 2 giao dịch (2 bước thay vì 2048 lần forward),
    IEEE-CIS dài nhất 22. Kết quả: một bước huấn luyện nhanh hơn 28–42
    lần, sai số so với bản tham chiếu ~4·10⁻⁶ (nhiễu dấu phẩy động) —
    xem tests/test_train_vectorized.py.

    Trả về tensor đã align đúng thứ tự hàng gốc của `window`, giống hệt
    bản tham chiếu.
    """
    accounts = window["account_id"].to_numpy()
    codes, uniques = pd.factorize(pd.Series(accounts), sort=False)
    # Vị trí của mỗi giao dịch trong chuỗi của chính tài khoản nó. window đã
    # sắp theo thời gian từ make_time_windows, nên cumcount = thứ tự thời gian.
    # `observed=True` KHÔNG phải tinh chỉnh vi mô — nó là khác biệt giữa chạy
    # được và không chạy được. Loader AMLWorld để `account_id` là Categorical
    # với ~497.000 category; với mặc định `observed=False`, pandas tạo một
    # nhóm cho MỌI category kể cả tài khoản không hề có mặt trong cửa sổ, nên
    # mỗi cửa sổ 32.768 dòng phải duyệt 497.000 nhóm thay vì ~30.000. Đo được
    # (2026-10-20): 342 giây cho 200.000 dòng, tức ~1,4 giờ/epoch và 28 giờ
    # cho 20 epoch. Thiếu đúng một tham số này là mất gần hai ngày máy.
    position = window.groupby("account_id", sort=False, observed=True).cumcount().to_numpy()

    missing = [a for a in uniques if a not in account_to_idx]
    if missing:
        raise KeyError(
            f"{len(missing)} account_id không có trong account_to_idx (vd. {missing[:3]}) — "
            "build_account_graph phải bao gồm mọi account_id làm node."
        )

    x_all = torch.as_tensor(
        window[list(feature_columns)].to_numpy(dtype=np.float32), device=device)
    node_idx_by_account = torch.as_tensor(
        np.array([account_to_idx[a] for a in uniques], dtype=np.int64), device=device)
    codes_t = torch.as_tensor(codes.astype(np.int64), device=device)

    # Graph Encoder chỉ phụ thuộc đồ thị của cửa sổ -> chạy một lần.
    g_all = model.graph_encoder(node_features, edge_index)
    h_state, m_state = model.behavior_memory.init_state(len(uniques), device=device)
    if state_store is not None:
        h_state, m_state = state_store.load(uniques, h_state, m_state)

    e_parts: list[torch.Tensor] = []
    pred_parts: list[torch.Tensor] = []
    row_parts: list[torch.Tensor] = []

    for step in range(int(position.max()) + 1):
        rows = torch.as_tensor(
            np.flatnonzero(position == step), dtype=torch.long, device=device)
        acc = codes_t[rows]
        out = model(
            x_all[rows], h_state[acc], m_state[acc],
            node_features, edge_index, node_idx_by_account[acc], g_all=g_all,
        )
        # index_copy KHÔNG tại chỗ: giữ đúng đồ thị tính đạo hàm qua các bước
        h_state = h_state.index_copy(0, acc, out["h_t"])
        m_state = m_state.index_copy(0, acc, out["m_t"])
        e_parts.append(out["e_t"])
        pred_parts.append(out["e_pred"])
        row_parts.append(rows)

    if state_store is not None:
        state_store.save(uniques, h_state, m_state)

    rows_all = torch.cat(row_parts)
    inverse = torch.empty_like(rows_all)
    inverse[rows_all] = torch.arange(rows_all.numel(), device=device)
    return {"e": torch.cat(e_parts)[inverse], "pred": torch.cat(pred_parts)[inverse]}


# --------------------------------------------------------------------------
# Lấy cặp (t, t+1) cho L_pred và (anchor, positive, negatives) cho L_contrast
# --------------------------------------------------------------------------

def build_prediction_pairs(window: pd.DataFrame) -> tuple[list[int], list[int]]:
    """Cặp (nguồn, đích) trong CÙNG account, liên tiếp trong cửa sổ —
    nguồn dùng predictor(online), đích dùng target network (EMA)."""
    src, tgt = [], []
    for _, group in window.groupby("account_id", sort=False, observed=True):
        idxs = group.index.tolist()
        for i in range(len(idxs) - 1):
            src.append(idxs[i])
            tgt.append(idxs[i + 1])
    return src, tgt


def build_contrastive_triplets(
    window: pd.DataFrame, num_negatives: int, rng: random.Random
) -> tuple[list[int], list[int], list[list[int]]]:
    """anchor/positive: hai giao dịch khác nhau CÙNG account trong cửa sổ.
    negative: num_negatives giao dịch từ các account KHÁC trong cửa sổ
    (in-batch negatives).
    """
    account_of = window["account_id"].to_numpy()
    anchors, positives = [], []
    for _, group in window.groupby("account_id", sort=False, observed=True):
        idxs = group.index.tolist()
        if len(idxs) < 2:
            continue
        for i in idxs:
            candidates = [j for j in idxs if j != i]
            positives.append(rng.choice(candidates))
            anchors.append(i)

    if not anchors:
        return anchors, positives, []

    # Lấy mẫu negative bằng BÁC BỎ, vector hóa. Bản cũ dựng lại pool ứng
    # viên cho TỪNG anchor bằng vòng lặp Python -> O(n²): đo thực tế 42ms
    # mỗi cửa sổ trên PaySim và 301ms trên IEEE-CIS (cửa sổ 2048 dòng), và
    # tăng theo bình phương khi tăng batch-size.
    #
    # ⚠ KHÁC BIỆT CÓ CHỦ Ý so với bản cũ: bản cũ dùng rng.sample (không lặp
    # trong cùng một anchor), bản này lấy có lặp rồi bác bỏ các mẫu trùng
    # tài khoản. Cùng phân phối in-batch negatives, nhưng KHÔNG cho ra cùng
    # một dãy mẫu — nên kết quả huấn luyện không trùng khít bản cũ theo
    # từng bit. Phần encoder thì trùng khít (xem run_encoder_over_window).
    anchor_arr = np.asarray(anchors)
    n = len(window)
    gen = np.random.default_rng(rng.randrange(2 ** 32))
    neg = gen.integers(0, n, size=(anchor_arr.size, num_negatives))
    collide = account_of[neg] == account_of[anchor_arr][:, None]
    for _ in range(MAX_NEGATIVE_REJECTION_ROUNDS):
        if not collide.any():
            break
        neg[collide] = gen.integers(0, n, size=int(collide.sum()))
        collide = account_of[neg] == account_of[anchor_arr][:, None]

    if collide.any():
        # Cửa sổ bị một tài khoản chiếm gần hết -> không còn negative hợp lệ.
        # Giữ nguyên hành vi suy biến của bản cũ: dùng chính anchor.
        logger.warning(
            "Còn %d mẫu negative trùng tài khoản sau %d vòng bác bỏ — cửa sổ gần như "
            "chỉ có một tài khoản. Dùng chính anchor làm negative (suy biến).",
            int(collide.sum()), MAX_NEGATIVE_REJECTION_ROUNDS,
        )
        neg[collide] = np.repeat(anchor_arr[:, None], num_negatives, axis=1)[collide]

    return anchors, positives, neg.tolist()


# --------------------------------------------------------------------------
# Một bước huấn luyện trên một cửa sổ
# --------------------------------------------------------------------------

def train_on_window(
    model: DRRE,
    risk_engine: RiskScoringEngine,
    target: EMATargetEncoder,
    loss_fn: DRRECompositeLoss,
    optimizer: torch.optim.Optimizer,
    window: pd.DataFrame,
    device: torch.device,
    num_negatives: int,
    rng: random.Random,
    use_graph: bool = True,
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
    online_store: BehaviorStateStore | None = None,
    target_store: BehaviorStateStore | None = None,
) -> dict[str, float]:
    node_features, edge_index, account_to_idx = build_account_graph(
        window, list(feature_columns), use_graph=use_graph)
    node_features, edge_index = node_features.to(device), edge_index.to(device)

    # Online và target network có trọng số KHÁC nhau nên trạng thái của chúng
    # cũng khác — phải dùng hai store riêng, nếu dùng chung thì cái sau ghi đè
    # cái trước và cả hai đều nhận trạng thái sai.
    online_out = run_encoder_over_window(
        model, window, node_features, edge_index, account_to_idx, device,
        feature_columns=feature_columns, state_store=online_store)

    with torch.no_grad():
        target_out = run_encoder_over_window(
            target.target_encoder, window, node_features, edge_index, account_to_idx, device,
            feature_columns=feature_columns, state_store=target_store,
        )

    pred_src, pred_tgt = build_prediction_pairs(window)
    if pred_src:
        e_pred_sel = online_out["pred"][pred_src]
        e_target_sel = target_out["e"][pred_tgt].detach()
    else:
        # Không có tài khoản nào lặp lại trong cửa sổ -> không tính được
        # L_pred cho bước này. Trả về loss=0 (không đóng góp gradient),
        # KHÔNG che giấu — được log ở train() để người dùng biết tăng batch_size.
        e_pred_sel = torch.zeros(1, model.fusion.phi_1.out_features, device=device)
        e_target_sel = e_pred_sel.detach().clone()

    anchors, positives, neg_lists = build_contrastive_triplets(window, num_negatives, rng)
    if anchors:
        e_anchor = online_out["e"][anchors]
        e_pos = online_out["e"][positives]
        # Gom negative bằng một phép đánh chỉ số 2 chiều thay vì stack theo
        # vòng lặp Python — cùng kết quả, không phụ thuộc số anchor.
        neg_idx = torch.as_tensor(
            np.asarray(neg_lists, dtype=np.int64), dtype=torch.long, device=device)
        e_neg = online_out["e"][neg_idx]
    else:
        dim = online_out["e"].shape[-1]
        e_anchor = torch.zeros(1, dim, device=device)
        e_pos = torch.zeros(1, dim, device=device)
        e_neg = torch.zeros(1, num_negatives, dim, device=device)

    bce_logits, bce_labels = None, None
    if "label" in window.columns:
        # Chỉ tính BCE trên vũ trụ chấm điểm. Với PaySim, 31% nhãn dương nằm
        # trong các giờ mà mô hình KHÔNG có đặc trưng nào để phân biệt (không
        # có đặc trưng giờ), nên nếu tính cả thì đó là nhãn không học được —
        # chỉ thêm nhiễu vào gradient. Xem preprocessing.add_eval_universe.
        valid = window["label"].notna() & pd.Series(scored_mask(window), index=window.index)
        if valid.any():
            valid_idx = window.index[valid].tolist()
            bce_logits = risk_engine.logits(online_out["e"][valid_idx])
            bce_labels = torch.tensor(
                window.loc[valid_idx, "label"].astype(float).to_numpy(), device=device
            )

    losses = loss_fn(
        e_pred=e_pred_sel, e_target=e_target_sel,
        e_anchor=e_anchor, e_pos=e_pos, e_neg=e_neg,
        bce_logits=bce_logits, bce_labels=bce_labels,
    )

    optimizer.zero_grad()
    losses["loss"].backward()
    torch.nn.utils.clip_grad_norm_(
        list(model.parameters()) + list(risk_engine.parameters()), max_norm=5.0
    )
    optimizer.step()
    target.update(model)

    return {
        "loss": float(losses["loss"].item()),
        "l_pred": float(losses["l_pred"].item()),
        "l_contrast": float(losses["l_contrast"].item()),
        "l_bce": float(losses["l_bce"].item()),
        "n_pred_pairs": len(pred_src),
        "n_contrastive_anchors": len(anchors),
        "n_bce_labels": 0 if bce_labels is None else len(bce_labels),
    }


# --------------------------------------------------------------------------
# Đánh giá trên tập val (nếu có nhãn) — AUPRC
# --------------------------------------------------------------------------

def eval_chunk_bounds(n_rows: int, batch_size: int) -> list[tuple[int, int]]:
    """Khoảng của từng lô khi CHẤM ĐIỂM — khác window_bounds ở chỗ KHÔNG
    được bỏ dòng nào, vì chấm điểm phải phủ toàn bộ tập.

    Nếu lô cuối chỉ còn 1 dòng thì gộp vào lô trước, để mọi lô đều có ít
    nhất 2 dòng (đủ cho việc dựng đồ thị).
    """
    if n_rows <= 0:
        return []
    bounds = [(start, min(start + batch_size, n_rows)) for start in range(0, n_rows, batch_size)]
    if len(bounds) >= 2 and bounds[-1][1] - bounds[-1][0] < 2:
        bounds[-2] = (bounds[-2][0], bounds[-1][1])
        bounds.pop()
    return bounds


@torch.no_grad()
def evaluate_auprc(
    model: DRRE, risk_engine: RiskScoringEngine, df_val: pd.DataFrame, device: torch.device,
    use_graph: bool = True, batch_size: int = 8192,
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
    state_store: BehaviorStateStore | None = None,
) -> float | None:
    """AUPRC trên tập val, chấm THEO LÔ cùng kích thước cửa sổ huấn luyện.

    Trước đây hàm này chấm cả tập trong MỘT lô. Hai vấn đề:

      1. Bộ nhớ: với lát PaySim 3 triệu dòng (val 450.000 dòng, hơn 700
         nghìn tài khoản), dựng một đồ thị cho cả tập làm tiến trình chết
         vì hết bộ nhớ (segfault, đo thực tế trên máy RAM 7,4 GB).
      2. Tính đúng đắn: đồ thị dựng trên CẢ tập val cho mỗi giao dịch
         "nhìn thấy" quan hệ hình thành ở tương lai của nó. Chấm theo lô
         theo thứ tự thời gian loại bỏ phần rò rỉ đó, nên số AUPRC mới sẽ
         THẤP HƠN số cũ với PaySim có đồ thị — đây là số trung thực hơn,
         không phải sự suy giảm của mô hình.
    """
    from src.evaluation.metrics import compute_auprc

    if "label" not in df_val.columns or df_val["label"].notna().sum() == 0:
        return None

    time_col = "step" if "step" in df_val.columns else "timestamp"
    df_sorted = df_val.sort_values(time_col, kind="stable").reset_index(drop=True)

    chunk_scores: list[np.ndarray] = []
    for start, stop in eval_chunk_bounds(len(df_sorted), batch_size):
        window = df_sorted.iloc[start:stop].reset_index(drop=True)
        node_features, edge_index, account_to_idx = build_account_graph(
            window, list(feature_columns), use_graph=use_graph)
        out = run_encoder_over_window(
            model, window, node_features.to(device), edge_index.to(device),
            account_to_idx, device, feature_columns=feature_columns,
            state_store=state_store)
        chunk_scores.append(risk_engine(out["e"]).cpu().numpy())

    scores = np.concatenate(chunk_scores) if chunk_scores else np.empty(0)
    # Điểm được chấm cho MỌI dòng (đồ thị và lịch sử cần cả các dòng ngoài vũ
    # trụ), nhưng chỉ số chỉ tính trên vũ trụ chấm điểm — đúng như lúc train.
    valid = df_sorted["label"].notna() & pd.Series(
        scored_mask(df_sorted), index=df_sorted.index)
    y_true = df_sorted.loc[valid, "label"].astype(int).to_numpy()
    y_score = scores[valid.to_numpy()]
    if len(set(y_true.tolist())) < 2:
        logger.warning("Tập val chỉ có 1 lớp nhãn duy nhất — không tính được AUPRC có ý nghĩa.")
        return None
    return compute_auprc(y_true, y_score)


# --------------------------------------------------------------------------
# Vòng lặp huấn luyện chính
# --------------------------------------------------------------------------

def load_drre_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, encoding="utf-8") as f:
        full_config = yaml.safe_load(f)
    return full_config.get("drre", {})


# --------------------------------------------------------------------------
# Chọn checkpoint
# --------------------------------------------------------------------------

def select_epoch(
    val_auprc: Sequence[float],
    selection: str = "smoothed",
    window: int = SMOOTH_WINDOW,
) -> tuple[int, float, float] | None:
    """Chọn epoch tốt nhất từ dãy val AUPRC (phần tử 0 ứng với epoch 1).

    - ``"argmax"``: lấy epoch có val AUPRC cao nhất. Đây là cách cũ, và nó
      CHỌN CẢ NHIỄU: nếu đường val dao động mạnh thì giá trị cao nhất một
      phần là may mắn, nên con số val của epoch được chọn bị thiên lệch
      lạc quan (đo trên PaySim 3M: biên độ giữa hai epoch liền nhau tới
      0,14 AUPRC khi val chỉ có 262 nhãn dương và pos_weight = 988,6).
    - ``"smoothed"``: chấm mỗi epoch bằng ``min(trung bình trượt CÓ TÂM window
      epoch, val AUPRC của chính epoch đó)`` rồi lấy epoch có điểm cao nhất.
      Hai thành phần chặn hai kiểu nhiễu khác nhau và cần CẢ HAI:

        * trung bình trượt chặn đỉnh đơn lẻ — một epoch chỉ thắng khi hai
          epoch kề nó cũng tốt;
        * phép ``min`` chặn epoch ĐỨNG KỀ một đỉnh đơn lẻ. Không có nó, dãy
          ``[0.10, 0.11, 0.90, 0.12, 0.40, 0.42, 0.41]`` sẽ chọn epoch 4 (val
          thô 0,12!) vì trung bình trượt của nó được epoch 3 kéo lên 0,47.
          Nguyên tắc: KHÔNG epoch nào được điểm cao hơn mức chính nó đã
          thể hiện.

      Giá phải trả: ``window // 2`` epoch ở mỗi đầu KHÔNG được chọn (không đủ
      lân cận để biết chúng có phải đỉnh nhiễu hay không).

    Nếu có ít hơn ``window`` giá trị thì ``"smoothed"`` rơi về ``"argmax"``,
    vì không tính được trung bình trượt nào.

    Returns:
        ``(epoch 1-based, val AUPRC thô của epoch đó, điểm dùng để chọn)``,
        hoặc None nếu ``val_auprc`` rỗng. Với ``"argmax"`` thì hai số sau
        bằng nhau.
    """
    if selection not in SELECTION_MODES:
        raise ValueError(f"selection phải thuộc {SELECTION_MODES}, nhận {selection!r}")
    values = list(val_auprc)
    if not values:
        return None
    if selection == "argmax" or len(values) < window:
        i = max(range(len(values)), key=lambda k: values[k])
        return i + 1, values[i], values[i]
    half = window // 2
    best_i, best_score = -1, float("-inf")
    for i in range(half, len(values) - half):
        score = min(sum(values[i - half: i + half + 1]) / window, values[i])
        if score > best_score:
            best_i, best_score = i, score
    return best_i + 1, values[best_i], best_score


def _cpu_state(module: nn.Module) -> dict:
    """Bản sao state_dict trên CPU — để giữ lại trọng số của một epoch cũ mà
    không giữ tham chiếu tới tensor đang được optimizer cập nhật tại chỗ."""
    return {k: v.detach().to("cpu").clone() for k, v in module.state_dict().items()}


def _online_candidate(
    snapshots: Sequence[tuple[int, float, dict, dict]], selection: str, window: int = SMOOTH_WINDOW,
) -> tuple[int, float, float, dict, dict] | None:
    """Ứng viên checkpoint MỚI sinh ra sau epoch vừa chấm, tính từ tối đa
    ``window`` snapshot gần nhất.

    Gọi hàm này sau mỗi epoch rồi giữ lại ứng viên có điểm cao nhất thì được
    đúng kết quả của select_epoch chạy trên toàn bộ dãy — xem
    test_online_selection_matches_select_epoch. Lý do phải làm trực tuyến:
    trọng số của epoch được chọn phải còn trong bộ nhớ lúc ghi checkpoint.

    Returns:
        ``(epoch, điểm chọn, val AUPRC thô, state_dict model, state_dict risk_engine)``
        hoặc None khi chưa đủ dữ liệu để sinh ứng viên.
    """
    if not snapshots:
        return None
    if selection == "argmax":
        epoch, auprc, model_sd, risk_sd = snapshots[-1]
        return epoch, auprc, auprc, model_sd, risk_sd
    if len(snapshots) < window:
        return None
    mid = snapshots[len(snapshots) // 2]
    score = min(sum(s[1] for s in snapshots) / len(snapshots), mid[1])
    return mid[0], score, mid[1], mid[2], mid[3]


def _mark_saved(epochs_recorded: list[dict], epoch_record: dict, epoch: int) -> None:
    """Đánh dấu epoch nào đã từng được ghi ra checkpoint. Nhiều epoch có thể
    mang dấu này (mỗi lần điểm chọn tốt hơn là một lần ghi đè) — epoch CUỐI
    CÙNG thắng nằm ở manifest.results["best_epoch"]."""
    if epoch_record.get("epoch") == epoch:
        epoch_record["saved_checkpoint"] = True
        return
    for rec in epochs_recorded:
        if rec.get("epoch") == epoch:
            rec["saved_checkpoint"] = True
            return


def train(
    source: str,
    epochs: int = 5,
    batch_size: int = 128,
    lr: float = 1e-3,
    num_negatives: int = 5,
    config_path: str = "config/config.yaml",
    processed_dir: str = "data/processed",
    checkpoint_path: str = "models/drre_checkpoint.pt",
    seed: int = 42,
    use_graph: bool = True,
    disabled_branches: list[str] | None = None,
    runs_dir: str | Path | None = DEFAULT_RUNS_DIR,
    selection: str = "smoothed",
    feature_columns: Sequence[str] | None = None,
    carry_state: bool = False,
    state_cache: int = DEFAULT_STATE_CACHE,
    lambda_bce: float | None = None,
    resume: bool = False,
    stop_after: int | None = None,
) -> None:
    """Huấn luyện DRRE.

    Args:
        runs_dir: thư mục ghi manifest thí nghiệm. Truyền None để KHÔNG ghi
            — kiểm thử phải dùng None hoặc một thư mục tạm, nếu không mỗi
            lần chạy pytest sẽ để lại rác trong reports/runs/ của dự án.
        selection: cách chọn checkpoint theo val AUPRC — "smoothed" (mặc định,
            trung bình trượt 3 epoch) hoặc "argmax" (hành vi cũ). Xem select_epoch.
        resume: chạy tiếp từ tệp `resume_state_path(checkpoint_path)` nếu có
            (chưa có thì bắt đầu mới). Trạng thái được ghi sau MỖI epoch, kể cả
            khi không bật cờ này, nên một lần chạy bị ngắt luôn chạy tiếp được.
        stop_after: dừng sau bấy nhiêu epoch trong phiên này (None = chạy hết).
            Manifest chỉ được ghi khi đã chạy đủ `epochs`.
    """
    if selection not in SELECTION_MODES:
        raise ValueError(f"selection phải thuộc {SELECTION_MODES}, nhận {selection!r}")
    if stop_after is not None and stop_after < 1:
        raise ValueError(f"stop_after phải >= 1, nhận {stop_after}")
    resume_path = resume_state_path(checkpoint_path)
    resume_state = None
    if resume_path.exists():
        if not resume:
            raise FileExistsError(
                f"Đã có trạng thái chạy dở {resume_path}. Thêm --resume để chạy tiếp, hoặc "
                "xóa tệp đó nếu thật sự muốn huấn luyện lại từ đầu (sẽ mất các epoch đã chạy).")
        resume_state = torch.load(resume_path, map_location="cpu")
        if resume_state.get("format") != RESUME_FORMAT:
            raise ValueError(
                f"{resume_path} có định dạng {resume_state.get('format')!r}, mã hiện tại cần "
                f"{RESUME_FORMAT} — không nạp được.")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    torch.manual_seed(seed)
    rng = random.Random(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Thiết bị huấn luyện: %s", device)

    drre_cfg = load_drre_config(config_path)
    hidden_dim = drre_cfg.get("behavior_memory", {}).get("hidden_dim", 64)
    num_slots = drre_cfg.get("behavior_memory", {}).get("num_slots", 4)
    graph_hidden_dim = drre_cfg.get("graph_encoder", {}).get("hidden_dim", 64)
    graph_num_layers = drre_cfg.get("graph_encoder", {}).get("num_layers", 2)
    embedding_dim = drre_cfg.get("fusion", {}).get("embedding_dim", 128)
    ema_decay = drre_cfg.get("loss", {}).get("ema_decay", 0.996)
    lambda_contrast = drre_cfg.get("loss", {}).get("lambda_contrast", 0.5)
    # Trọng số của nhánh có giám sát. Mặc định trong config là 0,3 — thấp hơn
    # lambda_contrast (0,5), nên tín hiệu NHÃN bị lấn bởi hai mất mát tự giám
    # sát. Đây là một nghi vấn cần đo, không phải hằng số bất biến: trên
    # IEEE-CIS, Logistic với đúng 2 đặc trưng đạt 0,1010 còn DRRE chỉ 0,0900-0,0970,
    # và risk_engine CHỈ học được qua số hạng BCE này.
    if lambda_bce is None:
        lambda_bce = drre_cfg.get("loss", {}).get("lambda_bce", 0.3)
    temperature = drre_cfg.get("loss", {}).get("contrastive_temperature", 0.1)

    train_path = Path(processed_dir) / f"{source}_train.parquet"
    val_path = Path(processed_dir) / f"{source}_val.parquet"
    if not train_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {train_path}. Chạy trước: python -m src.data.preprocessing "
            f"--source {source} --raw-path data/raw/{source}.csv --output {processed_dir}"
        )
    df_train = pd.read_parquet(train_path)
    df_val = pd.read_parquet(val_path) if val_path.exists() else None

    feature_cols = features_for(source, feature_columns, available=df_train.columns)
    missing = set(feature_cols) - set(df_train.columns)
    if missing:
        raise ValueError(f"Dữ liệu thiếu cột đặc trưng: {missing}. Chạy lại preprocessing.py.")
    if not feature_cols:
        raise ValueError(
            f"Không còn đặc trưng nào dùng được cho nguồn '{source}'. Chạy lại preprocessing.py.")
    time_col = "step" if "step" in df_train.columns else "timestamp"
    logger.info("Đặc trưng đầu vào (%d): %s", len(feature_cols), feature_cols)

    # Vũ trụ chấm điểm: bao nhiêu dòng/nhãn dương thực sự đi vào mất mát BCE.
    universe = scored_mask(df_train)
    if not universe.all():
        n_pos_all = int(df_train["label"].sum()) if "label" in df_train.columns else 0
        n_pos_universe = int(df_train.loc[universe, "label"].sum()) if n_pos_all else 0
        logger.info(
            "Vũ trụ chấm điểm: %d/%d dòng (%.1f%%), %d/%d nhãn dương. Các dòng ngoài vũ trụ "
            "VẪN được dùng để dựng lịch sử và đồ thị, chỉ không tính vào BCE/chỉ số.",
            int(universe.sum()), len(df_train), 100 * universe.mean(),
            n_pos_universe, n_pos_all,
        )

    input_dim = len(feature_cols)
    model = DRRE(
        input_dim=input_dim, hidden_dim=hidden_dim, num_slots=num_slots,
        graph_hidden_dim=graph_hidden_dim, graph_num_layers=graph_num_layers,
        embedding_dim=embedding_dim,
    ).to(device)
    risk_engine = RiskScoringEngine(embedding_dim=embedding_dim).to(device)

    # Tắt nhánh PHẢI làm trước khi tạo target network (EMA sao chép cả mặt nạ).
    disabled = set(disabled_branches or [])
    if not use_graph:
        disabled.add("graph")
    model.fusion.set_disabled_branches(disabled)
    if disabled:
        logger.info("Tắt nhánh: %s | dựng đồ thị: %s", sorted(disabled), use_graph)
    target = EMATargetEncoder(model, decay=ema_decay)

    # pos_weight phải tính trên ĐÚNG tập dòng mà BCE nhìn thấy (vũ trụ chấm
    # điểm), không phải trên cả tập train — hai tỷ lệ này lệch 1,5 lần với
    # PaySim, nên dùng sai sẽ đặt sai trọng số lớp dương.
    pos_rate = (
        float(df_train.loc[universe, "label"].mean()) if "label" in df_train.columns else 0.0
    )
    pos_weight = (1.0 - pos_rate) / pos_rate if 0.0 < pos_rate < 1.0 else None
    if pos_weight is not None:
        logger.info(
            "Tỷ lệ nhãn dương trên tập train = %.4f%% -> pos_weight = %.1f",
            pos_rate * 100, pos_weight,
        )
    online_store = target_store = None
    if carry_state:
        online_store = BehaviorStateStore(hidden_dim, num_slots, capacity=state_cache)
        target_store = BehaviorStateStore(hidden_dim, num_slots, capacity=state_cache)
        logger.warning(
            "Bật mang trạng thái Behavior Memory qua cửa sổ (bộ đệm %d tài khoản/store, "
            "≈%.0f MB cho cả hai). ĐÃ TẮT xáo trộn thứ tự cửa sổ: mang trạng thái chỉ "
            "đúng khi các cửa sổ chạy theo đúng thứ tự thời gian.",
            state_cache, 2 * state_cache * hidden_dim * (1 + num_slots) * 4 / 1e6,
        )

    loss_fn = DRRECompositeLoss(
        lambda_contrast=lambda_contrast, lambda_bce=lambda_bce, pos_weight=pos_weight,
    ).to(device)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(risk_engine.parameters()), lr=lr
    )

    model_config = dict(
        input_dim=input_dim, hidden_dim=hidden_dim, num_slots=num_slots,
        graph_hidden_dim=graph_hidden_dim, graph_num_layers=graph_num_layers,
        embedding_dim=embedding_dim,
    )

    # Manifest thí nghiệm: gắn commit, phiên bản dữ liệu, seed và siêu tham số
    # vào lần chạy này (xem src/tracking.py). Ghi ra reports/runs/ và nhúng
    # phần gọn vào checkpoint, nhờ đó mỗi con số đều truy được về nguồn.
    started = time.monotonic()
    manifest = RunManifest(
        kind=f"drre_train_{source}",
        device=str(device),
        seed=seed,
        hyperparameters={
            "epochs": epochs, "batch_size": batch_size, "lr": lr,
            "num_negatives": num_negatives, "use_graph": use_graph,
            "disabled_branches": sorted(disabled), "feature_columns": list(feature_cols),
            "scored_rows": int(universe.sum()), "total_rows": len(df_train),
            "model_config": model_config, "pos_weight": pos_weight,
            "lambda_contrast": lambda_contrast, "lambda_bce": lambda_bce,
            "ema_decay": ema_decay, "contrastive_temperature": temperature,
            "config_path": config_path, "checkpoint_path": checkpoint_path,
            "selection": selection, "smooth_window": SMOOTH_WINDOW,
            "carry_state": carry_state,
            "state_cache": state_cache if carry_state else None,
            "shuffle_windows": not carry_state,
        },
        data={
            "train_file": {**file_fingerprint(train_path), **dataframe_summary(df_train)},
            "val_file": (
                {**file_fingerprint(val_path), **dataframe_summary(df_val)}
                if df_val is not None else {"exists": False}
            ),
        },
        results={"epochs": []},
    )
    extra = {
        "use_graph": use_graph,
        "disabled_branches": sorted(disabled),
        # Dashboard/evaluate đọc khóa này để chấm điểm bằng ĐÚNG bộ đặc trưng
        # đã huấn luyện — xem scoring_service.try_load_drre_checkpoint.
        "feature_columns": list(feature_cols),
        "carry_state": carry_state,
        "run": manifest.checkpoint_fields(),
    }

    # Chọn checkpoint: giữ trọng số của tối đa SMOOTH_WINDOW epoch gần nhất trên
    # CPU (56k tham số ≈ 0,2 MB/bản) để còn ghi được epoch ở GIỮA cửa sổ trượt.
    snapshots: deque[tuple[int, float, dict, dict]] = deque(maxlen=SMOOTH_WINDOW)
    best_score, best_epoch = float("-inf"), None
    start_epoch, elapsed_before = 1, 0.0
    session = {"started_utc": manifest.started_utc, "git": manifest.git, "epochs": []}
    if resume_state is not None:
        saved = resume_state["manifest"]
        # `epochs` được phép TĂNG khi chạy tiếp (đường val còn đi lên thì chạy
        # thêm) — nó chỉ là điểm dừng, không đổi bản chất các epoch đã chạy.
        # Mọi siêu tham số khác phải giữ nguyên.
        saved_epochs = saved["hyperparameters"].get("epochs")
        mismatches = _resume_mismatches(
            {k: v for k, v in saved["hyperparameters"].items() if k != "epochs"},
            {k: v for k, v in manifest.hyperparameters.items() if k != "epochs"})
        mismatches += _resume_mismatches(
            {k: v.get("sha256") for k, v in saved["data"].items()},
            {k: v.get("sha256") for k, v in manifest.data.items()})
        if mismatches:
            raise ValueError(
                f"Cấu hình hoặc dữ liệu khác với lúc bắt đầu {resume_path} — không chạy tiếp "
                "được:\n  " + "\n  ".join(mismatches))
        if resume_state["next_epoch"] > epochs:
            logger.info("%s đã chạy đủ %d epoch (run_id=%s) — không còn gì để chạy.",
                        resume_path, epochs, saved["run_id"])
            return
        model.load_state_dict(resume_state["model"])
        risk_engine.load_state_dict(resume_state["risk_engine"])
        target.target_encoder.load_state_dict(resume_state["target"])
        optimizer.load_state_dict(resume_state["optimizer"])
        rng.setstate(resume_state["python_rng"])
        torch.set_rng_state(resume_state["torch_rng"])
        snapshots.extend(resume_state["snapshots"])
        best_score, best_epoch = resume_state["best_score"], resume_state["best_epoch"]
        start_epoch, elapsed_before = resume_state["next_epoch"], resume_state["elapsed_seconds"]
        manifest.run_id, manifest.started_utc = saved["run_id"], saved["started_utc"]
        manifest.results, manifest.notes = saved["results"], saved["notes"]
        if saved_epochs != epochs:
            manifest.notes.append(
                f"Số epoch đổi từ {saved_epochs} thành {epochs} khi chạy tiếp từ epoch "
                f"{start_epoch} — các epoch đã chạy giữ nguyên.")
        extra["run"] = manifest.checkpoint_fields()
        logger.info("Chạy tiếp %s từ epoch %d/%d (đã chạy %.1f giờ ở các phiên trước).",
                    manifest.run_id, start_epoch, epochs, elapsed_before / 3600)

    prior_sessions = list(manifest.results.get("sessions", []))

    def _elapsed() -> float:
        return elapsed_before + time.monotonic() - started

    for epoch in range(start_epoch, epochs + 1):
        # rng=None -> giữ nguyên thứ tự thời gian của các cửa sổ, bắt buộc khi
        # mang trạng thái qua cửa sổ (xem BehaviorStateStore).
        n_windows, windows = iter_time_windows(
            df_train, batch_size, time_col, None if carry_state else rng)
        if online_store is not None:
            online_store.reset()
            target_store.reset()

        epoch_stats = {"loss": 0.0, "l_pred": 0.0, "l_contrast": 0.0, "l_bce": 0.0}
        n_windows_no_pred_pairs = 0
        n_windows_no_bce = 0
        model.train()
        epoch_started = time.monotonic()
        for i_window, window in enumerate(windows, start=1):
            if i_window % PROGRESS_EVERY == 0:
                spent = time.monotonic() - epoch_started
                logger.info(
                    "Epoch %d | cửa sổ %d/%d (%.0f%%) | %.1f phút | còn khoảng %.1f phút",
                    epoch, i_window, n_windows, 100 * i_window / max(1, n_windows),
                    spent / 60, spent / i_window * (n_windows - i_window) / 60)
            stats = train_on_window(
                model, risk_engine, target, loss_fn, optimizer, window, device, num_negatives, rng,
                use_graph=use_graph, feature_columns=feature_cols,
                online_store=online_store, target_store=target_store,
            )
            for k in epoch_stats:
                epoch_stats[k] += stats[k]
            if stats["n_pred_pairs"] == 0:
                n_windows_no_pred_pairs += 1
            if stats["n_bce_labels"] == 0:
                n_windows_no_bce += 1

        n = max(1, n_windows)
        logger.info(
            "Epoch %d/%d | loss=%.4f l_pred=%.4f l_contrast=%.4f l_bce=%.4f | "
            "%d/%d cửa sổ không có cặp L_pred (tăng --batch-size nếu tỷ lệ này cao)",
            epoch, epochs, epoch_stats["loss"] / n, epoch_stats["l_pred"] / n,
            epoch_stats["l_contrast"] / n, epoch_stats["l_bce"] / n,
            n_windows_no_pred_pairs, n_windows,
        )
        if online_store is not None:
            # Tỷ lệ "gặp lại": bao nhiêu phần trăm tài khoản trong một cửa sổ đã
            # có trạng thái từ cửa sổ trước. Đây chính là con số nói Behavior
            # Memory có dữ liệu để nhớ hay không — trước đây nó bằng 0 tuyệt đối.
            logger.info(
                "Epoch %d | Behavior Memory gặp lại %.1f%% tài khoản (%d/%d lượt tra), "
                "%d lần đẩy khỏi bộ đệm",
                epoch, 100 * online_store.hit_rate, online_store.hits,
                online_store.lookups, online_store.evictions,
            )
        if n_windows_no_bce:
            # Mặt nạ vũ trụ chấm điểm có thể làm một cửa sổ không còn nhãn nào
            # (vd. cửa sổ nằm trọn trong giờ bị loại của PaySim). Vài cửa sổ là
            # bình thường; gần hết thì mặt nạ đang sai và BCE không được học.
            level = logger.warning if n_windows_no_bce > 0.5 * n_windows else logger.info
            level("Epoch %d | %d/%d cửa sổ không có nhãn nào trong vũ trụ chấm điểm "
                  "(BCE không đóng góp gradient ở các cửa sổ đó)",
                  epoch, n_windows_no_bce, n_windows)

        epoch_record = {
            "epoch": epoch,
            "loss": epoch_stats["loss"] / n,
            "l_pred": epoch_stats["l_pred"] / n,
            "l_contrast": epoch_stats["l_contrast"] / n,
            "l_bce": epoch_stats["l_bce"] / n,
            "n_windows": n_windows,
            "n_windows_no_pred_pairs": n_windows_no_pred_pairs,
            "n_windows_no_bce": n_windows_no_bce,
            "state_hit_rate": (None if online_store is None else online_store.hit_rate),
            "seconds_elapsed": round(_elapsed(), 1),
        }

        if df_val is not None:
            model.eval()
            auprc = evaluate_auprc(
                model, risk_engine, df_val, device, use_graph=use_graph, batch_size=batch_size,
                feature_columns=feature_cols,
                # Tập val nằm SAU tập train theo thời gian, nên nó được phép
                # tiếp tục trạng thái cuối của train — đúng như hệ thống thật.
                state_store=online_store)
            if auprc is not None:
                epoch_record["val_auprc"] = auprc
                logger.info("Epoch %d | Val AUPRC = %.4f", epoch, auprc)
                snapshots.append((epoch, auprc, _cpu_state(model), _cpu_state(risk_engine)))
                cand = _online_candidate(snapshots, selection)
                if cand is not None and cand[1] > best_score:
                    cand_epoch, cand_score, cand_raw, model_sd, risk_sd = cand
                    best_score, best_epoch = cand_score, cand_epoch
                    _mark_saved(manifest.results["epochs"], epoch_record, cand_epoch)
                    manifest.results["best_epoch"] = cand_epoch
                    manifest.results["best_val_auprc"] = cand_raw
                    manifest.results["selection"] = selection
                    if selection == "smoothed":
                        manifest.results["best_val_auprc_smoothed"] = cand_score
                    extra["run"] = manifest.checkpoint_fields()
                    _save_state_dicts(model_sd, risk_sd, model_config, checkpoint_path, extra=extra)
                    logger.info(
                        "Đã lưu checkpoint của epoch %d (val AUPRC=%.4f, điểm chọn %s=%.4f) -> %s",
                        cand_epoch, cand_raw, selection, cand_score, checkpoint_path,
                    )

        manifest.results["epochs"].append(epoch_record)

        session["epochs"].append(epoch)
        session["finished_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        manifest.results["sessions"] = [*prior_sessions, dict(session, epochs=list(session["epochs"]))]
        _save_resume_state(resume_path, {
            "format": RESUME_FORMAT,
            "next_epoch": epoch + 1,
            "elapsed_seconds": _elapsed(),
            "model": _cpu_state(model),
            "risk_engine": _cpu_state(risk_engine),
            "target": _cpu_state(target.target_encoder),
            "optimizer": optimizer.state_dict(),
            "python_rng": rng.getstate(),
            "torch_rng": torch.get_rng_state(),
            "snapshots": list(snapshots),
            "best_score": best_score,
            "best_epoch": best_epoch,
            "manifest": {
                "run_id": manifest.run_id, "started_utc": manifest.started_utc,
                "hyperparameters": manifest.hyperparameters, "data": manifest.data,
                "results": manifest.results, "notes": manifest.notes,
            },
        })
        if epoch < epochs and stop_after is not None and len(session["epochs"]) >= stop_after:
            logger.info(
                "Dừng phiên sau epoch %d/%d theo --stop-after %d (phiên này %.1f giờ). Chạy "
                "lại ĐÚNG lệnh cũ kèm --resume để tiếp tục từ epoch %d. Trạng thái: %s",
                epoch, epochs, stop_after, (time.monotonic() - started) / 3600, epoch + 1,
                resume_path)
            return

    if best_epoch is None and snapshots:
        # Chưa đủ SMOOTH_WINDOW epoch val để tính trung bình trượt -> rơi về argmax
        # trên những epoch đã có, chứ KHÔNG mặc nhiên lấy epoch cuối.
        cand_epoch, cand_raw, model_sd, risk_sd = max(snapshots, key=lambda s: s[1])
        best_epoch = cand_epoch
        manifest.notes.append(
            f"Chỉ có {len(snapshots)} epoch val < cửa sổ trượt {SMOOTH_WINDOW} — đã rơi về "
            f"argmax (epoch {cand_epoch}). Con số val của epoch này bị thiên lệch lạc quan."
        )
        _mark_saved(manifest.results["epochs"], {}, cand_epoch)
        manifest.results["best_epoch"] = cand_epoch
        manifest.results["best_val_auprc"] = cand_raw
        manifest.results["selection"] = f"{selection}->argmax (ít hơn {SMOOTH_WINDOW} epoch val)"
        extra["run"] = manifest.checkpoint_fields()
        _save_state_dicts(model_sd, risk_sd, model_config, checkpoint_path, extra=extra)
        logger.info("Đã lưu checkpoint của epoch %d (val AUPRC=%.4f) -> %s",
                    cand_epoch, cand_raw, checkpoint_path)
    elif selection == "smoothed" and best_epoch is not None:
        manifest.notes.append(
            f"Checkpoint được chọn bằng trung bình trượt có tâm {SMOOTH_WINDOW} epoch "
            f"(--select smoothed): epoch 1 và epoch {epochs} KHÔNG nằm trong diện được chọn "
            "vì không đủ epoch kề để biết chúng có phải đỉnh nhiễu hay không."
        )

    if best_epoch is None:
        # Không có val/nhãn để chọn "best" -> lưu checkpoint cuối cùng
        manifest.notes.append(
            "Không có tập val gắn nhãn để chọn checkpoint tốt nhất — đã lưu checkpoint "
            "sau epoch cuối. Số liệu val KHÔNG có nghĩa là đã chọn model."
        )
        extra["run"] = manifest.checkpoint_fields()
        _save_checkpoint(model, risk_engine, model_config, checkpoint_path, extra=extra)
        logger.info("Không có tập val gắn nhãn — đã lưu checkpoint sau epoch cuối -> %s", checkpoint_path)

    manifest.finish(started)
    # Thời lượng là tổng các phiên, không chỉ phiên cuối.
    manifest.duration_seconds = round(_elapsed(), 1)
    if runs_dir is None:
        logger.info("Không ghi manifest (runs_dir=None) — run_id=%s, %.0fs",
                    manifest.run_id, manifest.duration_seconds or 0.0)
    else:
        manifest_path = manifest.write(runs_dir)
        logger.info("Manifest thí nghiệm: %s (run_id=%s, %.0fs)",
                    manifest_path, manifest.run_id, manifest.duration_seconds or 0.0)


def _save_checkpoint(
    model: DRRE, risk_engine: RiskScoringEngine, model_config: dict, path: str,
    extra: dict | None = None,
) -> None:
    """Lưu trọng số HIỆN TẠI của model/risk_engine. Khi cần lưu trọng số của
    một epoch cũ (chọn checkpoint bằng trung bình trượt) thì gọi
    _save_state_dicts trực tiếp."""
    _save_state_dicts(
        model.state_dict(), risk_engine.state_dict(), model_config, path, extra=extra)


def _save_state_dicts(
    model_state: dict, risk_engine_state: dict, model_config: dict, path: str,
    extra: dict | None = None,
) -> None:
    """Lưu đúng định dạng mà src/dashboard/scoring_service.py::try_load_drre_checkpoint
    kỳ vọng — hai bên PHẢI khớp nhau, xem test_train_checkpoint_compatible_with_dashboard.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_config": model_config,
            "drre_state_dict": model_state,
            "risk_engine_state_dict": risk_engine_state,
            **(extra or {}),
        },
        out_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Huấn luyện DRRE (Module 3′)")
    parser.add_argument(
        "--source", required=True,
        choices=["paysim", "ieee_cis", "synthetic_vn", "amlworld", "transxion"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128,
                         help="Kích thước cửa sổ thời gian (không phải batch ngẫu nhiên)")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-negatives", type=int, default=5)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--checkpoint", default="models/drre_checkpoint.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs-dir", default=str(DEFAULT_RUNS_DIR),
                        help="Thư mục ghi manifest thí nghiệm. Để trống để không ghi.")
    parser.add_argument("--select", default="smoothed", choices=list(SELECTION_MODES),
                        help="Cách chọn checkpoint theo val AUPRC. 'smoothed' (mặc định) dùng "
                             f"trung bình trượt có tâm {SMOOTH_WINDOW} epoch nên không chọn theo "
                             "một đỉnh nhiễu, nhưng epoch đầu và epoch cuối không được chọn. "
                             "'argmax' là hành vi cũ (lấy val AUPRC cao nhất).")
    parser.add_argument("--features", default=None, metavar="COT1,COT2",
                        help="Ghi đè danh sách đặc trưng đầu vào (cách nhau bằng dấu phẩy). "
                             "Mặc định lấy theo nguồn từ FEATURES_BY_SOURCE. Ví dụ dùng để "
                             "ablation cột rò rỉ của PaySim: --features amount_norm")
    parser.add_argument(
        "--lambda-bce", type=float, default=None,
        help="Ghi đè trọng số mất mát có giám sát (mặc định lấy từ config, 0.3). "
             "risk_engine chỉ học qua số hạng này, nên đây là tham số đáng quét.")
    parser.add_argument(
        "--carry-state", action="store_true",
        help="Mang trạng thái Behavior Memory qua các cửa sổ theo account_id. Bật cái này "
             "thì thứ tự cửa sổ KHÔNG còn được xáo trộn (bắt buộc, nếu không sẽ rò rỉ "
             "tương lai). Không bật thì bộ nhớ hành vi chỉ dài bằng một cửa sổ — với "
             "PaySim cửa sổ 32.768 chỉ 2,1%% giao dịch gian lận có lịch sử trong cửa sổ.")
    parser.add_argument(
        "--state-cache", type=int, default=DEFAULT_STATE_CACHE,
        help=f"Số tài khoản tối đa giữ trạng thái (mặc định {DEFAULT_STATE_CACHE}, "
             "≈192 MB mỗi store, có hai store).")
    parser.add_argument(
        "--resume", action="store_true",
        help="Chạy tiếp từ <checkpoint>.resume.pt (ghi sau MỖI epoch). Phải dùng ĐÚNG các "
             "tham số của lần đầu — khác siêu tham số hay dữ liệu thì bị từ chối. Riêng "
             "--epochs được phép tăng để chạy thêm.")
    parser.add_argument(
        "--stop-after", type=int, default=None, metavar="N",
        help="Dừng sau N epoch trong phiên này để chia lần huấn luyện dài thành nhiều phiên. "
             "Vd. --epochs 5 --stop-after 1, rồi lặp lại lệnh kèm --resume.")
    parser.add_argument(
        "--keep-awake", action="store_true",
        help="Chỉ Windows: xin hệ điều hành không ngủ khi rảnh trong lúc huấn luyện "
             "(SetThreadExecutionState). Không đổi cài đặt nguồn; tự hết khi tiến trình thoát.")
    parser.add_argument("--no-graph", action="store_true",
                         help="Không dựng đồ thị, tắt nhánh đồ thị — bắt buộc với nguồn không có "
                              "người nhận (ieee_cis)")
    parser.add_argument("--disable-branch", action="append", default=[],
                         choices=["transaction", "behavior", "graph"],
                         help="Tắt một nhánh để làm ablation, vd. --disable-branch behavior")
    args = parser.parse_args()

    if args.keep_awake and os.name == "nt":
        import ctypes

        es_continuous, es_system_required = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | es_system_required)

    train(
        source=args.source, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        num_negatives=args.num_negatives, config_path=args.config,
        processed_dir=args.processed_dir, checkpoint_path=args.checkpoint, seed=args.seed,
        use_graph=not args.no_graph, disabled_branches=args.disable_branch,
        runs_dir=args.runs_dir or None, selection=args.select,
        feature_columns=(
            [c.strip() for c in args.features.split(",") if c.strip()] if args.features else None
        ),
        carry_state=args.carry_state, state_cache=args.state_cache,
        lambda_bce=args.lambda_bce, resume=args.resume, stop_after=args.stop_after,
    )


if __name__ == "__main__":
    main()
