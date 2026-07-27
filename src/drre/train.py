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

⚠ HIỆU NĂNG: Behavior Memory chạy bằng vòng lặp Python từng giao dịch một
(không vector hóa theo account) — ĐÚNG về mặt toán học nhưng CHẬM. Chấp
nhận được cho MVP/thực nghiệm (hàng nghìn giao dịch), KHÔNG phù hợp để
huấn luyện trên toàn bộ PaySim (~6.3 triệu dòng) mà không tối ưu lại.
"""

from __future__ import annotations

import argparse
import logging
import random
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

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = ["amount_norm", "channel_encoded"]


# --------------------------------------------------------------------------
# Batching theo cửa sổ thời gian
# --------------------------------------------------------------------------

def make_time_windows(df: pd.DataFrame, batch_size: int, time_col: str) -> list[pd.DataFrame]:
    """Chia dữ liệu (đã sort theo thời gian từ preprocessing.py) thành các
    cửa sổ liên tiếp không chồng lấn, mỗi cửa sổ batch_size giao dịch.
    """
    df_sorted = df.sort_values(time_col).reset_index(drop=True)
    windows = []
    for start in range(0, len(df_sorted), batch_size):
        window = df_sorted.iloc[start:start + batch_size].reset_index(drop=True)
        if len(window) >= 2:  # cửa sổ 1 dòng không tạo được cặp/negative nào có ích
            windows.append(window)
    return windows


# --------------------------------------------------------------------------
# Forward pass tuần tự theo account trong một cửa sổ
# --------------------------------------------------------------------------

def run_encoder_over_window(
    model: DRRE,
    window: pd.DataFrame,
    node_features: torch.Tensor,
    edge_index: torch.Tensor,
    account_to_idx: dict[str, int],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Chạy DRRE cho một cửa sổ, xử lý TUẦN TỰ theo từng account_id (đúng
    ngữ nghĩa Behavior Memory — GRU cần thứ tự thời gian), Graph Encoder
    dùng chung đồ thị đã dựng sẵn cho cả cửa sổ.

    Trả về tensor đã align đúng thứ tự hàng gốc của `window` (RangeIndex
    0..n-1), để có thể ghép với nhãn/thông tin khác của window bằng index.
    """
    n = len(window)
    e_out: list[torch.Tensor | None] = [None] * n
    pred_out: list[torch.Tensor | None] = [None] * n

    for account_id, group in window.groupby("account_id", sort=False):
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
                window.loc[row_idx, FEATURE_COLUMNS].to_numpy(dtype=np.float32), device=device
            ).unsqueeze(0)
            out = model(x_t, h, m, node_features, edge_index, node_idx)
            h, m = out["h_t"], out["m_t"]
            e_out[row_idx] = out["e_t"].squeeze(0)
            pred_out[row_idx] = out["e_pred"].squeeze(0)

    return {"e": torch.stack(e_out), "pred": torch.stack(pred_out)}  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# Lấy cặp (t, t+1) cho L_pred và (anchor, positive, negatives) cho L_contrast
# --------------------------------------------------------------------------

def build_prediction_pairs(window: pd.DataFrame) -> tuple[list[int], list[int]]:
    """Cặp (nguồn, đích) trong CÙNG account, liên tiếp trong cửa sổ —
    nguồn dùng predictor(online), đích dùng target network (EMA)."""
    src, tgt = [], []
    for _, group in window.groupby("account_id", sort=False):
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
    for _, group in window.groupby("account_id", sort=False):
        idxs = group.index.tolist()
        if len(idxs) < 2:
            continue
        for i in idxs:
            candidates = [j for j in idxs if j != i]
            positives.append(rng.choice(candidates))
            anchors.append(i)

    negatives = []
    n = len(window)
    for a in anchors:
        pool = [j for j in range(n) if account_of[j] != account_of[a]]
        if not pool:
            negatives.append([a] * num_negatives)  # fallback suy biến — cửa sổ chỉ có 1 account
        elif len(pool) < num_negatives:
            negatives.append(rng.choices(pool, k=num_negatives))
        else:
            negatives.append(rng.sample(pool, num_negatives))
    return anchors, positives, negatives


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
) -> dict[str, float]:
    node_features, edge_index, account_to_idx = build_account_graph(window, FEATURE_COLUMNS)
    node_features, edge_index = node_features.to(device), edge_index.to(device)

    online_out = run_encoder_over_window(model, window, node_features, edge_index, account_to_idx, device)

    with torch.no_grad():
        target_out = run_encoder_over_window(
            target.target_encoder, window, node_features, edge_index, account_to_idx, device
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
        e_neg = torch.stack([online_out["e"][negs] for negs in neg_lists])
    else:
        dim = online_out["e"].shape[-1]
        e_anchor = torch.zeros(1, dim, device=device)
        e_pos = torch.zeros(1, dim, device=device)
        e_neg = torch.zeros(1, num_negatives, dim, device=device)

    bce_logits, bce_labels = None, None
    if "label" in window.columns:
        valid = window["label"].notna()
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

@torch.no_grad()
def evaluate_auprc(model: DRRE, risk_engine: RiskScoringEngine, df_val: pd.DataFrame, device: torch.device) -> float | None:
    from src.evaluation.metrics import compute_auprc

    if "label" not in df_val.columns or df_val["label"].notna().sum() == 0:
        return None

    window = df_val.reset_index(drop=True)
    node_features, edge_index, account_to_idx = build_account_graph(window, FEATURE_COLUMNS)
    node_features, edge_index = node_features.to(device), edge_index.to(device)
    out = run_encoder_over_window(model, window, node_features, edge_index, account_to_idx, device)
    scores = risk_engine(out["e"]).cpu().numpy()

    valid = window["label"].notna()
    y_true = window.loc[valid, "label"].astype(int).to_numpy()
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
) -> None:
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

    missing = set(FEATURE_COLUMNS) - set(df_train.columns)
    if missing:
        raise ValueError(f"Dữ liệu thiếu cột đặc trưng: {missing}. Chạy lại preprocessing.py.")
    time_col = "step" if "step" in df_train.columns else "timestamp"

    input_dim = len(FEATURE_COLUMNS)
    model = DRRE(
        input_dim=input_dim, hidden_dim=hidden_dim, num_slots=num_slots,
        graph_hidden_dim=graph_hidden_dim, graph_num_layers=graph_num_layers,
        embedding_dim=embedding_dim,
    ).to(device)
    risk_engine = RiskScoringEngine(embedding_dim=embedding_dim).to(device)
    target = EMATargetEncoder(model, decay=ema_decay)

    loss_fn = DRRECompositeLoss(lambda_contrast=lambda_contrast, lambda_bce=lambda_bce)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(risk_engine.parameters()), lr=lr
    )

    model_config = dict(
        input_dim=input_dim, hidden_dim=hidden_dim, num_slots=num_slots,
        graph_hidden_dim=graph_hidden_dim, graph_num_layers=graph_num_layers,
        embedding_dim=embedding_dim,
    )

    best_auprc = -1.0
    for epoch in range(1, epochs + 1):
        windows = make_time_windows(df_train, batch_size, time_col)
        rng.shuffle(windows)

        epoch_stats = {"loss": 0.0, "l_pred": 0.0, "l_contrast": 0.0, "l_bce": 0.0}
        n_windows_no_pred_pairs = 0
        model.train()
        for window in windows:
            stats = train_on_window(
                model, risk_engine, target, loss_fn, optimizer, window, device, num_negatives, rng,
            )
            for k in epoch_stats:
                epoch_stats[k] += stats[k]
            if stats["n_pred_pairs"] == 0:
                n_windows_no_pred_pairs += 1

        n = max(1, len(windows))
        logger.info(
            "Epoch %d/%d | loss=%.4f l_pred=%.4f l_contrast=%.4f l_bce=%.4f | "
            "%d/%d cửa sổ không có cặp L_pred (tăng --batch-size nếu tỷ lệ này cao)",
            epoch, epochs, epoch_stats["loss"] / n, epoch_stats["l_pred"] / n,
            epoch_stats["l_contrast"] / n, epoch_stats["l_bce"] / n,
            n_windows_no_pred_pairs, len(windows),
        )

        if df_val is not None:
            model.eval()
            auprc = evaluate_auprc(model, risk_engine, df_val, device)
            if auprc is not None:
                logger.info("Epoch %d | Val AUPRC = %.4f", epoch, auprc)
                if auprc > best_auprc:
                    best_auprc = auprc
                    _save_checkpoint(model, risk_engine, model_config, checkpoint_path)
                    logger.info("Đã lưu checkpoint tốt nhất (AUPRC=%.4f) -> %s", auprc, checkpoint_path)

    if df_val is None or best_auprc < 0:
        # Không có val/nhãn để chọn "best" -> lưu checkpoint cuối cùng
        _save_checkpoint(model, risk_engine, model_config, checkpoint_path)
        logger.info("Không có tập val gắn nhãn — đã lưu checkpoint sau epoch cuối -> %s", checkpoint_path)


def _save_checkpoint(model: DRRE, risk_engine: RiskScoringEngine, model_config: dict, path: str) -> None:
    """Lưu đúng định dạng mà src/dashboard/scoring_service.py::try_load_drre_checkpoint
    kỳ vọng — hai bên PHẢI khớp nhau, xem test_train_checkpoint_compatible_with_dashboard.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_config": model_config,
            "drre_state_dict": model.state_dict(),
            "risk_engine_state_dict": risk_engine.state_dict(),
        },
        out_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Huấn luyện DRRE (Module 3′)")
    parser.add_argument("--source", required=True, choices=["paysim", "ieee_cis", "synthetic_vn"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128,
                         help="Kích thước cửa sổ thời gian (không phải batch ngẫu nhiên)")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-negatives", type=int, default=5)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--checkpoint", default="models/drre_checkpoint.pt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(
        source=args.source, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        num_negatives=args.num_negatives, config_path=args.config,
        processed_dir=args.processed_dir, checkpoint_path=args.checkpoint, seed=args.seed,
    )


if __name__ == "__main__":
    main()
