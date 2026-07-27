"""Scoring Service cho Dashboard.

=== CẢNH BÁO TÍNH TRUNG THỰC — ĐỌC TRƯỚC KHI DÙNG ===

DRRE (src/drre/model.py) CHƯA ĐƯỢC HUẤN LUYỆN trong dự án này — chưa có
checkpoint. Nếu dashboard dùng trọng số DRRE khởi tạo ngẫu nhiên để tính
"Risk Score", con số đó VÔ NGHĨA (tương đương nhiễu ngẫu nhiên) nhưng
trông giống một điểm rủi ro thật — đây là rủi ro trình bày sai lệch
nghiêm trọng nếu demo trước ban giám khảo.

Do đó, scoring_service triển khai HAI backend rõ ràng, không đánh lẫn:

  1. HeuristicScorer (mặc định, LUÔN chạy được ngay, không cần train):
     tính độ lệch thống kê thật (z-score của log(amount) so với lịch sử
     CHÍNH tài khoản đó + cờ "người nhận lần đầu xuất hiện"). Đây là tín
     hiệu có căn cứ thống kê, tương tự logic delta_t của Context-aware
     Attention (Mục 4.2) nhưng đơn giản hóa, KHÔNG học được pattern phức
     tạp như DRRE thật.

  2. DRRECheckpointScorer (chỉ kích hoạt khi có checkpoint đã huấn luyện
     tại models/drre_checkpoint.pt): dùng model thật. Nếu file không tồn
     tại, get_scorer() tự động rơi về HeuristicScorer và LOG CẢNH BÁO —
     không bao giờ âm thầm dùng trọng số ngẫu nhiên.

Dashboard PHẢI hiển thị rõ backend nào đang được dùng (xem cột `method`
trong kết quả trả về) — không được ẩn thông tin này khỏi người xem.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Suy ra từ vị trí FILE này, không phụ thuộc thư mục đang chạy (cwd) — xem ghi
# chú tương tự trong src/dashboard/data_service.py.
DEFAULT_CHECKPOINT_PATH = Path(__file__).resolve().parents[2] / "models" / "drre_checkpoint.pt"


def compute_heuristic_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Tính risk score heuristic dựa trên độ lệch thống kê thật.

    Công thức (minh bạch, không phải hộp đen):
        z = (log1p(amount) - mean_lịch_sử_tài_khoản) / std_lịch_sử_tài_khoản
        new_counterparty_bonus = +0.8 CHỈ KHI account_id đã có >= 3 giao
            dịch lịch sử (tránh việc mọi giao dịch của tài khoản mới đều
            bị coi là bất thường — tài khoản mới nào cũng có counterparty
            "lần đầu" theo định nghĩa, không mang tín hiệu gì)
        raw_score = z + new_counterparty_bonus - 1.2   (dịch nền để phân
            phối tập trung quanh vùng Low/Medium, chỉ các trường hợp lệch
            thật sự mới đẩy lên Critical — hiệu chỉnh bằng thực nghiệm
            trên dữ liệu synthetic_vn, xem tests/test_scoring_service.py)
        risk_score = 100 * sigmoid(raw_score)  -> khoảng 0-100

    Args:
        df: DataFrame có tối thiểu account_id, counterparty_id, amount,
            và một cột thời gian (step hoặc timestamp) để sắp xếp lịch sử.

    Returns:
        DataFrame gốc + các cột: z_score, is_new_counterparty,
        account_history_count, risk_score, method (luôn = "heuristic_fallback").
    """
    time_col = "step" if "step" in df.columns else "timestamp"
    df = df.sort_values(time_col).reset_index(drop=True).copy()
    df["log_amount"] = np.log1p(df["amount"].astype(float))

    # Thống kê lịch sử TÍCH LŨY theo từng tài khoản (chỉ dùng dữ liệu quá
    # khứ tại thời điểm giao dịch — tránh leakage nhìn thấy tương lai)
    grouped = df.groupby("account_id")["log_amount"]
    rolling_mean = grouped.transform(lambda s: s.shift(1).expanding().mean())
    rolling_std = grouped.transform(lambda s: s.shift(1).expanding().std())
    rolling_count = grouped.transform(lambda s: s.shift(1).expanding().count())
    df["account_history_count"] = rolling_count.fillna(0).astype(int)

    # Tài khoản mới/giao dịch đầu tiên: chưa có lịch sử -> không đủ căn cứ
    # đánh giá độ lệch, gán z_score = 0 (trung tính) thay vì NaN/lỗi.
    z_score = (df["log_amount"] - rolling_mean) / rolling_std.replace(0, np.nan)
    df["z_score"] = z_score.fillna(0.0).clip(-5, 5)

    # Cờ người nhận lần đầu xuất hiện với account_id này
    seen_pairs: set[tuple[str, str]] = set()
    is_new = []
    for acc, cp in zip(df["account_id"], df["counterparty_id"]):
        pair = (acc, cp)
        is_new.append(pair not in seen_pairs)
        seen_pairs.add(pair)
    df["is_new_counterparty"] = is_new

    # Chỉ tính bonus khi tài khoản ĐÃ CÓ lịch sử đủ dài — tránh mọi giao
    # dịch của tài khoản mới đều bị coi là bất thường (xem docstring).
    has_enough_history = df["account_history_count"] >= 3
    new_cp_bonus = (df["is_new_counterparty"] & has_enough_history).astype(float) * 0.8

    raw_score = df["z_score"] + new_cp_bonus - 1.2
    df["risk_score"] = 100.0 / (1.0 + np.exp(-raw_score))
    df["method"] = "heuristic_fallback"
    return df


def try_load_drre_checkpoint(checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH):
    """Thử tải checkpoint DRRE + RiskScoringEngine đã huấn luyện. Trả về
    None nếu chưa có — KHÔNG raise, để dashboard tự động rơi về heuristic
    một cách êm ái.

    Định dạng checkpoint kỳ vọng (lưu bởi training loop, src/drre/model.py):
        {
            "model_config": {...kwargs khởi tạo DRRE...},
            "drre_state_dict": drre.state_dict(),
            "risk_engine_state_dict": risk_engine.state_dict(),
        }

    Returns:
        tuple (drre_model, risk_engine) nếu tải thành công, None nếu không.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        logger.warning(
            "Không tìm thấy checkpoint DRRE tại %s — dashboard dùng "
            "HeuristicScorer (xem cảnh báo tính trung thực trong "
            "scoring_service.py). KHÔNG dùng trọng số DRRE ngẫu nhiên.",
            checkpoint_path,
        )
        return None
    try:
        import torch

        from src.drre.model import DRRE
        from src.scoring.risk_scoring import RiskScoringEngine

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model = DRRE(**checkpoint["model_config"])
        model.load_state_dict(checkpoint["drre_state_dict"])
        model.eval()

        risk_engine = RiskScoringEngine(embedding_dim=checkpoint["model_config"]["embedding_dim"])
        risk_engine.load_state_dict(checkpoint["risk_engine_state_dict"])
        risk_engine.eval()

        logger.info("Đã tải checkpoint DRRE + RiskScoringEngine từ %s", checkpoint_path)
        return model, risk_engine
    except ImportError:
        logger.warning("torch chưa được cài — không thể tải checkpoint DRRE. Dùng heuristic.")
        return None
    except Exception as e:  # noqa: BLE001 — log rõ lý do rồi rơi về fallback, không che giấu lỗi
        logger.warning("Lỗi khi tải checkpoint DRRE (%s) — dùng heuristic.", e)
        return None


def _score_with_drre(df: pd.DataFrame, drre, risk_engine) -> pd.DataFrame:
    """Nhánh DRRE thật — chạy khi có checkpoint đã huấn luyện.

    Tái dùng src/dashboard/graph_utils.py::build_account_graph (đã kiểm
    thử end-to-end với dữ liệu thật, xem lịch sử phát triển dashboard).
    Đặc trưng đầu vào PHẢI khớp FEATURE_COLUMNS đã dùng lúc huấn luyện —
    xem src/dashboard/graph_utils.py và model_config trong checkpoint.
    """
    import numpy as np
    import torch

    from src.dashboard.graph_utils import build_account_graph

    feature_cols = ["amount_norm", "channel_encoded"]
    missing = set(feature_cols) - set(df.columns)
    if missing:
        raise ValueError(
            f"DataFrame thiếu cột đặc trưng cho DRRE: {missing}. "
            "Chạy src/data/preprocessing.py trước."
        )

    node_features, edge_index, account_to_idx = build_account_graph(df, feature_cols)
    x_t = torch.tensor(df[feature_cols].to_numpy(dtype=np.float32))
    batch_size = len(df)

    # Cold-start tuyệt đối mỗi lần load Dashboard — xem ghi chú tương tự
    # trong lịch sử phát triển src/dashboard (inference.py, nay đã hợp nhất
    # vào đây). Hệ thống sản xuất thật cần lưu/khôi phục (h_prev, m_prev)
    # theo account_id giữa các batch.
    h_prev, m_prev = drre.behavior_memory.init_state(batch_size, device=x_t.device)
    node_index = torch.tensor([account_to_idx[a] for a in df["account_id"]], dtype=torch.long)

    with torch.no_grad():
        out = drre(x_t, h_prev, m_prev, node_features, edge_index, node_index)
        scores = risk_engine(out["e_t"])

    result = df.copy()
    result["risk_score"] = scores.numpy()
    result["max_attention_weight"] = out["alpha_k"].max(dim=-1).values.numpy()
    beta = out["beta"].numpy()
    result["beta_transaction"] = beta[:, 0]
    result["beta_behavior"] = beta[:, 1]
    result["beta_graph"] = beta[:, 2]
    result["delta_t"] = out["delta_t"].numpy()
    result["method"] = "drre_trained"
    return result


def score_batch(
    df: pd.DataFrame, checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH
) -> pd.DataFrame:
    """Điểm vào chính cho dashboard: tự chọn backend, LUÔN gắn cột `method`
    để UI hiển thị minh bạch nguồn gốc điểm số.
    """
    loaded = try_load_drre_checkpoint(checkpoint_path)
    if loaded is not None:
        drre, risk_engine = loaded
        return _score_with_drre(df, drre, risk_engine)
    return compute_heuristic_scores(df)


def classify_tier(score: float, low_max: int = 30, medium_max: int = 70) -> str:
    """Dùng chung ngưỡng với src/scoring/risk_scoring.py::classify_tier
    (giữ độc lập ở đây để dashboard không bắt buộc cài torch chỉ để phân loại)."""
    if score <= low_max:
        return "low"
    if score <= medium_max:
        return "medium"
    return "critical"
