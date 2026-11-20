"""P2 không rò rỉ: DRRE có bổ sung thông tin cho mô hình cổ điển không?

=== VÌ SAO KHÔNG DÙNG drre_contribution.py CHO P2 ===

drre_contribution.py học GBDT với cột điểm/embedding DRRE trên chính tập TRAIN — nơi DRRE đã
được huấn luyện. Trên train, điểm DRRE "đúng" hơn thực tế (trong mẫu), nên GBDT học cách tin nó
quá mức rồi thua trên val: đo ngày 15/11 (HI-Small, seed 42) thêm embedding làm AUPRC GIẢM
0,067 [−0,086; −0,046]. Đó là rò rỉ của phép đo, không phải kết luận về DRRE.

=== CÁCH ĐO Ở ĐÂY ===

Cả GBDT lẫn DRRE đều học trên train và chưa thấy val. Tập val được chia THEO THỜI GIAN làm
hai nửa: nửa đầu (A) để học bộ trộn, nửa sau (B) để đo. Trên B, so ghép cặp:
  * GBDT đứng riêng (bộ trộn chỉ có GBDT thì thứ hạng y hệt GBDT);
  * bộ trộn hồi quy logistic trên [logit GBDT, logit DRRE];
  * trung bình thứ hạng có trọng số w (w chọn trên A).
KTC 95% bootstrap Poisson ghép cặp trên B (scripts/leaderboard.py::paired_bootstrap).

Đầu vào là điểm đã lưu của scripts/leaderboard.py (theo thứ tự dòng gốc của tệp val).

    python -X utf8 scripts/p2_stacking.py --processed-dir data/processed_aml_v2 \\
        --scores-dir reports/leaderboard/processed_aml_v2_val/scores --base gbdt_28 --drre R5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import leaderboard as lb  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


def _logit(p: np.ndarray, scale: float) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64) / scale, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def _ap(y, s) -> float:
    return lb.weighted_ap(y, *lb.ranking_prep(s))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--scores-dir", required=True)
    parser.add_argument("--source", default="amlworld")
    parser.add_argument("--base", default="gbdt_28")
    parser.add_argument("--drre", default="R5")
    parser.add_argument("--bootstrap", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    path = Path(args.processed_dir) / f"{args.source}_val.parquet"
    frame = pd.read_parquet(path, columns=["timestamp", "label", "eval_universe"])
    scores_dir = Path(args.scores_dir)
    base = np.load(scores_dir / f"{args.base}.npy")
    drre = np.load(scores_dir / f"{args.drre}.npy")
    if not (len(base) == len(drre) == len(frame)):
        raise SystemExit("Số dòng điểm và tệp val không khớp.")

    keep = frame["eval_universe"].fillna(True).to_numpy(bool)
    order = np.argsort(frame["timestamp"].to_numpy()[keep], kind="stable")
    y = frame["label"].fillna(False).astype(bool).to_numpy()[keep][order]
    base, drre = base[keep][order], drre[keep][order]
    half = len(y) // 2
    a, b = slice(0, half), slice(half, None)

    x = np.c_[_logit(base, 1.0), _logit(drre, 100.0)]
    stacker = LogisticRegression(max_iter=1000).fit(x[a], y[a])
    stack_b = stacker.decision_function(x[b])

    # Trung bình thứ hạng: w chọn trên A (lưới 0..1), áp nguyên lên B.
    def blend(w: float, part: slice) -> np.ndarray:
        return (1 - w) * rankdata(base[part]) + w * rankdata(drre[part])

    grid = np.round(np.linspace(0, 1, 21), 2)
    w_best = float(max(grid, key=lambda w: _ap(y[a], blend(w, a))))
    blend_b = blend(w_best, b)

    candidates = {args.base: base[b], f"{args.drre}": drre[b],
                  "tron_logistic": stack_b, "tron_thu_hang": blend_b}
    ci = lb.paired_bootstrap(y[b], candidates, args.base, args.bootstrap, args.seed)
    result = {
        "data": str(path), "base": args.base, "drre": args.drre,
        "half_a": {"rows": half, "positives": int(y[a].sum())},
        "half_b": {"rows": len(y) - half, "positives": int(y[b].sum())},
        "logistic_coef": stacker.coef_.ravel().tolist(), "rank_weight_drre": w_best,
        "auprc_b": {k: _ap(y[b], v) for k, v in candidates.items()},
        "caught2000_b": {k: int(y[b][np.argsort(-v, kind="stable")[:2000]].sum())
                         for k, v in candidates.items()},
        "vs_base_ci95": ci,
    }
    out = Path(args.out or scores_dir.parent / f"p2_stacking_{args.drre}.json")
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    base_ap = result["auprc_b"][args.base]
    print(f"Nửa sau val: {result['half_b']['rows']:,} dòng, {result['half_b']['positives']} vụ "
          f"(bộ trộn học trên nửa đầu, {result['half_a']['positives']} vụ)")
    for name, value in result["auprc_b"].items():
        extra = "" if name == args.base else (
            f"  Δ {value - base_ap:+.4f} [{ci[name]['lo']:+.4f}; {ci[name]['hi']:+.4f}]")
        print(f"  {name:>14}: AUPRC {value:.4f}, bắt@2000 {result['caught2000_b'][name]}{extra}")
    print(f"  trọng số DRRE trong trung bình thứ hạng (chọn trên nửa đầu): {w_best}")
    print(f"Đã ghi {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
