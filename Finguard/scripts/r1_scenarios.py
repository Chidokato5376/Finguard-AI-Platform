"""R1: đo đóng góp từng mô-đun kịch bản bằng mô hình đối chứng trên tập val.

Mốc so sánh là GBDT cân bằng trên 18 đặc trưng (20 đặc trưng trừ is_round_amount và
hour_sin). Mỗi biến thể chỉ thêm đúng một mô-đun. Quy tắc giữ (docs/lo-trinh-train-dot-2.md):
KTC 95% của mức tăng AUPRC nằm trên 0, HOẶC recall của đúng kiểu hình mô-đun nhắm tới tăng
có ý nghĩa. Chưa hiệu chỉnh cho việc so sánh nhiều lần: đó là lý do phải khớp cả HI và LI.

Điểm từng biến thể được lưu lại nên chạy tiếp được. Tập test không được đọc.
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

from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.metrics import average_precision_score as ap  # noqa: E402

from src.data.features import features_for_source  # noqa: E402
from src.data.scenarios import DROPPED_BASE_FEATURES, REGISTRY, features_of  # noqa: E402
from src.data.scenarios.store import load_columns  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

BOOTSTRAP = 150
TYPOLOGY_BOOTSTRAP = 2000
BASELINE = "base18"
COMBOS = {
    "+core": ["mule_chain", "cross_bank_fx", "cycle"],                     # đạt 3–4/4
    "+core_li": ["mule_chain", "cross_bank_fx", "cycle", "gather_scatter", "integration"],
    "+kept": ["mule_chain", "cross_bank_fx", "cycle", "gather_scatter", "integration",
              "cash_structuring"],                                          # mọi mô-đun đạt ≥ 2/4
}


def save_results(path: Path, results: dict) -> None:
    """Ghi nguyên tử: mất điện giữa chừng không làm hỏng file dùng để chạy tiếp."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(results, ensure_ascii=False, indent=1), "utf-8")
    os.replace(tmp, path)


def recall_by_typology(y, pattern, scores_a, scores_b, budget, rng) -> dict:
    """Recall của từng kiểu hình trong top-`budget`; A là biến thể, B là mốc."""
    in_a = scores_a >= np.partition(scores_a, -budget)[-budget]
    in_b = scores_b >= np.partition(scores_b, -budget)[-budget]
    rows = {}
    for typology in sorted(set(pattern[y])):
        members = np.flatnonzero(y & (pattern == typology))
        diff = in_a[members].astype(float) - in_b[members].astype(float)
        draws = rng.integers(0, len(members), (TYPOLOGY_BOOTSTRAP, len(members)))
        means = diff[draws].mean(axis=1)
        rows[typology] = {
            "n": int(len(members)), "recall": float(in_a[members].mean()),
            "recall_base": float(in_b[members].mean()), "diff": float(diff.mean()),
            "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
        }
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--tag", required=True, help="vd. hi_small")
    parser.add_argument("--source", default="amlworld")
    parser.add_argument("--out-dir", default="reports/r1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep-awake", action="store_true")
    parser.add_argument("--no-early-stop", action="store_true",
                        help="đủ 200 cây, không dừng sớm: để số cây không khác nhau giữa các biến thể")
    args = parser.parse_args()

    if args.keep_awake and sys.platform == "win32":
        import ctypes

        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)

    out = Path(args.out_dir) / args.tag
    (out / "scores").mkdir(parents=True, exist_ok=True)
    processed = Path(args.processed_dir)
    base20 = features_for_source(args.source)
    base18 = [c for c in base20 if c not in DROPPED_BASE_FEATURES]
    active = [name for name, module in REGISTRY.items() if module.FEATURES]

    extra = ["label", "eval_universe"]
    train = pd.read_parquet(processed / f"{args.source}_train.parquet", columns=base20 + extra)
    val = pd.read_parquet(processed / f"{args.source}_val.parquet",
                          columns=base20 + extra + ["laundering_pattern"])
    scenario_train = load_columns(processed, "train", active)
    scenario_val = load_columns(processed, "val", active)

    keep_train = train["eval_universe"].astype(bool).to_numpy()
    keep_val = val["eval_universe"].astype(bool).to_numpy()
    y_train = train["label"].astype(bool).to_numpy()[keep_train]
    y = val["label"].astype(bool).to_numpy()[keep_val]
    pattern = val["laundering_pattern"].astype(object).fillna("").to_numpy(dtype=str)[keep_val]

    def matrix(columns: list[str]) -> tuple[np.ndarray, np.ndarray]:
        def build(frame, scenario, mask):
            parts = [frame[c].to_numpy(np.float32)[mask] if c in frame.columns else scenario[c][mask]
                     for c in columns]
            return np.column_stack(parts)
        return build(train, scenario_train, keep_train), build(val, scenario_val, keep_val)

    variants = {"base20": base20, BASELINE: base18}
    variants.update({f"+{name}": base18 + features_of([name]) for name in active})
    variants["+all"] = base18 + features_of(active)
    # Tổ hợp theo kết quả lượt R1 trước (10/11/2026): số lần đạt trên 4 lượt đo HI/LI × 2 cấu hình.
    for name, modules in COMBOS.items():
        variants[name] = base18 + features_of(modules)

    results_path = out / "results.json"
    results: dict = json.loads(results_path.read_text("utf-8")) if results_path.exists() else {}
    scores: dict[str, np.ndarray] = {}
    def versions(columns: list[str]) -> dict[str, int]:
        """Phiên bản các mô-đun mà biến thể dùng: đổi VERSION thì biến thể được tính lại."""
        return {n: m.VERSION for n, m in REGISTRY.items() if set(m.FEATURES) & set(columns)}

    for name, columns in variants.items():
        path = out / "scores" / f"{name}.npy"
        cached = results.get(name, {})
        if path.exists() and cached and cached.get("modules", {}) == versions(columns):
            scores[name] = np.load(path)
            continue
        started = time.monotonic()
        fit_x, eval_x = matrix(columns)
        model = HistGradientBoostingClassifier(
            max_iter=200, class_weight="balanced", random_state=args.seed,
            early_stopping=False if args.no_early_stop else "auto").fit(fit_x, y_train)
        scores[name] = model.predict_proba(eval_x)[:, 1]
        del fit_x, eval_x
        np.save(path, scores[name])
        order = np.argsort(-scores[name], kind="stable")
        results[name] = {
            "n_features": len(columns), "n_iter": int(model.n_iter_), "modules": versions(columns),
            "auprc": float(ap(y, scores[name])),
            **{f"caught@{k}": int(y[order[:k]].sum()) for k in (500, 1000, 2000)},
        }
        save_results(results_path, results)
        logger.info("%s: AUPRC %.4f (%d cột, %.0f giây)", name, results[name]["auprc"], len(columns),
                    time.monotonic() - started)

    rng = np.random.default_rng(args.seed)
    positive, negative = np.flatnonzero(y), np.flatnonzero(~y)
    base_scores = scores[BASELINE]
    budget = int(2 * len(positive))
    for name in variants:
        if name == BASELINE or "ci95" in results[name]:
            continue
        diffs = []
        for _ in range(BOOTSTRAP):
            idx = np.concatenate([rng.choice(positive, len(positive)), rng.choice(negative, len(negative))])
            diffs.append(ap(y[idx], scores[name][idx]) - ap(y[idx], base_scores[idx]))
        results[name]["minus_base"] = results[name]["auprc"] - results[BASELINE]["auprc"]
        results[name]["ci95"] = [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]
        results[name]["typology"] = recall_by_typology(y, pattern, scores[name], base_scores, budget, rng)
        module = name[1:]
        targets = REGISTRY[module].TARGETS if module in REGISTRY else ()
        hits = [t for t in targets if results[name]["typology"].get(t, {}).get("ci95", [0, 0])[0] > 0]
        results[name]["target_typology_gain"] = hits
        results[name]["keep"] = bool(results[name]["ci95"][0] > 0 or hits)
        save_results(results_path, results)
        logger.info("%s − %s: %+.4f, KTC [%+.4f; %+.4f], giữ=%s %s", name, BASELINE,
                    results[name]["minus_base"], *results[name]["ci95"], results[name]["keep"], hits)
    logger.info("Hoàn tất R1 cho %s (ngân sách theo kiểu hình = %d cảnh báo)", args.tag, budget)
    return 0


if __name__ == "__main__":
    sys.exit(main())
