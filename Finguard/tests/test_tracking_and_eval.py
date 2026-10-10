"""Kiểm thử theo dõi thí nghiệm (src/tracking.py) và script đánh giá
(scripts/evaluate.py).

Hai bất biến quan trọng nhất ở đây:
  - Dấu vân tay dữ liệu phải ĐỔI khi nội dung đổi, nếu không manifest
    không chứng minh được "đã train trên đúng lát dữ liệu này".
  - Đặc trưng lịch sử trong script đánh giá phải tính NHÂN QUẢ: giá trị
    của một dòng không được thay đổi khi các dòng SAU nó thay đổi.
"""

from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.tracking import (
    RunManifest,
    dataframe_summary,
    file_fingerprint,
    git_commit,
    library_versions,
)

EVALUATE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "evaluate.py"


def _load_evaluate_module():
    spec = importlib.util.spec_from_file_location("evaluate_script", EVALUATE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _frame(n: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = datetime(2026, 3, 1)
    return pd.DataFrame({
        "account_id": [f"ACC{i % 6}" for i in range(n)],
        "counterparty_id": [f"CP{i % 4}" for i in range(n)],
        "timestamp": [start + timedelta(hours=i) for i in range(n)],
        "amount": rng.uniform(1e5, 1e7, n),
        "amount_norm": rng.normal(0, 1, n),
        "channel_encoded": (np.arange(n) % 3).astype(float),
        "label": (np.arange(n) % 7 == 0),
    })


# --------------------------------------------------------------------------
# src/tracking.py
# --------------------------------------------------------------------------

def test_file_fingerprint_changes_with_content(tmp_path) -> None:
    path = tmp_path / "data.parquet"
    _frame().to_parquet(path, index=False)
    first = file_fingerprint(path)

    _frame(seed=1).to_parquet(path, index=False)
    second = file_fingerprint(path)

    assert first["exists"] and first["hash_scope"] == "full"
    assert first["sha256"] != second["sha256"], \
        "đổi nội dung mà dấu vân tay không đổi thì manifest vô giá trị"


def test_file_fingerprint_is_stable_for_same_bytes(tmp_path) -> None:
    path = tmp_path / "a.bin"
    path.write_bytes(b"finguard" * 1000)
    assert file_fingerprint(path)["sha256"] == file_fingerprint(path)["sha256"]


def test_file_fingerprint_missing_file() -> None:
    assert file_fingerprint("khong/ton/tai.parquet")["exists"] is False


def test_dataframe_summary_counts_positives_and_time_range() -> None:
    summary = dataframe_summary(_frame(n=70))
    assert summary["n_rows"] == 70
    assert summary["n_positive"] == 10
    assert summary["time_col"] == "timestamp"
    assert summary["n_accounts"] == 6
    assert summary["positive_rate"] == pytest.approx(10 / 70)


def test_run_manifest_writes_json_and_checkpoint_fields(tmp_path) -> None:
    manifest = RunManifest(
        kind="drre_train_test",
        seed=7,
        hyperparameters={"epochs": 2, "batch_size": 64},
        data={"train_file": {"sha256": "abc123", "n_rows": 10}},
        results={"best_val_auprc": 0.42},
    )
    manifest.finish()
    out = manifest.write(tmp_path)

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["seed"] == 7
    assert payload["hyperparameters"]["epochs"] == 2
    assert payload["reproducibility"]["git"]["commit"] == git_commit()
    assert "torch" in payload["reproducibility"]["libraries"]
    assert payload["finished_utc"] is not None

    fields = manifest.checkpoint_fields()
    assert fields["seed"] == 7
    assert fields["data_fingerprint"] == {"train_file": "abc123"}
    assert fields["results"]["best_val_auprc"] == 0.42


def test_library_versions_reports_absent_instead_of_crashing() -> None:
    versions = library_versions()
    assert set(versions) >= {"python", "pandas", "numpy", "torch"}
    assert all(isinstance(v, str) and v for v in versions.values())


# --------------------------------------------------------------------------
# scripts/evaluate.py
# --------------------------------------------------------------------------

def test_history_features_do_not_look_into_the_future() -> None:
    """Sửa các dòng SAU không được làm đổi đặc trưng của dòng TRƯỚC."""
    evaluate = _load_evaluate_module()
    df = _frame(n=60)
    original = evaluate._causal_history_features(df)

    tampered = df.copy()
    tampered.loc[30:, "amount"] = tampered.loc[30:, "amount"] * 1000.0
    after = evaluate._causal_history_features(tampered)

    pd.testing.assert_frame_equal(original.iloc[:30], after.iloc[:30])
    assert not original.iloc[30:].equals(after.iloc[30:]), \
        "đổi số tiền của chính dòng đó thì đặc trưng phải đổi"


def test_history_features_first_transaction_has_no_history() -> None:
    evaluate = _load_evaluate_module()
    df = _frame(n=18)
    features = evaluate._causal_history_features(df)
    first_rows = df.groupby("account_id", sort=False).head(1).index
    assert (features.loc[first_rows, "n_prior"] == 0).all()
    assert (features.loc[first_rows, "z_amount"] == 0.0).all(), \
        "giao dịch đầu tiên chưa có lịch sử nên z-score phải trung tính"


def test_metrics_for_golden_values() -> None:
    """Kiểm thử hồi quy trên một tập cố định — nếu cách tính chỉ số đổi,
    test này đỏ và mọi số liệu cũ phải được tính lại."""
    evaluate = _load_evaluate_module()
    y = np.array([1, 0, 1, 0, 0, 0, 1, 0, 0, 0])
    scores = np.array([0.95, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10])

    metrics = evaluate.metrics_for(y, scores, budgets=(2, 5))
    assert metrics["caught@2"] == 1.0
    assert metrics["caught@5"] == 2.0
    assert metrics["recall@5"] == pytest.approx(2 / 3)
    assert metrics["fpr@5"] == pytest.approx(0.6)
    # 3 dương / 7 âm, 16/21 cặp xếp đúng thứ tự -> ROC-AUC = 16/21
    assert metrics["roc_auc"] == pytest.approx(16 / 21, abs=1e-9)
    assert metrics["auprc"] == pytest.approx(0.698412698, abs=1e-9)


def test_markdown_table_sorted_by_auprc() -> None:
    evaluate = _load_evaluate_module()
    rows = {
        "tot": {"auprc": 0.3, "roc_auc": 0.8, "caught@100": 30.0},
        "kem": {"auprc": 0.1, "roc_auc": 0.5, "caught@100": 10.0},
    }
    table = evaluate._markdown_table(rows, (100,))
    assert table.index("kem") < table.index("tot"), "bảng xếp từ kém tới tốt"
    assert "Bắt được / 100" in table


# ---------------------------------------------------------------------------
# Chỉ số theo giai đoạn thời gian (--by-period)
# ---------------------------------------------------------------------------

def test_period_bounds_covers_every_row_without_overlap() -> None:
    ev = _load_evaluate_module()
    for n, k in ((450_000, 4), (100, 3), (10, 4), (7, 7), (5, 1)):
        bounds = ev.period_bounds(n, k)
        assert bounds[0][0] == 0 and bounds[-1][1] == n, (n, k, bounds)
        for (_, hi), (lo2, _) in zip(bounds, bounds[1:]):
            assert hi == lo2, "giai đoạn phải liền nhau, không chồng và không hở"
        assert sum(hi - lo for lo, hi in bounds) == n


def test_period_bounds_drops_empty_periods_when_more_periods_than_rows() -> None:
    ev = _load_evaluate_module()
    bounds = ev.period_bounds(3, 10)
    assert all(hi > lo for lo, hi in bounds)
    assert sum(hi - lo for lo, hi in bounds) == 3


def test_period_bounds_rejects_zero_periods() -> None:
    ev = _load_evaluate_module()
    with pytest.raises(ValueError, match="n_periods"):
        ev.period_bounds(100, 0)


def test_metrics_by_period_lift_is_comparable_across_periods() -> None:
    """Bất biến cốt lõi: AUPRC thô KHÔNG so sánh được giữa các giai đoạn vì mốc
    ngẫu nhiên mỗi giai đoạn khác nhau. Ở đây model chấm điểm HOÀN HẢO ở cả hai
    giai đoạn nhưng giai đoạn 2 có tỷ lệ gian lận gấp 4 lần: AUPRC thô bằng
    nhau (=1.0) còn lift phải khác nhau đúng theo tỷ lệ nhãn dương."""
    ev = _load_evaluate_module()
    y = np.array([0] * 19 + [1] + [0] * 16 + [1] * 4)  # 20 dòng/giai đoạn
    scores = y.astype(float) * 100  # chấm hoàn hảo
    periods = ev.metrics_by_period(y, {"hoan hao": scores}, 2)

    assert [p["n_pos"] for p in periods] == [1, 4]
    assert periods[0]["methods"]["hoan hao"]["auprc"] == pytest.approx(1.0)
    assert periods[1]["methods"]["hoan hao"]["auprc"] == pytest.approx(1.0)
    assert periods[0]["methods"]["hoan hao"]["lift"] == pytest.approx(20.0)
    assert periods[1]["methods"]["hoan hao"]["lift"] == pytest.approx(5.0)


def test_metrics_by_period_skips_period_without_positives() -> None:
    ev = _load_evaluate_module()
    y = np.array([0] * 10 + [0, 1] * 5)
    periods = ev.metrics_by_period(y, {"m": np.arange(20, dtype=float)}, 2)
    assert periods[0]["methods"] == {} and "skipped" in periods[0]
    assert periods[1]["methods"]["m"]["auprc"] > 0


def test_metrics_by_period_records_time_span() -> None:
    ev = _load_evaluate_module()
    y = np.array([0, 1] * 10)
    times = np.arange(100, 120)
    periods = ev.metrics_by_period(y, {"m": y.astype(float)}, 2, time_values=times)
    assert (periods[0]["time_from"], periods[0]["time_to"]) == (100, 109)
    assert (periods[1]["time_from"], periods[1]["time_to"]) == (110, 119)


def test_metrics_by_period_matches_whole_set_when_one_period() -> None:
    ev = _load_evaluate_module()
    rng = np.random.default_rng(3)
    y = (rng.random(200) < 0.1).astype(int)
    scores = rng.random(200) + y * 0.3
    whole = ev.metrics_for(y, scores, budgets=(10,))
    one = ev.metrics_by_period(y, {"m": scores}, 1)[0]["methods"]["m"]
    assert one["auprc"] == pytest.approx(whole["auprc"])
    assert one["roc_auc"] == pytest.approx(whole["roc_auc"])


def test_by_period_table_marks_skipped_period() -> None:
    ev = _load_evaluate_module()
    y = np.array([0] * 10 + [0, 1] * 5)
    periods = ev.metrics_by_period(y, {"m": np.arange(20, dtype=float)}, 2)
    table = ev._by_period_table(periods, ["m"])
    assert "không có nhãn dương" in table
    assert table.count("\n") == 3  # tiêu đề + gạch + 2 giai đoạn


def test_metrics_by_period_formats_timestamps_to_minutes() -> None:
    ev = _load_evaluate_module()
    y = np.array([0, 1] * 10)
    times = pd.date_range("2026-03-18 02:48:40.388124", periods=20, freq="h").to_numpy()
    periods = ev.metrics_by_period(y, {"m": y.astype(float)}, 2, time_values=times)
    assert periods[0]["time_from"] == "2026-03-18 02:48"
    json.dumps(periods)  # phải ghi được ra manifest JSON


def test_evaluate_rejects_compare_baseline_with_no_baselines() -> None:
    """--compare-baseline so checkpoint với một baseline, nên không thể dùng khi
    baseline bị tắt — phải báo lỗi ngay thay vì chạy rồi mới hỏng."""
    ev = _load_evaluate_module()
    with pytest.raises(SystemExit):
        ev.main(["--source", "paysim", "--checkpoint", "x.pt",
                 "--compare-baseline", "gbdt_them_lich_su", "--no-baselines"])
