"""Bảng xếp hạng (scripts/leaderboard.py): các phép đo phải khớp sklearn."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import leaderboard as lb  # noqa: E402


@pytest.mark.parametrize("ties", [False, True])
def test_weighted_ap_khop_sklearn(ties):
    rng = np.random.default_rng(0)
    y = rng.random(5000) < 0.02
    scores = rng.random(5000) + 0.5 * y
    if ties:
        scores = np.round(scores, 1)                      # rất nhiều điểm bằng nhau
    order, boundary = lb.ranking_prep(scores)
    assert lb.weighted_ap(y, order, boundary) == pytest.approx(
        average_precision_score(y, scores), abs=1e-12)
    w = rng.poisson(1.0, len(y)).astype(float)
    assert lb.weighted_ap(y, order, boundary, w) == pytest.approx(
        average_precision_score(y, scores, sample_weight=w), abs=1e-12)


def test_f1_tot_nhat_khop_quet_nguong():
    rng = np.random.default_rng(1)
    y = rng.random(3000) < 0.05
    scores = np.round(rng.random(3000) + 0.4 * y, 2)
    got = lb.best_f1(y, *lb.ranking_prep(scores))
    want = max(f1_score(y, scores >= t) for t in np.unique(scores))
    assert got["f1_best"] == pytest.approx(want, abs=1e-12)


def test_xep_hang_kieu_thi_dau():
    assert lb.rank_desc({"a": 0.5, "b": 0.9, "c": 0.5, "d": 0.1}) == {"b": 1, "a": 2, "c": 2, "d": 4}


def test_bootstrap_ghep_cap_cung_mo_hinh_bang_khong():
    rng = np.random.default_rng(2)
    y = rng.random(4000) < 0.03
    s = rng.random(4000) + y
    ci = lb.paired_bootstrap(y, {"a": s, "b": s.copy(), "c": rng.random(4000)}, "a", 50, 0)
    assert ci["b"] == {"lo": 0.0, "hi": 0.0}
    assert ci["c"]["hi"] < 0
