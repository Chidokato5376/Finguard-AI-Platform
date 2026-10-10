"""Unit tests cho src/augmentation/ — tầng tăng cường dữ liệu bằng kịch bản
tổng hợp (Mục 6).

Trọng tâm là các BẤT BIẾN mà tầng này phải giữ, vì đó là chỗ dễ vỡ im lặng:
  - validator thực sự chặn ba loại vi phạm;
  - simulator tất định và gán nhãn đúng theo cấu tạo;
  - prompt không mang dữ liệu giao dịch thật;
  - transport offline không bao giờ bị dán nhãn "llm";
  - dòng tổng hợp rải đều trên cửa sổ thời gian (không tạo khối thời gian riêng);
  - cột truy vết không lọt vào danh sách đặc trưng;
  - tập đánh giá không chứa dòng tổng hợp.

Không test nào gọi mạng.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.augmentation.calibration import Calibration, calibrate_from_real
from src.augmentation.dataset import load_versioned, write_versioned
from src.augmentation.llm_adapter import (
    OfflineVariantTransport,
    SpecGenerator,
    assert_prompt_free_of_raw_data,
    build_user_prompt,
    extract_payload,
)
from src.augmentation.mining import MiningConfig, apply_focus, mine_missed_specs
from src.augmentation.protocol import (
    assert_eval_set_is_real,
    decide_llm_claim,
    false_positive_rate_at_budget,
    pr_auc,
    recall_at_budget,
)
from src.augmentation.quality import (
    assert_no_trace_leakage,
    background_window,
    check_background_compatibility,
    inject,
    scenario_group_split,
    time_axis,
)
from src.augmentation.rule_specs import build_rule_specs
from src.augmentation.simulator import (
    channel_override_needed,
    infer_amount_decimals,
    simulate_all,
    simulate_spec,
)
from src.augmentation.spec_schema import validate_spec

# --------------------------------------------------------------------------
# Dụng cụ
# --------------------------------------------------------------------------

def _calibration() -> Calibration:
    """Biên đặt tay để test không phụ thuộc vào phân vị của dữ liệu mẫu."""
    return Calibration(
        bounds={
            "amount_vnd": (1_000_000.0, 50_000_000.0),
            "hop_gap_minutes": (2.0, 180.0),
            "chain_length": (3.0, 6.0),
            "amount_decay": (0.92, 0.99),
            "burst_count": (2.0, 5.0),
            "n_cases": (1.0, 4.0),
        },
        sources={"amount_vnd": "du_lieu_thuc (test)", "chain_length": "tri_thuc_mien"},
        unit="VND (test)",
        n_real_rows=400,
        n_real_fraud=25,
    )


def _raw(**over) -> dict:
    raw = {
        "archetype": "money_mule_chain",
        "description": "dac ta test",
        "n_cases": 2,
        "chain_length": 4,
        "amount_vnd_min": 2_000_000.0,
        "amount_vnd_max": 20_000_000.0,
        "hop_gap_minutes_min": 5.0,
        "hop_gap_minutes_max": 60.0,
        "amount_decay": 0.95,
        "day_part": "dem",
        "channel": "domestic",
        "new_device": True,
        "foreign_ip": False,
        "burst_count": 3,
    }
    raw.update(over)
    return raw


def _spec(**over):
    spec, rejection = validate_spec(_raw(**over), _calibration().bounds, origin="rule")
    assert rejection is None, rejection
    return spec


def _background(n_accounts: int = 8, n_rows: int = 400, n_fraud: int = 25,
                days: int = 60, seed: int = 7, step_axis: bool = False) -> pd.DataFrame:
    rng = random.Random(seed)
    start = datetime(2026, 1, 1)
    accounts = [f"ACC{i:03d}" for i in range(n_accounts)]
    rows = []
    for i in range(n_rows):
        sender = rng.choice(accounts)
        receiver = rng.choice([a for a in accounts if a != sender])
        is_fraud = i < n_fraud
        rows.append({
            "account_id": sender,
            "counterparty_id": receiver,
            "timestamp": start + timedelta(seconds=rng.uniform(0, days * 86400)),
            "amount": rng.uniform(20_000_000, 60_000_000) if is_fraud
            else rng.uniform(200_000, 8_000_000),
            "channel": rng.choice(["domestic", "e_wallet"]),
            "device_id": f"dev_{sender}",
            "ip_country": "VN",
            "label": is_fraud,
            "source": "synthetic_vn",
        })
    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    if step_axis:
        hours = (df["timestamp"] - df["timestamp"].min()).dt.total_seconds() / 3600.0
        df = df.drop(columns=["timestamp"])
        df["step"] = hours.round().astype(int)
        df = df.sort_values("step").reset_index(drop=True)
    return df


# --------------------------------------------------------------------------
# Spec validator (lớp kiểm soát 1)
# --------------------------------------------------------------------------

def test_valid_spec_accepted() -> None:
    spec, rejection = validate_spec(_raw(), _calibration().bounds, origin="rule")
    assert rejection is None
    assert spec is not None and spec.origin == "rule"
    assert spec.scenario_code.startswith("money_mule_chain:rule:")


def test_rejects_value_outside_closed_vocabulary() -> None:
    for field, bad in (("archetype", "khong_ton_tai"), ("channel", "crypto"),
                       ("day_part", "nua_dem")):
        spec, rejection = validate_spec(
            _raw(**{field: bad}), _calibration().bounds, origin="rule")
        assert spec is None
        assert rejection.reason_code == "closed_vocabulary", field


def test_rejects_key_outside_template() -> None:
    raw = _raw()
    raw["extra_field"] = 1
    spec, rejection = validate_spec(raw, _calibration().bounds, origin="llm")
    assert spec is None and rejection.reason_code == "schema"


def test_rejects_missing_required_key() -> None:
    raw = _raw()
    del raw["amount_decay"]
    spec, rejection = validate_spec(raw, _calibration().bounds, origin="llm")
    assert spec is None and rejection.reason_code == "schema"


def test_rejects_amount_outside_calibrated_bounds() -> None:
    spec, rejection = validate_spec(
        _raw(amount_vnd_max=900_000_000.0), _calibration().bounds, origin="llm")
    assert spec is None and rejection.reason_code == "out_of_bounds"


def test_non_chain_archetype_must_have_chain_length_two() -> None:
    """Hồi quy: biên chain_length (3–6) từ tri thức miền KHÔNG được áp cho
    kiểu hình không phải chuỗi, nếu không mọi đặc tả chiếm quyền/lừa đầu tư
    đều bị loại oan."""
    ok, rejection = validate_spec(
        _raw(archetype="account_takeover", chain_length=2), _calibration().bounds, origin="rule")
    assert rejection is None and ok is not None

    bad, rejection = validate_spec(
        _raw(archetype="account_takeover", chain_length=4), _calibration().bounds, origin="rule")
    assert bad is None and rejection.reason_code == "schema"


def test_spec_id_is_content_addressed() -> None:
    assert _spec().spec_id == _spec().spec_id
    assert _spec().spec_id != _spec(amount_decay=0.93).spec_id


def test_rule_specs_are_all_valid() -> None:
    """Đường cơ sở (ii) phải sinh ra đặc tả hợp lệ hoàn toàn — nếu có đặc tả
    bị loại thì biên hiệu chỉnh đang xung đột với biên tuyệt đối."""
    specs, rejections = build_rule_specs(_calibration(), n_cases_per_spec=1)
    assert rejections == []
    assert specs and all(s.origin == "rule" for s in specs)
    assert {s.archetype for s in specs} == {
        "money_mule_chain", "account_takeover", "investment_romance_scam",
        "smurfing_structuring"}


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------

def test_calibration_records_source_of_each_bound() -> None:
    cal = calibrate_from_real(_background(), unit="VND (test)")
    assert cal.bounds["amount_vnd"][0] < cal.bounds["amount_vnd"][1]
    assert cal.sources["amount_vnd"].startswith("du_lieu_thuc")
    # Hai tham số này không suy được từ dữ liệu phẳng không có chuỗi
    assert cal.sources["chain_length"].startswith("tri_thuc_mien")
    assert cal.sources["amount_decay"].startswith("tri_thuc_mien")
    assert cal.n_real_fraud == 25


def test_calibration_requires_positive_labels() -> None:
    df = _background(n_fraud=0)
    with pytest.raises(ValueError, match="không có nhãn dương"):
        calibrate_from_real(df, unit="VND (test)")


# --------------------------------------------------------------------------
# Simulator
# --------------------------------------------------------------------------

def test_simulator_is_deterministic() -> None:
    bg = _background()
    origin, horizon = background_window(bg)
    args = ([_spec()], list(bg["account_id"].unique()), origin, horizon, 11)
    first, second = simulate_all(*args), simulate_all(*args)
    pd.testing.assert_frame_equal(first, second)


def test_simulator_labels_by_construction() -> None:
    bg = _background()
    origin, horizon = background_window(bg)
    mule = simulate_spec(_spec(n_cases=3), list(bg["account_id"].unique()), origin, horizon, 3)
    assert mule["label"].all(), "mọi chặng của chuỗi mule đều là gian lận"
    assert (mule["fraud_scenario_type"] == "money_mule_chain").all()

    ato_spec = _spec(archetype="account_takeover", chain_length=2, burst_count=4)
    ato = simulate_spec(ato_spec, list(bg["account_id"].unique()), origin, horizon, 3)
    # Mỗi ca: 3 dòng lịch sử label=False + đúng 1 dòng đột biến label=True
    assert int(ato["label"].sum()) == ato_spec.n_cases
    assert int((~ato["label"]).sum()) == ato_spec.n_cases * 3


def test_amount_pool_used_only_when_large_enough() -> None:
    """Bể mẫu quá nhỏ thì bỏ qua, để không bốc lại cùng vài giá trị hàng
    chục lần (chất lượng của cách bốc đó không kiểm chứng được)."""
    bg = _background()
    origin, horizon = background_window(bg)
    accounts = list(bg["account_id"].unique())
    spec = _spec(n_cases=4)

    tiny_pool = [5_000_000.0 + i for i in range(5)]
    with_tiny = simulate_spec(spec, accounts, origin, horizon, 31, amount_pool=tiny_pool)
    without = simulate_spec(spec, accounts, origin, horizon, 31)
    pd.testing.assert_frame_equal(with_tiny, without)

    big_pool = np.linspace(2_000_000, 20_000_000, 400).tolist()
    with_big = simulate_spec(spec, accounts, origin, horizon, 31, amount_pool=big_pool)
    assert not with_big["amount"].equals(without["amount"])
    # Vẫn tất định
    again = simulate_spec(spec, accounts, origin, horizon, 31, amount_pool=big_pool)
    pd.testing.assert_frame_equal(with_big, again)


def test_amount_precision_matches_background() -> None:
    """Hồi quy: PaySim lượng tử hóa số tiền về 2 chữ số thập phân, còn
    simulator phát float đầy đủ — khi đó bộ phân biệt tách được 100% chỉ
    bằng phần thập phân (đo thực tế AUC 1,000)."""
    quantised = np.round(np.random.default_rng(0).uniform(1e4, 1e7, 500), 2)
    assert infer_amount_decimals(quantised) == 2
    assert infer_amount_decimals(np.random.default_rng(0).uniform(1e4, 1e7, 500)) is None
    assert infer_amount_decimals(np.arange(100.0)) == 0

    bg = _background()
    origin, horizon = background_window(bg)
    rows = simulate_all([_spec(n_cases=3)], list(bg["account_id"].unique()),
                        origin, horizon, 5, amount_decimals=2)
    cents = rows["amount"].to_numpy() * 100
    assert np.allclose(cents, np.round(cents)), "số tiền phải khớp lưới 2 chữ số thập phân"


def test_channel_overridden_when_background_uses_other_vocabulary() -> None:
    """Hồi quy: PaySim ghi loại giao dịch PaySim vào cột `channel`, nên giá
    trị từ từ vựng Channel không tồn tại trong nền và bị tách ngay."""
    paysim_like = ["cash_out", "transfer"] * 20
    assert channel_override_needed(["domestic"], paysim_like) is True
    assert channel_override_needed(["domestic"], ["domestic", "e_wallet"]) is False

    bg = _background()
    origin, horizon = background_window(bg)
    spec = _spec(n_cases=4)
    rows = simulate_all([spec], list(bg["account_id"].unique()), origin, horizon, 6,
                        channel_pool=paysim_like)
    assert set(rows["channel"]) <= {"cash_out", "transfer"}
    # Không ghi đè khi nền đã dùng cùng từ vựng
    same = simulate_all([spec], list(bg["account_id"].unique()), origin, horizon, 6,
                        channel_pool=["domestic"] * 20)
    assert set(same["channel"]) == {spec.channel}


def test_quality_status_is_three_valued() -> None:
    """Thiếu mẫu phải ra 'chưa kết luận', không được ra 'đạt'."""
    from src.augmentation.quality import QualityReport

    assert QualityReport(float("nan"), 0.5, 10, 3).status == "chua_ket_luan"
    assert QualityReport(0.52, 0.5, 400, 400).status == "dat"
    assert QualityReport(0.89, 0.5, 400, 400).status == "khong_dat"
    assert QualityReport(0.89, 0.5, 400, 400).passes is False


def test_simulator_chain_length_honoured() -> None:
    bg = _background()
    origin, horizon = background_window(bg)
    spec = _spec(n_cases=1, chain_length=5)
    rows = simulate_spec(spec, list(bg["account_id"].unique()), origin, horizon, 5)
    assert len(rows) == 4, "chuỗi 5 tài khoản tạo 4 chặng chuyển tiền"


def test_simulator_reuses_background_accounts() -> None:
    """Tái sử dụng mã tài khoản để cạnh tổng hợp nối vào mạng lưới thật."""
    bg = _background()
    accounts = set(bg["account_id"].unique())
    origin, horizon = background_window(bg)
    rows = simulate_spec(_spec(n_cases=4), list(accounts), origin, horizon, 9)
    assert set(rows["account_id"]) & accounts, "phải có tài khoản nguồn lấy từ nền"
    assert any(str(c).startswith("SYN_") for c in rows["counterparty_id"]), \
        "tài khoản trung gian phải là mã mới"


def test_simulator_stays_inside_background_window() -> None:
    bg = _background()
    origin, horizon = background_window(bg)
    specs = [
        _spec(n_cases=3),
        _spec(archetype="account_takeover", chain_length=2, burst_count=5),
        _spec(archetype="investment_romance_scam", chain_length=2, burst_count=4),
    ]
    rows = simulate_all(specs, list(bg["account_id"].unique()), origin, horizon, 13)
    assert rows["timestamp"].min() >= pd.Timestamp(origin), \
        "không dòng nào được rơi trước mốc đầu cửa sổ nền"
    assert rows["timestamp"].max() <= pd.Timestamp(origin) + timedelta(seconds=horizon + 86400)


def test_fraud_spread_across_whole_timeline() -> None:
    """Hồi quy cho lỗi đã gặp thật: nếu dồn nhãn dương vào một khối thời gian,
    time-based split sẽ đẩy toàn bộ gian lận vào một tập duy nhất."""
    bg = _background()
    origin, horizon = background_window(bg)
    specs, _ = build_rule_specs(_calibration(), n_cases_per_spec=3)
    rows = simulate_all(specs, list(bg["account_id"].unique()), origin, horizon, 21)
    injected = inject(bg, rows)
    synth = injected[injected["is_augmented"].astype(bool)]
    thirds = pd.cut(
        (synth["timestamp"] - pd.Timestamp(origin)).dt.total_seconds(),
        bins=[-1, horizon / 3, 2 * horizon / 3, horizon + 86400], labels=["dau", "giua", "cuoi"])
    assert set(thirds.dropna().unique()) == {"dau", "giua", "cuoi"}


# --------------------------------------------------------------------------
# Adapter LLM: nguồn gốc và rào dữ liệu
# --------------------------------------------------------------------------

def test_offline_transport_never_claims_llm_origin() -> None:
    bg = _background()
    generator = SpecGenerator(_calibration(), transport=OfflineVariantTransport(seed=3))
    result = generator.propose(6, bg)
    assert result.origin == "rule", "transport không gọi model thì không được dán nhãn llm"
    assert result.specs, f"không đặc tả nào hợp lệ: {[r.detail for r in result.rejections]}"
    assert all(s.origin == "rule" for s in result.specs)


def test_offline_generator_rejection_rate_is_reported() -> None:
    generator = SpecGenerator(_calibration(), transport=OfflineVariantTransport(seed=1))
    result = generator.propose(8, _background())
    assert 0.0 <= result.rejection_rate <= 1.0
    assert set(result.rejection_histogram()) <= {
        "schema", "closed_vocabulary", "out_of_bounds", "parse"}


def test_prompt_carries_no_real_transaction_data() -> None:
    bg = _background()
    prompt = build_user_prompt(_calibration(), ("money_mule_chain",), 4,
                               ("account_id", "amount", "label"))
    assert_prompt_free_of_raw_data(prompt, bg)  # không được raise
    payload = extract_payload(prompt)
    assert "bounds" in payload and "allowed_values" in payload
    assert payload["n_specs"] == 4


def test_privacy_guard_detects_leaked_identifier() -> None:
    bg = _background()
    leaked = bg["account_id"].iloc[0]
    with pytest.raises(AssertionError, match="định danh"):
        assert_prompt_free_of_raw_data(f"ngu canh: tai khoan {leaked} co bat thuong", bg)


def test_privacy_guard_detects_leaked_amount() -> None:
    bg = _background()
    amount = bg["amount"].iloc[0]
    with pytest.raises(AssertionError, match="số tiền"):
        assert_prompt_free_of_raw_data(f"giao dich {amount:.2f} dong", bg)


# --------------------------------------------------------------------------
# Injector + tương thích nền
# --------------------------------------------------------------------------

def test_inject_converts_to_step_axis() -> None:
    bg = _background(step_axis=True)
    assert time_axis(bg) == ("step", 3600.0)
    origin, horizon = background_window(bg)
    rows = simulate_all([_spec(n_cases=2)], list(bg["account_id"].unique()), origin, horizon, 4)
    injected = inject(bg, rows)
    assert "timestamp" not in injected.columns
    assert injected["step"].is_monotonic_increasing
    synth = injected[injected["is_augmented"].astype(bool)]
    assert synth["step"].min() >= bg["step"].min()


def test_chain_archetype_refused_on_background_without_counterparty() -> None:
    """IEEE-CIS không có counterparty thật — chèn kiểu hình chuỗi vào đó sẽ
    để lộ dấu vết nhân tạo ngay lập tức."""
    bg = _background()
    bg["counterparty_id"] = None
    with pytest.raises(ValueError, match="counterparty_id"):
        check_background_compatibility(bg, [_spec()])
    # Kiểu hình không phải chuỗi thì vẫn cho qua
    check_background_compatibility(
        bg, [_spec(archetype="account_takeover", chain_length=2)])


# --------------------------------------------------------------------------
# Chống rò rỉ (lớp kiểm soát 3)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("column", ["fraud_scenario_type", "scenario_code", "spec_id",
                                    "is_augmented", "label"])
def test_trace_columns_cannot_be_features(column: str) -> None:
    with pytest.raises(ValueError, match="không được dùng làm đặc trưng"):
        assert_no_trace_leakage(["amount_norm", "channel_encoded", column])


def test_feature_columns_of_training_pass_leakage_check() -> None:
    from src.drre.train import FEATURE_COLUMNS

    assert_no_trace_leakage(FEATURE_COLUMNS)  # không được raise


def test_scenario_group_split_holds_out_archetype() -> None:
    bg = _background()
    origin, horizon = background_window(bg)
    specs, _ = build_rule_specs(_calibration(), n_cases_per_spec=1)
    rows = simulate_all(specs, list(bg["account_id"].unique()), origin, horizon, 2)
    injected = inject(bg, rows)
    train_part, holdout = scenario_group_split(injected, ["smurfing_structuring"])
    assert "smurfing_structuring" not in set(train_part["fraud_scenario_type"].dropna())
    assert "smurfing_structuring" in set(holdout["fraud_scenario_type"].dropna())


# --------------------------------------------------------------------------
# Dataset có phiên bản
# --------------------------------------------------------------------------

def test_manifest_records_provenance_without_api_key(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("FAKE_KEY_FOR_TEST", "sk-khong-duoc-ghi-ra-manifest")
    bg = _background()
    cal = _calibration()
    origin, horizon = background_window(bg)
    specs, rejections = build_rule_specs(cal, n_cases_per_spec=1)
    rows = simulate_all(specs, list(bg["account_id"].unique()), origin, horizon, 17)

    out_dir = write_versioned(
        rows=rows, specs=specs, rejections=rejections, calibration=cal, quality=None,
        llm_info={"provider": "deepseek", "api_key_env": "FAKE_KEY_FOR_TEST"},
        seed=17, background_source="synthetic_vn_train", root=tmp_path,
    )
    loaded_rows, manifest = load_versioned(out_dir)

    assert len(loaded_rows) == len(rows)
    assert manifest["seed"] == 17
    assert manifest["specs"]["accepted"] == len(specs)
    assert manifest["calibration"]["sources"]["chain_length"].startswith("tri_thuc_mien")
    assert manifest["origin_mix"] == {"rule": len(specs)}
    blob = (out_dir / "manifest.json").read_text(encoding="utf-8")
    assert "sk-khong-duoc-ghi-ra-manifest" not in blob, "manifest không được chứa giá trị khóa"
    assert "FAKE_KEY_FOR_TEST" in blob, "manifest nên ghi TÊN biến môi trường đã dùng"


def test_versioned_dataset_is_immutable(tmp_path) -> None:
    bg = _background()
    cal = _calibration()
    origin, horizon = background_window(bg)
    specs, rejections = build_rule_specs(cal, n_cases_per_spec=1)
    rows = simulate_all(specs, list(bg["account_id"].unique()), origin, horizon, 23)
    kwargs = dict(rows=rows, specs=specs, rejections=rejections, calibration=cal,
                  quality=None, llm_info={}, seed=23,
                  background_source="synthetic_vn_train", root=tmp_path)
    write_versioned(**kwargs)
    with pytest.raises(FileExistsError):
        write_versioned(**kwargs)


# --------------------------------------------------------------------------
# Giao thức đánh giá
# --------------------------------------------------------------------------

def test_eval_set_must_contain_no_synthetic_rows() -> None:
    real = _background(n_rows=50, n_fraud=5)
    assert_eval_set_is_real(real)  # không raise

    tainted = real.copy()
    tainted["is_augmented"] = [True] + [False] * (len(tainted) - 1)
    with pytest.raises(ValueError, match="dòng tổng hợp"):
        assert_eval_set_is_real(tainted)


def test_recall_and_fpr_at_budget() -> None:
    y = np.array([1, 1, 0, 0, 0])
    scores = np.array([0.9, 0.2, 0.8, 0.1, 0.05])
    # Ngân sách 2: chọn 0.9 (dương) và 0.8 (âm)
    assert recall_at_budget(y, scores, 2) == pytest.approx(0.5)
    assert false_positive_rate_at_budget(y, scores, 2) == pytest.approx(0.5)
    assert recall_at_budget(y, scores, 5) == pytest.approx(1.0)


def test_claim_gate_requires_three_seeds() -> None:
    rng = np.random.default_rng(0)
    y = (rng.random(300) < 0.1).astype(int)
    better = y + rng.normal(0, 0.5, size=300)  # tương quan mạnh với nhãn
    worse = rng.normal(0, 1.0, size=300)

    one_seed = decide_llm_claim(y, [better], [worse], metric=pr_auc, n_boot=60)
    assert one_seed.can_claim is False
    assert "seed" in one_seed.message

    three = decide_llm_claim(y, [better] * 3, [worse] * 3, metric=pr_auc, n_boot=60)
    assert three.n_seeds == 3
    assert three.can_claim is True, three.message


def test_claim_gate_refuses_when_ci_contains_zero() -> None:
    rng = np.random.default_rng(1)
    y = (rng.random(300) < 0.1).astype(int)
    same = rng.normal(0, 1.0, size=300)
    decision = decide_llm_claim(y, [same] * 3, [same] * 3, metric=pr_auc, n_boot=60)
    assert decision.can_claim is False
    assert decision.diff == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------
# Adversarial mining
# --------------------------------------------------------------------------

def test_mining_is_disabled_by_default() -> None:
    report = mine_missed_specs(pd.DataFrame(), [], MiningConfig())
    assert report.enabled is False
    assert "TẮT" in report.notes[0]


def test_apply_focus_never_widens_bounds() -> None:
    cal = _calibration()
    scored = pd.DataFrame({
        "risk_score": [1.0, 2.0, 3.0, 99.0] * 10,
        "label": [True, True, True, True] * 10,
        "spec_id": [_spec().spec_id] * 40,
    })
    report = mine_missed_specs(scored, [_spec()], MiningConfig(enabled=True))
    assert report.enabled is True
    focused = apply_focus(cal, report, MiningConfig(enabled=True))
    for param, (lo, hi) in focused.bounds.items():
        orig_lo, orig_hi = cal.bounds[param]
        assert lo >= orig_lo - 1e-9 and hi <= orig_hi + 1e-9, param
