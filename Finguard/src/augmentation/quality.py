"""Injector + ba lớp kiểm soát chất lượng dữ liệu tổng hợp (Mục 6.4).

Lớp 1 — trước khi sinh: spec_schema.validate_spec (ở file khác).
Lớp 2 — sau khi sinh: kiểm định phân biệt (adversarial validation) +
         khoảng cách phân phối, cả hai ở file này.
Lớp 3 — chống rò rỉ: assert_no_trace_leakage + chia tập theo nhóm kịch
         bản (scenario_group_split).

Về kiểm định phân biệt: mục tiêu là AUC ≈ 0,5 trên các cột KHÔNG mang
ngữ nghĩa gian lận. Nếu so dòng tổng hợp với TOÀN BỘ dữ liệu nền thì
AUC cao là đương nhiên (gian lận vốn khác giao dịch bình thường) và con
số đó vô nghĩa. Vì vậy phép đo chính ở đây là:

    dòng gian lận TỔNG HỢP  vs  dòng gian lận THẬT

trên các đặc trưng trung tính (phần thập phân của số tiền, phần thập phân
của log số tiền, giờ trong ngày, kênh). Các đặc trưng này không nên phân
biệt được nếu dữ liệu tổng hợp không để lộ dấu vết nhân tạo (vd. số tiền
quá tròn, log-uniform lộ rõ, giờ rơi đúng biên khung thời gian).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from src.augmentation.spec_schema import CHAIN_ARCHETYPES, TRACE_ONLY_COLUMNS, ScenarioSpec

#: Cột thời gian của PaySim tính theo giờ; của IEEE-CIS tính theo giây.
#: Ngưỡng này phân biệt hai trường hợp (PaySim tối đa ~744 step).
STEP_HOURS_MAX = 1e5


@dataclass
class QualityReport:
    """Kết quả lớp kiểm soát 2 — đưa nguyên vẹn vào manifest."""

    adversarial_auc_vs_real_fraud: float
    adversarial_auc_vs_real_all: float
    n_synthetic: int
    n_real_fraud: int
    #: AUC khi thêm cả giờ trong ngày — CHỈ để chẩn đoán, không vào cổng
    #: đạt/không đạt, vì khung thời gian là lựa chọn có ngữ nghĩa của đặc tả.
    adversarial_auc_including_time: float = float("nan")
    distances: dict[str, dict[str, float]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        """Ba trạng thái, KHÔNG phải hai.

        - "dat"            : AUC trong [0,40; 0,60] — gần như không phân biệt được.
        - "khong_dat"      : AUC ngoài khoảng đó — có dấu vết nhân tạo.
        - "chua_ket_luan"  : không đủ mẫu để chạy phép kiểm (xem
          MIN_CLASS_FOR_ADVERSARIAL). Phải báo cáo đúng là chưa kết luận,
          không được trình bày như đã đạt.
        """
        auc = self.adversarial_auc_vs_real_fraud
        if np.isnan(auc):
            return "chua_ket_luan"
        return "dat" if 0.40 <= auc <= 0.60 else "khong_dat"

    @property
    def passes(self) -> bool:
        """True khi KHÔNG phát hiện dấu vết nhân tạo. Lưu ý: trạng thái
        "chua_ket_luan" cũng trả True để pipeline không bị chặn, nên khi báo
        cáo hãy đọc `status` chứ không đọc riêng cờ này."""
        return self.status != "khong_dat"

    def to_json(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "adversarial_auc_vs_real_fraud": self.adversarial_auc_vs_real_fraud,
            "adversarial_auc_vs_real_all": self.adversarial_auc_vs_real_all,
            "adversarial_auc_including_time_diagnostic": self.adversarial_auc_including_time,
            "pass_threshold": [0.40, 0.60],
            "min_class_for_conclusion": MIN_CLASS_FOR_ADVERSARIAL,
            "passes": self.passes,
            "n_synthetic": self.n_synthetic,
            "n_real_fraud": self.n_real_fraud,
            "distances": self.distances,
            "warnings": list(self.warnings),
        }


# --------------------------------------------------------------------------
# Trục thời gian và tương thích nền
# --------------------------------------------------------------------------

def time_axis(df: pd.DataFrame) -> tuple[str, float]:
    """Trả về (tên cột thời gian, số giây mỗi đơn vị) của dữ liệu nền."""
    if "timestamp" in df.columns:
        return "timestamp", 1.0
    if "step" in df.columns:
        unit = 3600.0 if float(df["step"].max()) < STEP_HOURS_MAX else 1.0
        return "step", unit
    raise KeyError("Dữ liệu nền cần có cột 'timestamp' hoặc 'step'")


def background_window(df: pd.DataFrame) -> tuple[datetime, float]:
    """Mốc đầu và độ dài (giây) cửa sổ thời gian của dữ liệu nền."""
    col, unit = time_axis(df)
    if col == "timestamp":
        ts = pd.to_datetime(df["timestamp"])
        return ts.min().to_pydatetime(), float((ts.max() - ts.min()).total_seconds())
    step = df["step"].astype(float)
    # Mốc quy ước cho dữ liệu chỉ có step tương đối — chỉ dùng nội bộ để
    # rải thời gian, rồi quy đổi lại về step khi chèn.
    return datetime(2026, 1, 1), float((step.max() - step.min()) * unit)


def check_background_compatibility(
    background: pd.DataFrame, specs: Sequence[ScenarioSpec]
) -> None:
    """Chặn các tổ hợp nền × kiểu hình sẽ tạo ra dấu vết nhân tạo rõ rệt.

    Trường hợp thật đã gặp: IEEE-CIS KHÔNG có counterparty_id (toàn bộ
    bằng None). Nếu chèn kiểu hình dạng chuỗi vào nền này, mọi dòng tổng
    hợp sẽ là dòng duy nhất có counterparty — bộ phân loại phân biệt được
    ngay, và chuỗi đồ thị cũng không nối vào đâu cả.

    Raises:
        ValueError: nếu tổ hợp không dùng được, kèm cách xử lý.
    """
    if "counterparty_id" not in background.columns:
        raise ValueError("Dữ liệu nền thiếu cột counterparty_id")
    has_counterparty = background["counterparty_id"].notna().any()
    chain_specs = [s for s in specs if s.archetype in CHAIN_ARCHETYPES]
    if chain_specs and not has_counterparty:
        names = sorted({s.archetype for s in chain_specs})
        raise ValueError(
            f"Dữ liệu nền không có counterparty_id thật (vd. IEEE-CIS), nhưng đặc tả "
            f"chứa kiểu hình dạng chuỗi {names}. Chèn vào đây sẽ để lộ dấu vết nhân tạo "
            "và chuỗi không nối vào mạng lưới nào. Hãy dùng nền có quan hệ gửi–nhận "
            "(synthetic_vn, PaySim), hoặc bỏ các kiểu hình dạng chuỗi khỏi đợt sinh này."
        )


# --------------------------------------------------------------------------
# Injector
# --------------------------------------------------------------------------

def inject(
    background: pd.DataFrame, synthetic: pd.DataFrame, is_synthetic_col: str = "is_augmented"
) -> pd.DataFrame:
    """Chèn dòng tổng hợp vào nền, đồng nhất định dạng và trục thời gian.

    - Quy đổi `timestamp` của dòng tổng hợp về đúng cột thời gian của nền
      (giữ nguyên nếu nền dùng timestamp; đổi sang `step` nếu nền dùng step).
    - Thêm cột đánh dấu `is_augmented` để truy vết và để kiểm định phân biệt.
    - Bổ sung cột thiếu ở hai phía thành NaN/None thay vì làm rơi cột.
    """
    if synthetic.empty:
        out = background.copy()
        out[is_synthetic_col] = False
        return out

    col, unit = time_axis(background)
    synth = synthetic.copy()
    if col == "step":
        origin, _ = background_window(background)
        step_min = float(background["step"].astype(float).min())
        delta_seconds = (pd.to_datetime(synth["timestamp"]) - pd.Timestamp(origin)).dt.total_seconds()
        synth["step"] = step_min + delta_seconds / unit
        synth = synth.drop(columns=["timestamp"])
    else:
        synth["timestamp"] = pd.to_datetime(synth["timestamp"])

    base = background.copy()
    base[is_synthetic_col] = False
    synth[is_synthetic_col] = True

    # Để pd.concat tự hợp nhất cột (outer join) và tự điền NaN cho cột chỉ
    # có ở một phía — vd. scenario_code chỉ có ở dòng tổng hợp. Không tự tạo
    # sẵn các cột toàn NaN, vì pandas cảnh báo về cách suy kiểu dữ liệu khi
    # nối các cột như vậy.
    out = pd.concat([base, synth], ignore_index=True)
    return out.sort_values(col, kind="stable").reset_index(drop=True)


# --------------------------------------------------------------------------
# Lớp 2: kiểm định phân biệt + khoảng cách phân phối
# --------------------------------------------------------------------------

def hour_of_day(df: pd.DataFrame) -> np.ndarray:
    """Giờ trong ngày, chấp nhận cả trục timestamp và trục step."""
    if "timestamp" in df.columns:
        return pd.to_datetime(df["timestamp"]).dt.hour.to_numpy(dtype=float)
    if "step" in df.columns:
        return np.mod(df["step"].astype(float).to_numpy(), 24.0)
    return np.zeros(len(df), dtype=float)


def neutral_features(df: pd.DataFrame, include_time: bool = False) -> pd.DataFrame:
    """Đặc trưng TRUNG TÍNH — không mang ngữ nghĩa gian lận.

    Chủ ý loại `amount` thô: số tiền lớn vốn là tín hiệu gian lận thật,
    nên đưa vào sẽ làm AUC cao vì lý do đúng đắn chứ không phải vì dấu
    vết nhân tạo. Thay vào đó dùng phần thập phân của số tiền và của
    log10(số tiền) — hai thứ chỉ phản ánh CÁCH SINH SỐ, không phản ánh
    mức độ rủi ro.

    Cũng chủ ý loại GIỜ TRONG NGÀY khỏi nhóm trung tính: khung thời gian
    (`day_part`) là một lựa chọn CÓ NGỮ NGHĨA của đặc tả — giao dịch lúc
    rạng sáng là dấu hiệu rủi ro thật, và chính giao diện dùng nó để giải
    thích. Nếu tính vào đây thì AUC cao vì lý do đúng đắn, làm cổng kiểm
    soát mất tác dụng. Đo thực tế trên synthetic_vn: AUC 0,656 khi có giờ,
    0,517 khi bỏ giờ — gần như không phân biệt được, đúng mục tiêu.

    Cùng lý do đó, KHÔNG dùng phần thập phân của log10(số tiền): đó là
    phần định trị (mantissa), tức là một hàm của ĐỘ LỚN số tiền, nên nó
    phát hiện lại chính việc tăng cường dữ liệu có chủ ý (bộ sinh cố tình
    quét đều các mức số tiền để phủ vùng thưa). Đo thực tế trên PaySim:
    log_amount_frac cho AUC 0,645 trong khi phân phối phần xu khớp rất tốt
    (KS 0,046) — tức là nó đang đo lệch phân phối số tiền, không đo dấu vết
    nhân tạo. Thay vào đó dùng PHẦN XU trên lưới lượng tử hóa, là dấu vết
    của cách sinh số chứ không phải của độ lớn.

    Phân phối giờ và phân phối số tiền KHÔNG bị bỏ qua: assess_quality()
    báo cáo riêng khoảng cách phân phối của chúng, để mọi lệch lạc vẫn nhìn
    thấy được mà không trộn vào cổng đạt/không đạt.
    """
    amount = df["amount"].astype(float).to_numpy()
    feats = pd.DataFrame({
        # Phần xu trên lưới 1/100 — dấu vết của CÁCH SINH SỐ, không phản
        # ánh độ lớn giao dịch.
        "amount_cents": np.mod(np.round(np.modf(amount)[0] * 100.0), 100.0),
        "amount_round_1000": (np.mod(amount, 1000.0) < 1e-6).astype(float),
    })
    if "channel" in df.columns:
        feats["channel_code"] = pd.Categorical(df["channel"]).codes
    if include_time:
        feats["hour"] = hour_of_day(df)
    return feats.fillna(0.0)


#: Số mẫu tối thiểu mỗi lớp để kiểm định phân biệt có ý nghĩa. Đo thực tế:
#: với 28 dòng gian lận thật, tập kiểm chỉ còn ~8 dòng và AUC dao động
#: 0,51–0,63 chỉ vì đổi seed — tức là con số không kết luận được gì. Dưới
#: ngưỡng này, assess_quality báo "không đủ mẫu" thay vì đạt/không đạt.
MIN_CLASS_FOR_ADVERSARIAL = 50

#: Số lần chia tập lặp lại để lấy AUC trung bình — giảm phương sai của
#: phép đo, thay vì phụ thuộc vào một lần chia duy nhất.
ADVERSARIAL_REPEATS = 5


def _auc(x: pd.DataFrame, y: np.ndarray, seed: int = 0) -> float:
    """AUC trung bình của bộ phân biệt qua nhiều lần chia tập.

    Trả về NaN khi lớp nhỏ hơn MIN_CLASS_FOR_ADVERSARIAL — không đủ mẫu để
    kết luận, và báo một con số trong trường hợp đó dễ gây hiểu sai hơn là
    không báo.
    """
    y = y.astype(int)
    if min(int(y.sum()), int((y == 0).sum())) < MIN_CLASS_FOR_ADVERSARIAL:
        return float("nan")
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    features = x.to_numpy(dtype=float)
    scores: list[float] = []
    for offset in range(ADVERSARIAL_REPEATS):
        x_tr, x_te, y_tr, y_te = train_test_split(
            features, y, test_size=0.3, random_state=seed + offset, stratify=y)
        if len(np.unique(y_te)) < 2:
            continue
        clf = HistGradientBoostingClassifier(max_iter=120, random_state=seed + offset)
        clf.fit(x_tr, y_tr)
        scores.append(float(roc_auc_score(y_te, clf.predict_proba(x_te)[:, 1])))
    return float(np.mean(scores)) if scores else float("nan")


def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    """Thống kê Kolmogorov–Smirnov hai mẫu, tự cài để không thêm phụ thuộc."""
    a = np.sort(np.asarray(a, dtype=float))
    b = np.sort(np.asarray(b, dtype=float))
    if a.size == 0 or b.size == 0:
        return float("nan")
    pooled = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, pooled, side="right") / a.size
    cdf_b = np.searchsorted(b, pooled, side="right") / b.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


def js_distance(a: np.ndarray, b: np.ndarray, bins: int = 30) -> float:
    """Khoảng cách Jensen–Shannon trên histogram dùng chung biên bin."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0 or b.size == 0:
        return float("nan")
    lo, hi = float(min(a.min(), b.min())), float(max(a.max(), b.max()))
    if hi <= lo:
        return 0.0
    edges = np.linspace(lo, hi, bins + 1)
    p = np.histogram(a, bins=edges)[0].astype(float) + 1e-12
    q = np.histogram(b, bins=edges)[0].astype(float) + 1e-12
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    kl = lambda u, v: float(np.sum(u * np.log2(u / v)))  # noqa: E731
    return float(np.sqrt(max(0.5 * kl(p, m) + 0.5 * kl(q, m), 0.0)))


def assess_quality(
    injected: pd.DataFrame,
    is_synthetic_col: str = "is_augmented",
    seed: int = 0,
) -> QualityReport:
    """Chạy toàn bộ lớp kiểm soát 2 trên tập đã chèn."""
    synth_mask = injected[is_synthetic_col].fillna(False).astype(bool).to_numpy()
    label = injected["label"].fillna(False).astype(bool).to_numpy()
    warnings: list[str] = []

    synth_fraud = injected[synth_mask & label]
    real_fraud = injected[~synth_mask & label]
    real_all = injected[~synth_mask]

    def auc_between(a: pd.DataFrame, b: pd.DataFrame, include_time: bool = False) -> float:
        if a.empty or b.empty:
            return float("nan")
        both = pd.concat([a, b], ignore_index=True)
        y = np.concatenate([np.ones(len(a), dtype=int), np.zeros(len(b), dtype=int)])
        return _auc(neutral_features(both, include_time=include_time), y, seed=seed)

    auc_fraud = auc_between(synth_fraud, real_fraud)
    auc_all = auc_between(injected[synth_mask], real_all)
    auc_time = auc_between(synth_fraud, real_fraud, include_time=True)

    if np.isnan(auc_fraud):
        warnings.append(
            "KHÔNG ĐỦ MẪU để kết luận về kiểm định phân biệt trên nhóm gian lận "
            f"(tổng hợp {len(synth_fraud)}, gian lận thật {len(real_fraud)}; cần ≥ "
            f"{MIN_CLASS_FOR_ADVERSARIAL} mỗi bên). Đây là trạng thái 'chưa kết luận', "
            "KHÔNG phải 'đạt' — khi báo cáo phải nói rõ. Muốn kết luận được thì chạy "
            "trên nền có nhiều gian lận thật hơn (vd. PaySim, IEEE-CIS) hoặc mở rộng "
            "lát dữ liệu nền."
        )

    distances: dict[str, dict[str, float]] = {}
    if not synth_fraud.empty and not real_fraud.empty:
        for name, getter in (
            ("amount", lambda d: d["amount"].astype(float).to_numpy()),
            ("log_amount", lambda d: np.log10(
                np.clip(d["amount"].astype(float).to_numpy(), 1e-9, None))),
            # Giờ trong ngày và phần xu là CHẨN ĐOÁN, không vào cổng
            # đạt/không đạt — xem docstring neutral_features.
            ("hour_of_day", hour_of_day),
            ("amount_cents", lambda d: np.mod(
                np.round(np.modf(d["amount"].astype(float).to_numpy())[0] * 100.0), 100.0)),
        ):
            distances[name] = {
                "ks": ks_statistic(getter(synth_fraud), getter(real_fraud)),
                "js": js_distance(getter(synth_fraud), getter(real_fraud)),
            }

    report = QualityReport(
        adversarial_auc_vs_real_fraud=auc_fraud,
        adversarial_auc_vs_real_all=auc_all,
        adversarial_auc_including_time=auc_time,
        n_synthetic=int(synth_mask.sum()),
        n_real_fraud=int(len(real_fraud)),
        distances=distances,
        warnings=warnings,
    )
    if not report.passes:
        report.warnings.append(
            f"AUC phân biệt = {auc_fraud:.3f}, ngoài khoảng [0,40; 0,60] — dữ liệu tổng hợp "
            "đang để lộ dấu vết nhân tạo trên các cột trung tính. Xem distances để biết "
            "phân phối nào lệch, rồi chỉnh simulator hoặc biên hiệu chỉnh."
        )
    amount_ks = distances.get("amount", {}).get("ks")
    if amount_ks is not None and not np.isnan(amount_ks) and amount_ks > 0.15:
        report.warnings.append(
            f"Khoảng cách phân phối số tiền KS = {amount_ks:.3f}: số tiền của dòng tổng hợp "
            "lệch rõ so với gian lận thật. Đây CÓ THỂ là chủ ý (bộ sinh quét đều các mức số "
            "tiền để phủ vùng thưa) — nhưng phải nói rõ khi báo cáo, vì nó làm phân phối huấn "
            "luyện khác phân phối thực tế."
        )
    return report


# --------------------------------------------------------------------------
# Lớp 3: chống rò rỉ
# --------------------------------------------------------------------------

def assert_no_trace_leakage(feature_columns: Iterable[str]) -> None:
    """Khẳng định các cột truy vết không bị dùng làm ĐẶC TRƯNG ĐẦU VÀO.

    Dùng chúng làm NHÃN cho đầu phân loại tình huống là hợp lệ và được
    khuyến khích; dùng làm đầu vào là rò rỉ vì mô hình sẽ đọc thẳng đáp án.

    Raises:
        ValueError: nếu có cột truy vết trong danh sách đặc trưng.
    """
    offenders = sorted(set(feature_columns) & set(TRACE_ONLY_COLUMNS) | (
        set(feature_columns) & {"is_augmented", "label"}))
    if offenders:
        raise ValueError(
            f"Cột {offenders} là cột truy vết/nhãn, không được dùng làm đặc trưng đầu vào. "
            "Xem src/augmentation/__init__.py mục CHỐNG RÒ RỈ."
        )


def scenario_group_split(
    df: pd.DataFrame, holdout_archetypes: Sequence[str], scenario_col: str = "fraud_scenario_type"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chia theo NHÓM KỊCH BẢN (Mục 6.4): các kiểu hình giữ lại không xuất
    hiện trong phần huấn luyện, dùng để đo khả năng khái quát hóa sang kiểu
    hình chưa từng thấy (Mục 6.5).

    Returns:
        (phần huấn luyện — đã loại kiểu hình giữ lại,
         phần giữ lại — chỉ các dòng thuộc kiểu hình đó cùng toàn bộ dòng âm)
    """
    if scenario_col not in df.columns:
        raise KeyError(f"Thiếu cột '{scenario_col}' để chia theo nhóm kịch bản")
    held = df[scenario_col].isin(list(holdout_archetypes))
    is_fraud = df["label"].fillna(False).astype(bool)
    train_part = df[~held].reset_index(drop=True)
    holdout_part = df[held | ~is_fraud].reset_index(drop=True)
    return train_part, holdout_part
