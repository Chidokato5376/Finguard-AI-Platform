"""Synthetic dataset có phiên bản + manifest (Mục 6.3).

Mỗi lần sinh tạo ra một thư mục bất biến:

    data/synthetic/<version>/
        rows.parquet      — chỉ các dòng TỔNG HỢP (chưa chèn vào nền)
        specs.json        — toàn bộ đặc tả đã dùng, kèm spec_id và nguồn gốc
        manifest.json     — seed, phiên bản prompt, nhà cung cấp, biên hiệu
                            chỉnh và nguồn của từng biên, tỷ lệ đặc tả bị
                            loại, báo cáo kiểm soát chất lượng, commit, phiên
                            bản thư viện

Vì sao tách rows ra khỏi tập đã chèn: cùng một tập tổng hợp phải chèn
được vào nhiều nền khác nhau (synthetic_vn, PaySim) mà không sinh lại —
giữ cho so sánh giữa các cấu hình là so sánh công bằng.

Manifest KHÔNG BAO GIỜ chứa giá trị khóa API, chỉ chứa tên biến môi
trường (xem llm_adapter.LLMConfig.to_json).
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from src.augmentation.calibration import Calibration
from src.augmentation.quality import QualityReport
from src.augmentation.spec_schema import ScenarioSpec, SpecRejection
from src.tracking import git_commit, library_versions

DEFAULT_ROOT = Path("data/synthetic")
MANIFEST_NAME = "manifest.json"
ROWS_NAME = "rows.parquet"
SPECS_NAME = "specs.json"


@dataclass
class DatasetManifest:
    """Toàn bộ thông tin cần để tái lập một dataset tổng hợp."""

    version: str
    created_utc: str
    seed: int
    origin_mix: dict[str, int]
    n_rows: int
    n_fraud_rows: int
    archetype_counts: dict[str, int]
    n_specs_accepted: int
    n_specs_rejected: int
    rejection_histogram: dict[str, int]
    rejection_examples: list[dict[str, Any]]
    calibration: dict[str, Any]
    llm: dict[str, Any]
    quality: dict[str, Any]
    background_source: str
    git_commit: str = field(default_factory=git_commit)
    libraries: dict[str, str] = field(default_factory=library_versions)
    argv: list[str] = field(default_factory=lambda: list(sys.argv))
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "created_utc": self.created_utc,
            "seed": self.seed,
            "origin_mix": self.origin_mix,
            "n_rows": self.n_rows,
            "n_fraud_rows": self.n_fraud_rows,
            "archetype_counts": self.archetype_counts,
            "specs": {
                "accepted": self.n_specs_accepted,
                "rejected": self.n_specs_rejected,
                "rejection_rate": (
                    self.n_specs_rejected / (self.n_specs_accepted + self.n_specs_rejected)
                    if (self.n_specs_accepted + self.n_specs_rejected) else 0.0
                ),
                "rejection_histogram": self.rejection_histogram,
                "rejection_examples": self.rejection_examples,
            },
            "calibration": self.calibration,
            "llm": self.llm,
            "quality": self.quality,
            "background_source": self.background_source,
            "reproducibility": {
                "git_commit": self.git_commit,
                "libraries": self.libraries,
                "argv": self.argv,
            },
            "notes": self.notes,
        }


def make_version(specs: Sequence[ScenarioSpec], seed: int, calibration: Calibration) -> str:
    """Phiên bản = thời điểm + băm nội dung, nên vừa dễ đọc vừa bất biến."""
    blob = json.dumps(
        {
            "specs": sorted(s.spec_id for s in specs),
            "seed": seed,
            "bounds": {k: list(v) for k, v in sorted(calibration.bounds.items())},
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:8]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    return f"{stamp}-{digest}"


def write_versioned(
    rows: pd.DataFrame,
    specs: Sequence[ScenarioSpec],
    rejections: Sequence[SpecRejection],
    calibration: Calibration,
    quality: QualityReport | None,
    llm_info: dict[str, Any],
    seed: int,
    background_source: str,
    root: str | Path = DEFAULT_ROOT,
    notes: Sequence[str] = (),
) -> Path:
    """Ghi dataset + manifest, trả về thư mục phiên bản.

    Raises:
        FileExistsError: nếu phiên bản đã tồn tại — dataset là bất biến,
            muốn sinh lại thì đổi seed hoặc đặc tả.
    """
    version = make_version(specs, seed, calibration)
    out_dir = Path(root) / version
    if out_dir.exists():
        raise FileExistsError(
            f"Phiên bản {version} đã tồn tại tại {out_dir}. Dataset tổng hợp là bất biến — "
            "đổi seed hoặc đặc tả để tạo phiên bản mới thay vì ghi đè."
        )
    out_dir.mkdir(parents=True)

    rows.to_parquet(out_dir / ROWS_NAME, index=False)
    (out_dir / SPECS_NAME).write_text(
        json.dumps([s.to_json() for s in specs], ensure_ascii=False, indent=2), encoding="utf-8")

    origin_mix: dict[str, int] = {}
    for s in specs:
        origin_mix[s.origin] = origin_mix.get(s.origin, 0) + 1
    hist: dict[str, int] = {}
    for r in rejections:
        hist[r.reason_code] = hist.get(r.reason_code, 0) + 1

    label = rows["label"].fillna(False).astype(bool) if "label" in rows.columns else pd.Series(dtype=bool)
    archetype_counts = (
        rows["fraud_scenario_type"].value_counts().to_dict()
        if "fraud_scenario_type" in rows.columns else {}
    )

    manifest = DatasetManifest(
        version=version,
        created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        seed=seed,
        origin_mix=origin_mix,
        n_rows=int(len(rows)),
        n_fraud_rows=int(label.sum()) if len(label) else 0,
        archetype_counts={str(k): int(v) for k, v in archetype_counts.items()},
        n_specs_accepted=len(specs),
        n_specs_rejected=len(rejections),
        rejection_histogram=hist,
        rejection_examples=[
            {"reason_code": r.reason_code, "detail": r.detail} for r in list(rejections)[:10]
        ],
        calibration=calibration.to_json(),
        llm=llm_info,
        quality=quality.to_json() if quality is not None else {"status": "not_run"},
        background_source=background_source,
        notes=list(notes),
    )
    (out_dir / MANIFEST_NAME).write_text(
        json.dumps(manifest.to_json(), ensure_ascii=False, indent=2), encoding="utf-8")
    return out_dir


def load_versioned(version_dir: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Đọc lại một phiên bản: (các dòng tổng hợp, manifest)."""
    version_dir = Path(version_dir)
    rows = pd.read_parquet(version_dir / ROWS_NAME)
    manifest = json.loads((version_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    return rows, manifest


def latest_version(root: str | Path = DEFAULT_ROOT) -> Path | None:
    """Thư mục phiên bản mới nhất theo tên (tên có tiền tố thời gian)."""
    root = Path(root)
    if not root.exists():
        return None
    versions = sorted(p for p in root.iterdir() if (p / MANIFEST_NAME).exists())
    return versions[-1] if versions else None
