"""Theo dõi thí nghiệm — gắn mã nguồn, phiên bản dữ liệu, seed và siêu
tham số vào từng lần chạy.

Vì sao cần: nếu không ghi lại những thứ này, một con số AUPRC không truy
được về đã chạy bằng commit nào, dữ liệu nào, seed nào — và khi đó
benchmark không tái lập được, dù mã nguồn có đúng.

Dùng ở hai chỗ: `src/drre/train.py` (mỗi lần huấn luyện) và
`src/augmentation/dataset.py` (mỗi phiên bản dữ liệu tổng hợp).
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

#: File lớn hơn mức này chỉ được băm một phần (đầu + cuối) để không mất
#: hàng phút cho mỗi lần chạy. Dấu vân tay khi đó được ghi rõ là "partial".
FULL_HASH_MAX_BYTES = 512 * 1024 * 1024
HASH_CHUNK = 1024 * 1024
PARTIAL_HASH_EDGE = 16 * 1024 * 1024

DEFAULT_RUNS_DIR = Path("reports/runs")


def git_commit() -> str:
    """Commit đang dùng, hoặc "unknown" nếu không chạy trong repo git."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10, check=False)
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        return "unknown"


def git_dirty() -> bool | None:
    """True nếu cây làm việc có thay đổi chưa commit.

    Quan trọng cho tính trung thực: khi cây làm việc "bẩn", riêng mã commit
    KHÔNG xác định được mã nguồn đã chạy. Trả None nếu không xác định được.
    """
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True,
            timeout=15, check=False)
        if out.returncode != 0:
            return None
        return bool(out.stdout.strip())
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        return None


def library_versions() -> dict[str, str]:
    versions = {"python": platform.python_version(), "pandas": pd.__version__}
    for name in ("numpy", "torch", "sklearn", "qiskit"):
        try:
            versions[name] = __import__(name).__version__
        except Exception:  # noqa: BLE001 — thiếu thư viện không được làm vỡ việc ghi manifest
            versions[name] = "absent"
    return versions


def file_fingerprint(path: str | Path) -> dict[str, Any]:
    """Dấu vân tay của một file dữ liệu: kích thước, thời điểm sửa, băm nội dung."""
    path = Path(path)
    if not path.exists():
        return {"path": str(path), "exists": False}
    size = path.stat().st_size
    digest = hashlib.sha256()
    partial = size > FULL_HASH_MAX_BYTES
    with path.open("rb") as handle:
        if partial:
            digest.update(handle.read(PARTIAL_HASH_EDGE))
            handle.seek(max(size - PARTIAL_HASH_EDGE, 0))
            digest.update(handle.read(PARTIAL_HASH_EDGE))
            digest.update(str(size).encode())
        else:
            for chunk in iter(lambda: handle.read(HASH_CHUNK), b""):
                digest.update(chunk)
    return {
        "path": str(path),
        "exists": True,
        "bytes": size,
        "modified_utc": datetime.fromtimestamp(
            path.stat().st_mtime, tz=timezone.utc).isoformat(timespec="seconds"),
        "sha256": digest.hexdigest()[:32],
        "hash_scope": "partial_head_tail" if partial else "full",
    }


def dataframe_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Thống kê tối thiểu để nhận ra mình đã train trên lát dữ liệu nào."""
    summary: dict[str, Any] = {"n_rows": int(len(df)), "columns": list(df.columns)}
    if "label" in df.columns:
        label = df["label"].fillna(False).astype(bool)
        summary["n_positive"] = int(label.sum())
        summary["positive_rate"] = float(label.mean())
    for col in ("step", "timestamp"):
        if col in df.columns:
            summary["time_col"] = col
            summary["time_min"] = str(df[col].min())
            summary["time_max"] = str(df[col].max())
            break
    if "account_id" in df.columns:
        summary["n_accounts"] = int(df["account_id"].nunique())
    return summary


@dataclass
class RunManifest:
    """Một lần chạy thí nghiệm, đủ thông tin để tái lập."""

    kind: str  # "drre_train" | "evaluation" | ...
    run_id: str = ""
    started_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    finished_utc: str | None = None
    duration_seconds: float | None = None
    argv: list[str] = field(default_factory=lambda: list(sys.argv))
    git: dict[str, Any] = field(
        default_factory=lambda: {"commit": git_commit(), "dirty": git_dirty()})
    libraries: dict[str, str] = field(default_factory=library_versions)
    device: str = "cpu"
    seed: int | None = None
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)
    results: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.run_id:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            self.run_id = f"{stamp}-{self.kind}"

    def finish(self, started_monotonic: float | None = None) -> None:
        self.finished_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if started_monotonic is not None:
            import time

            self.duration_seconds = round(time.monotonic() - started_monotonic, 1)

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "run_id": self.run_id,
            "started_utc": self.started_utc,
            "finished_utc": self.finished_utc,
            "duration_seconds": self.duration_seconds,
            "device": self.device,
            "seed": self.seed,
            "hyperparameters": self.hyperparameters,
            "data": self.data,
            "results": self.results,
            "reproducibility": {
                "git": self.git,
                "libraries": self.libraries,
                "argv": self.argv,
            },
            "notes": self.notes,
        }

    def write(self, runs_dir: str | Path = DEFAULT_RUNS_DIR) -> Path:
        runs_dir = Path(runs_dir)
        runs_dir.mkdir(parents=True, exist_ok=True)
        out_path = runs_dir / f"{self.run_id}.json"
        out_path.write_text(
            json.dumps(self.to_json(), ensure_ascii=False, indent=2), encoding="utf-8")
        return out_path

    def checkpoint_fields(self) -> dict[str, Any]:
        """Phần gọn của manifest để nhúng thẳng vào checkpoint, nhờ đó mở một
        file .pt là biết nó ra đời từ đâu."""
        return {
            "run_id": self.run_id,
            "git_commit": self.git.get("commit"),
            "git_dirty": self.git.get("dirty"),
            "seed": self.seed,
            "hyperparameters": self.hyperparameters,
            "data_fingerprint": {
                k: v.get("sha256") for k, v in self.data.items() if isinstance(v, dict)
            },
            "results": self.results,
        }
