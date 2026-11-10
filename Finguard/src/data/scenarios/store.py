"""Nơi lưu đặc trưng kịch bản: `<thư mục đã xử lý>/scenarios/<mô-đun>.<tập>.parquet`.

Tách khỏi parquet chính để không đụng tới dữ liệu mà các checkpoint đã học, và để mỗi
mô-đun được tính lại riêng. Mỗi file thẳng hàng theo dòng với `amlworld_<tập>.parquet`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.scenarios import REGISTRY

FOLDER = "scenarios"


def folder_of(processed_dir: str | Path) -> Path:
    return Path(processed_dir) / FOLDER


def part_path(processed_dir: str | Path, module: str, split: str) -> Path:
    return folder_of(processed_dir) / f"{module}.{split}.parquet"


def module_of(feature: str) -> str | None:
    """Mô-đun sinh ra cột `feature`, hoặc None nếu đó không phải cột kịch bản."""
    for name, module in REGISTRY.items():
        if feature in module.FEATURES:
            return name
    return None


def read_split(path: str | Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Đọc `<thư mục>/<nguồn>_<tập>.parquet` kèm các cột kịch bản được yêu cầu.

    Cột nào không có trong parquet chính mà thuộc một mô-đun kịch bản thì được
    nạp từ `scenarios/` (thẳng hàng theo dòng gốc), nên phải gọi TRƯỚC mọi phép
    sắp xếp lại. Cột lạ không thuộc mô-đun nào thì báo lỗi ngay.
    """
    import pyarrow.parquet as pq

    path = Path(path)
    if columns is None:
        return pd.read_parquet(path)
    have = set(pq.read_schema(path).names)
    frame = pd.read_parquet(path, columns=[c for c in columns if c in have])
    return attach_scenarios(frame, path, columns)[columns]


def attach_scenarios(frame: pd.DataFrame, path: str | Path, columns) -> pd.DataFrame:
    """Gắn vào `frame` (vừa đọc từ `path`, CHƯA sắp xếp lại) các cột kịch bản còn thiếu.

    Cột thiếu mà không thuộc mô-đun nào thì báo lỗi; cột đã có thì giữ nguyên.
    """
    path = Path(path)
    extra = [c for c in columns if c not in frame.columns]
    if not extra:
        return frame
    owners = {c: module_of(c) for c in extra}
    unknown = [c for c, m in owners.items() if m is None]
    if unknown:
        raise KeyError(f"{path.name} không có cột {unknown} và chúng không thuộc mô-đun kịch bản nào.")
    split = path.stem.rsplit("_", 1)[-1]
    loaded = load_columns(path.parent, split, sorted(set(owners.values())))
    for column in extra:
        if len(loaded[column]) != len(frame):
            raise ValueError(f"{column}: {len(loaded[column])} dòng, {path.name} có {len(frame)} dòng.")
        frame[column] = loaded[column]
    return frame


def load_columns(processed_dir: str | Path, split: str, modules: list[str]) -> dict[str, np.ndarray]:
    """Cột của các mô-đun được bật, thẳng hàng với parquet chính của `split`."""
    columns: dict[str, np.ndarray] = {}
    for module in modules:
        if not REGISTRY[module].FEATURES:
            continue
        path = part_path(processed_dir, module, split)
        if not path.exists():
            raise FileNotFoundError(f"Thiếu {path}. Chạy scripts/build_scenarios.py trước.")
        frame = pd.read_parquet(path)
        for feature in REGISTRY[module].FEATURES:
            columns[feature] = frame[feature].to_numpy(dtype=np.float32)
    return columns
