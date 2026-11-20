"""Tính đặc trưng theo kịch bản trên cả tuyến thời gian và ghi ra từng mô-đun.

Tiến trình chết giữa chừng thì chạy lại đúng lệnh này: mô-đun đã ghi xong được bỏ qua.
Đổi dữ liệu, tỷ giá hoặc VERSION của mô-đun thì mô-đun đó được tính lại.

    python -X utf8 scripts/build_scenarios.py --processed-dir data/processed_aml_v2 \
        --raw-csv ../../HI-Small_Trans.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pandas.api.types import union_categoricals

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from src.data.scenarios import REGISTRY, resolve  # noqa: E402
from src.data.scenarios.fx import to_usd, usd_rates  # noqa: E402
from src.data.scenarios.store import folder_of, part_path  # noqa: E402
from src.data.scenarios.stream import Stream  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

SPLITS = ("train", "val", "test", "drift")
COLUMNS = ["account_id", "counterparty_id", "timestamp", "amount", "channel", "currency"]


def _stamp(path: Path) -> list:
    stat = os.stat(path)
    return [path.name, stat.st_size, stat.st_mtime_ns]


def _write_atomic(table: pa.Table, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, tmp, compression="zstd")
    os.replace(tmp, path)


def _load_rates(folder: Path, raw_csv: Path, train_path: Path, train_end, currencies: list[str]) -> dict:
    path = folder / "fx_rates.json"
    identity = {"raw": _stamp(raw_csv), "train": _stamp(train_path), "train_end": str(train_end)}
    if path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))
        if cached["id"] == identity:
            logger.info("Dùng lại tỷ giá đã tính: %s", path)
            return cached["rates"]
    started = time.monotonic()
    rates = usd_rates(raw_csv, train_end, currencies)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"id": identity, "rates": rates}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    os.replace(tmp, path)
    logger.info("Tỷ giá về USD (giai đoạn train, %.0f giây): %s", time.monotonic() - started, rates)
    return rates


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--raw-csv", required=True, help="{bộ}_Trans.csv gốc, chỉ để dựng tỷ giá")
    parser.add_argument("--source", default="amlworld")
    parser.add_argument("--scenarios", default="all", help="'all', 'none', 'fan_in,cycle', 'all,-hub'")
    parser.add_argument("--keep-awake", action="store_true")
    args = parser.parse_args()

    if args.keep_awake and sys.platform == "win32":
        import ctypes

        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)

    processed = Path(args.processed_dir)
    folder = folder_of(processed)
    folder.mkdir(exist_ok=True)

    paths = {s: processed / f"{args.source}_{s}.parquet" for s in SPLITS}
    paths = {s: p for s, p in paths.items() if p.exists()}
    frames = {s: pd.read_parquet(p, columns=COLUMNS) for s, p in paths.items()}
    sizes = {s: len(f) for s, f in frames.items()}
    logger.info("Số dòng: %s", sizes)

    currencies = sorted({c for f in frames.values() for c in f["currency"].astype(str).unique()})
    rates = _load_rates(folder, Path(args.raw_csv), paths["train"], frames["train"]["timestamp"].max(),
                        currencies)
    rates_hash = hashlib.sha256(json.dumps(rates, sort_keys=True).encode()).hexdigest()[:16]

    ordered = [frames[s] for s in sizes]
    identities = union_categoricals(
        [f["account_id"].astype("category").array for f in ordered]
        + [f["counterparty_id"].astype("category").array for f in ordered])
    total = sum(sizes.values())
    codes = np.asarray(identities.codes, dtype=np.int64)
    channels = union_categoricals([f["channel"].astype("category").array for f in ordered])
    seconds = np.concatenate(
        [pd.to_datetime(f["timestamp"]).dt.as_unit("s").astype("int64").to_numpy() for f in ordered])
    usd = np.concatenate([to_usd(f["amount"].to_numpy(), f["currency"], rates) for f in ordered])
    if not np.isfinite(usd).all():
        raise RuntimeError("Có dòng quy đổi USD không hữu hạn.")
    del frames, ordered

    stream = Stream(codes[:total], codes[total:], seconds, usd, np.asarray(channels.codes, np.int64),
                    list(channels.categories), np.asarray(identities.categories, dtype=object))
    del codes, seconds, usd
    logger.info("Stream: %d dòng, %d thực thể", stream.n, stream.n_entities)

    stamps = [_stamp(p) for p in paths.values()]
    for name in resolve(args.scenarios):
        module = REGISTRY[name]
        meta_path = folder / f"{name}.meta.json"
        identity = {"version": module.VERSION, "data": stamps, "rates": rates_hash,
                    "features": module.FEATURES}
        complete = all(part_path(processed, name, s).exists() for s in sizes) or not module.FEATURES
        if complete and meta_path.exists() and json.loads(meta_path.read_text("utf-8")) == identity:
            logger.info("Bỏ qua %s (đã có)", name)
            continue
        started = time.monotonic()
        values = module.compute(stream)
        if set(values) != set(module.FEATURES):
            raise RuntimeError(f"{name}: cột sinh ra khác FEATURES.")
        for feature, column in values.items():
            if len(column) != stream.n or not np.isfinite(column).all():
                raise RuntimeError(f"{name}.{feature}: sai độ dài hoặc có giá trị không hữu hạn.")

        start = 0
        for split, size in sizes.items():
            table = pa.table({f: pa.array(values[f][start:start + size]) for f in module.FEATURES})
            if module.FEATURES:
                _write_atomic(table, part_path(processed, name, split))
            start += size
        tmp = meta_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(identity), encoding="utf-8")
        os.replace(tmp, meta_path)
        logger.info("Xong %s: %d cột, %.0f giây", name, len(module.FEATURES), time.monotonic() - started)
    logger.info("Hoàn tất dựng đặc trưng kịch bản: %s", folder)
    return 0


if __name__ == "__main__":
    sys.exit(main())
