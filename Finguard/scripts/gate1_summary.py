"""Tổng hợp Cổng 1 theo seed: trung vị và dải (min–max) của P1 và P2.

Đọc bảng xếp hạng (<gate>/leaderboard/*/leaderboard.json) và các báo cáo
drre_contribution (<gate>/*drre_contribution*.json), ghi <gate>/summary.{md,json}.

    python -X utf8 scripts/gate1_summary.py --gate-dir reports/gate1
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


def _seed_of(text: str) -> str:
    match = re.search(r"_s(\d+)", text)
    return match.group(1) if match else "42"


def _spread(values: list[float]) -> str:
    if not values:
        return "—"
    v = np.asarray(values, dtype=float)
    return f"{np.median(v):.4f} [{v.min():.4f}; {v.max():.4f}]"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gate-dir", default="reports/gate1")
    args = parser.parse_args()
    gate = Path(args.gate_dir)

    p1: dict[str, dict] = {}
    boards = sorted(gate.glob("leaderboard/*/leaderboard.json"))
    gbdt = None
    if boards:
        board = json.loads(boards[-1].read_text("utf-8"))
        for name, row in board["models"].items():
            if name.startswith("R5_s"):
                p1[_seed_of(name)] = {
                    "auprc": row["auprc"], "caught@2000": row["caught@2000"],
                    "vs_leader": board["vs_leader"].get(name),
                    "typology_recall@2000": row["typology_recall@2000"]}
        gbdt = {k: board["models"][k]["auprc"] for k in board["models"] if k.startswith("gbdt")}

    p2: dict[str, dict] = {}
    for path in sorted(gate.glob("*drre_contribution*.json")):
        report = json.loads(path.read_text("utf-8"))
        seed = _seed_of(report["hyperparameters"]["checkpoint"])
        p2[seed] = report["results"]          # bản mới nhất của mỗi seed thắng

    seeds = sorted(set(p1) | set(p2))
    summary = {"seeds": seeds, "p1": p1, "gbdt_auprc": gbdt,
               "p2": {s: {k: p2[s].get(k) for k in ("drre_alone_auprc", "gbdt", "gbdt+diem_drre",
                                                     "gbdt+emb_drre", "gbdt+emb_drre+diem",
                                                     "group_permutation")} for s in p2}}
    lines = ["# Cổng 1 — tổng hợp theo seed (val HI-Small)", "",
             f"Seed: {', '.join(seeds) or '—'}. Mỗi ô: trung vị [min; max] theo seed.", "",
             "## P1 — DRRE R5 so với mô hình cổ điển cùng 28 cột", "",
             f"- AUPRC DRRE: {_spread([p1[s]['auprc'] for s in p1])}",
             f"- Bắt được trong 2.000 cảnh báo: {_spread([p1[s]['caught@2000'] for s in p1])}",
             f"- Δ AUPRC so với hạng 1: {_spread([p1[s]['auprc'] - max(gbdt.values()) for s in p1]) if gbdt else '—'}",
             f"- GBDT (một lần, không phụ thuộc seed khi không dừng sớm): {gbdt}", "",
             "## P2 — GBDT có thêm embedding/điểm DRRE", "",
             "| Biến thể | AUPRC | Δ so với GBDT | Số seed có KTC 95% > 0 |", "|---|---|---|---|"]
    for key in ("gbdt", "gbdt+diem_drre", "gbdt+emb_drre", "gbdt+emb_drre+diem"):
        rows = [p2[s][key] for s in p2 if key in p2[s]]
        if not rows:
            continue
        deltas = [r.get("minus_gbdt") for r in rows if r.get("minus_gbdt") is not None]
        positive = sum(1 for r in rows if r.get("ci95_minus_gbdt", {}).get("lo", -1) > 0)
        lines.append(f"| {key} | {_spread([r['auprc'] for r in rows])} | "
                     f"{_spread(deltas) if deltas else '—'} | "
                     f"{positive}/{len(rows) if key != 'gbdt' else '—'} |")
    lines += ["", "## P4 — recall theo kiểu hình trong 2.000 cảnh báo đầu (trung vị theo seed)", ""]
    kinds = sorted({k for s in p1 for k in p1[s]["typology_recall@2000"]})
    for kind in kinds:
        lines.append(f"- {kind}: {_spread([p1[s]['typology_recall@2000'].get(kind, 0) for s in p1])}")
    (gate / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                       encoding="utf-8")
    (gate / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
