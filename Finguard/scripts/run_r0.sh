#!/usr/bin/env bash
# R0 của lộ trình đợt 2 (docs/lo-trinh-train-dot-2.md): đo lại DRRE v1 đúng cách.
#
# Ước lượng khi cắm sạc: ~3,5 giờ. Gọi lại ĐÚNG lệnh này nếu bị ngắt giữa chừng:
# bước đã có báo cáo thì bỏ qua, phần quét làm nóng chạy tiếp từ mốc đã lưu.
#
# Chạy từ thư mục Finguard/:
#   bash scripts/run_r0.sh
set -euo pipefail

source "$(dirname "$0")/_env.sh"   # đặt ROOT, PY (Windows lẫn Linux)
mkdir -p logs reports/r0
LOG=logs/r0.log

done_marker() { echo "reports/r0/$1.done"; }

step() {
  local name="$1"; shift
  if [ -f "$(done_marker "$name")" ]; then
    echo "=== Bỏ qua $name (đã xong)" | tee -a "$LOG"
    return
  fi
  echo "=== $(date '+%F %T') Bắt đầu $name" | tee -a "$LOG"
  "$@" 2>&1 | tee -a "$LOG"
  touch "$(done_marker "$name")"
  echo "=== $(date '+%F %T') Xong $name" | tee -a "$LOG"
}

# Với mỗi checkpoint, drre_contribution PHẢI chạy trước evaluate: lần quét train
# đó vừa ghi embedding vừa để lại trạng thái làm nóng mà evaluate dùng lại.
#
# HI-Small đi trước (~30 phút) để lộ lỗi sớm, trước phần HI-Medium dài. Checkpoint
# này học trên 14 đặc trưng của data/processed_aml nên đối chứng cũng dùng đúng bộ đó.
contrib() {
  "$PY" -X utf8 scripts/drre_contribution.py --keep-awake --out-dir reports/r0     --checkpoint "$1" --processed-dir "$2"
}
evaluate() {
  "$PY" -X utf8 scripts/evaluate.py --keep-awake --source amlworld --split val     --checkpoint "$1" --processed-dir "$2"     --compare-baseline gbdt_can_bang --by-typology --out-dir reports/r0
}

step contrib_hi_small contrib models/drre_aml_probe.pt data/processed_aml
step eval_hi_small evaluate models/drre_aml_probe.pt data/processed_aml
step contrib_hi_medium contrib models/drre_hi_medium.pt data/processed_hi_medium
step eval_hi_medium evaluate models/drre_hi_medium.pt data/processed_hi_medium

echo "=== $(date '+%F %T') R0 HOÀN TẤT" | tee -a "$LOG"
