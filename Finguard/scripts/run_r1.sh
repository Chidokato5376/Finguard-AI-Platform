#!/usr/bin/env bash
# R1 của lộ trình đợt 2: dựng đặc trưng kịch bản rồi đo từng mô-đun trên HI-Small và LI-Small.
#
# Gọi lại ĐÚNG lệnh này nếu bị ngắt: mô-đun và biến thể đã xong được bỏ qua.
# Chạy từ thư mục Finguard/:
#   bash scripts/run_r1.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PY="$ROOT/.venv/Scripts/python.exe"
mkdir -p logs reports/r1
LOG=logs/r1.log

run_set() {
  local tag="$1" processed="$2" raw="$3"
  echo "=== $(date '+%F %T') Dựng đặc trưng $tag" | tee -a "$LOG"
  "$PY" -X utf8 scripts/build_scenarios.py --keep-awake --processed-dir "$processed" --raw-csv "$raw" 2>&1 | tee -a "$LOG"
  echo "=== $(date '+%F %T') Đo R1 $tag" | tee -a "$LOG"
  "$PY" -X utf8 scripts/r1_scenarios.py --keep-awake --processed-dir "$processed" --tag "$tag" 2>&1 | tee -a "$LOG"
}

run_set hi_small data/processed_aml_v2 "$ROOT/HI-Small_Trans.csv"
run_set li_small data/processed_li_small "$ROOT/LI-Small_Trans.csv"

# Kiểm tra độ vững: đủ 200 cây, không dừng sớm (dừng sớm làm số cây lệch giữa các biến thể).
for tag in hi_small:data/processed_aml_v2 li_small:data/processed_li_small; do
  echo "=== $(date '+%F %T') Đo R1 ${tag%%:*} (200 cây cố định)" | tee -a "$LOG"
  "$PY" -X utf8 scripts/r1_scenarios.py --keep-awake --no-early-stop --processed-dir "${tag#*:}" \
    --tag "${tag%%:*}_fixed200" 2>&1 | tee -a "$LOG"
done

echo "=== $(date '+%F %T') R1 HOÀN TẤT" | tee -a "$LOG"
