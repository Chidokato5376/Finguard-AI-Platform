#!/usr/bin/env bash
# Cổng 1 của lộ trình đợt 2 (docs/lo-trinh-train-dot-2.md, mục 3–4), trên HI-Small:
# cấu hình DRRE chốt sau 2A–2C là R5 (2C không cải thiện, xem
# docs/ra-soat-logic-va-xep-hang-2026-11-14.md mục 6), chạy 3 seed (42, 43, 44):
#   1. train R5 cho từng seed (seed 42 = models/drre_r5_hi_small.pt nếu đã có thì bỏ qua)
#   2. P2/P3 cho từng seed: GBDT ± embedding/điểm DRRE, xáo trộn nhóm (drre_contribution.py)
#   3. P1/P4: bảng xếp hạng cùng quy tắc, mỗi seed một dòng, kèm theo kiểu hình
#   4. Tổng hợp trung vị và dải theo seed -> reports/gate1/summary.md
#
# Ước lượng trên i7-13620H CẮM SẠC: ~3,3 giờ/seed train + ~25 phút/seed P2/P3 + ~20 phút
# bảng xếp hạng. Gọi lại ĐÚNG lệnh này nếu bị ngắt: bước đã xong bỏ qua, train chạy tiếp từ
# epoch đã lưu. Chạy từ thư mục Finguard/:
#   bash scripts/run_gate1.sh
#   SEEDS="43" bash scripts/run_gate1.sh      # chỉ một phần (vd. chia cho hai máy)
set -euo pipefail

source "$(dirname "$0")/_env.sh"   # đặt ROOT, PY (Windows lẫn Linux)
DATA=data/processed_aml_v2
OUT=reports/gate1
mkdir -p logs "$OUT"
LOG=logs/gate1.log
SEEDS="${SEEDS:-42 43 44}"

checkpoint() { if [ "$1" = "42" ]; then echo models/drre_r5_hi_small.pt; else echo "models/drre_r5_s$1_hi_small.pt"; fi; }

step() {
  local name="$1"; shift
  if [ -f "$OUT/$name.done" ]; then echo "=== Bỏ qua $name (đã xong)" | tee -a "$LOG"; return; fi
  echo "=== $(date '+%F %T') Bắt đầu $name" | tee -a "$LOG"
  "$@" 2>&1 | tee -a "$LOG"
  touch "$OUT/$name.done"
  echo "=== $(date '+%F %T') Xong $name" | tee -a "$LOG"
}

[ -f "$DATA/amlworld_train.parquet" ] || { echo "Thiếu $DATA — chép thư mục dữ liệu trước (docs/huong-dan-chay-may-moi.md)." >&2; exit 1; }

for seed in $SEEDS; do
  ck="$(checkpoint "$seed")"
  if [ "$seed" = "42" ] && [ -f "$ck" ] && [ ! -f "$OUT/train_s42.done" ]; then
    echo "=== Seed 42 dùng R5 đã có ($ck)" | tee -a "$LOG"; touch "$OUT/train_s42.done"
  fi
  step "train_s$seed" env SEED="$seed" bash scripts/run_r23.sh r5
done

for seed in $SEEDS; do
  step "contrib_s$seed" "$PY" -X utf8 scripts/drre_contribution.py --keep-awake \
    --checkpoint "$(checkpoint "$seed")" --processed-dir "$DATA" --out-dir "$OUT"
done

args=()
for seed in 42 43 44; do
  [ -f "$(checkpoint "$seed")" ] && args+=(--drre "R5_s$seed=$(checkpoint "$seed")")
done
echo "=== $(date '+%F %T') Bảng xếp hạng Cổng 1" | tee -a "$LOG"
"$PY" -X utf8 scripts/leaderboard.py --keep-awake --processed-dir "$DATA" "${args[@]}" \
  --out-dir "$OUT/leaderboard" 2>&1 | tee -a "$LOG"
"$PY" -X utf8 scripts/gate1_summary.py --gate-dir "$OUT" 2>&1 | tee -a "$LOG"
echo "=== $(date '+%F %T') CỔNG 1 HOÀN TẤT — gửi lại thư mục $OUT và logs/" | tee -a "$LOG"
