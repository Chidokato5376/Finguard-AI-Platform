#!/usr/bin/env bash
# R6 của lộ trình đợt 2 — Giai đoạn 2C (cách huấn luyện), trên HI-Small:
#   R6a = bộ mã hóa R5 (đóng băng) + đầu chấm điểm học lại trên mọi vụ  (pha 2, ~20 phút)
#   R6b = R5 + số hạng chống sụp embedding khi huấn luyện                (pha 1, ~3 giờ)
#   R6  = bộ mã hóa R6b (đóng băng) + đầu chấm điểm học lại               (pha 1 + 2)
# Đầu chấm điểm chọn trên 15% CUỐI train (tập giữ lại), không nhìn val. Sau mỗi checkpoint
# mới, bảng xếp hạng được cập nhật (chỉ chấm thêm mô hình mới).
#
# Chạy từ Finguard/, MÁY CẮM SẠC. Gọi lại đúng lệnh nếu bị ngắt: bước đã xong bỏ qua,
# R6b chạy tiếp từ epoch đã lưu.
#   bash scripts/run_r6.sh
set -euo pipefail

source "$(dirname "$0")/_env.sh"   # đặt ROOT, PY (Windows lẫn Linux)
DATA=data/processed_aml_v2
mkdir -p logs reports/r6
LOG=logs/r6.log

leaderboard() {
  "$PY" -X utf8 scripts/leaderboard.py --keep-awake --processed-dir "$DATA" \
    --drre R2=models/drre_r2_hi_small.pt --drre R3=models/drre_r3_hi_small.pt \
    --drre R4=models/drre_r4_hi_small.pt --drre R5=models/drre_r5_hi_small.pt \
    --drre R6a=models/drre_r6a_hi_small.pt --drre R6b=models/drre_r6b_hi_small.pt \
    --drre R6=models/drre_r6_hi_small.pt --features-from models/drre_r3_hi_small.pt
}
refit() {  # $1 = checkpoint nguồn, $2 = checkpoint đích
  [ -f "$2" ] && { echo "=== Bỏ qua $2 (đã có)" | tee -a "$LOG"; return; }
  "$PY" -X utf8 scripts/refit_head.py --keep-awake --checkpoint "$1" --processed-dir "$DATA" \
    --out-checkpoint "$2" --report "reports/r6/$(basename "$2" .pt)_head_refit.json" 2>&1 | tee -a "$LOG"
}

echo "=== $(date '+%F %T') R6a: học lại đầu chấm điểm trên bộ mã hóa R5" | tee -a "$LOG"
refit models/drre_r5_hi_small.pt models/drre_r6a_hi_small.pt
leaderboard 2>&1 | tee -a "$LOG"

echo "=== $(date '+%F %T') R6b: huấn luyện lại có chống sụp embedding" | tee -a "$LOG"
bash scripts/run_r23.sh r6b
[ -f models/drre_r6b_hi_small.pt ] || { echo "R6b chưa xong (chạy tiếp bằng đúng lệnh này)"; exit 0; }

echo "=== $(date '+%F %T') R6: học lại đầu chấm điểm trên bộ mã hóa R6b" | tee -a "$LOG"
refit models/drre_r6b_hi_small.pt models/drre_r6_hi_small.pt
"$PY" -X utf8 scripts/embedding_health.py --checkpoint models/drre_r6b_hi_small.pt \
  --processed-dir "$DATA" --out reports/r6/embedding_r6b.json 2>&1 | tee -a "$LOG"
leaderboard 2>&1 | tee -a "$LOG"
echo "=== $(date '+%F %T') R6 HOÀN TẤT" | tee -a "$LOG"
