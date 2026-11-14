#!/usr/bin/env bash
# R2–R5 của lộ trình đợt 2 (docs/lo-trinh-train-dot-2.md, mục 4), trên HI-Small:
#   R2 = DRRE v1 + 28 đặc trưng (18 gốc + mule_chain, cross_bank_fx, cycle — bộ chốt ở R1)
#   R3 = như R2 nhưng tầng chấm điểm v2 (Giai đoạn 2A, --arch v2)
#   R4 = như R3 nhưng đồ thị chỉ gồm cạnh quá khứ 24 giờ (Giai đoạn 2B1, --graph past)
#   R5 = như R4 nhưng đồ thị hai chiều, đặc trưng cạnh, đọc theo cạnh, 3 lớp, fanout 10
#        (Giai đoạn 2B2–2B5, --graph-arch edge)
# Hai bậc liền nhau chỉ khác đúng một thay đổi, nên hiệu số là đóng góp của thay đổi đó.
#
# Chạy từ thư mục Finguard/, MÁY PHẢI CẮM SẠC (chạy pin chậm ~4 lần):
#   bash scripts/run_r23.sh r2
#   bash scripts/run_r23.sh r3
#   bash scripts/run_r23.sh r4
#   bash scripts/run_r23.sh r5
#   STOP_AFTER=2 bash scripts/run_r23.sh r3     # chia thành phiên ngắn hơn
# Gọi lại ĐÚNG lệnh đó để chạy tiếp từ epoch kế (<checkpoint>.resume.pt).
set -euo pipefail

STEP="${1:-}"
case "$STEP" in
  r2) ARCH=v1; EXTRA=() ;;
  r3) ARCH=v2; EXTRA=() ;;
  r4) ARCH=v2; EXTRA=(--graph past --graph-lookback-hours 24 --graph-block 2048) ;;
  r5) ARCH=v2; EXTRA=(--graph past --graph-lookback-hours 24 --graph-block 2048
                      --graph-arch edge --graph-fanout "${FANOUT:-10}" --graph-layers "${LAYERS:-3}") ;;
  *) echo "Dùng: $0 r2|r3|r4|r5" >&2; exit 2 ;;
esac
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PY="$ROOT/.venv/Scripts/python.exe"
mkdir -p logs

# --state-cache 500000 giữ trạng thái cho toàn bộ ~494 nghìn tài khoản HI-Small.
echo "=== $(date '+%F %T') Bắt đầu $STEP (arch $ARCH)" | tee -a "logs/${STEP}_hi_small.log"
"$PY" -X utf8 -m src.drre.train --keep-awake --source amlworld \
  --processed-dir data/processed_aml_v2 \
  --checkpoint "models/drre_${STEP}_hi_small.pt" \
  --epochs "${EPOCHS:-4}" --batch-size 32768 \
  --carry-state --state-cache 500000 --seed 42 \
  --arch "$ARCH" --scenarios mule_chain,cross_bank_fx,cycle "${EXTRA[@]}" \
  --resume --stop-after "${STOP_AFTER:-4}" 2>&1 | tee -a "logs/${STEP}_hi_small.log"
echo "=== $(date '+%F %T') Kết thúc phiên $STEP" | tee -a "logs/${STEP}_hi_small.log"
