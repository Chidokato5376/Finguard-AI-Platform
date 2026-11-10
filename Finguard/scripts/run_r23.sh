#!/usr/bin/env bash
# R2 và R3 của lộ trình đợt 2 (docs/lo-trinh-train-dot-2.md, mục 4), trên HI-Small:
#   R2 = DRRE v1 + 28 đặc trưng (18 gốc + mule_chain, cross_bank_fx, cycle — bộ chốt ở R1)
#   R3 = như R2 nhưng tầng chấm điểm v2 (Giai đoạn 2A, --arch v2)
# Hai bậc chỉ khác nhau đúng một thay đổi, nên hiệu số R3 − R2 là đóng góp của 2A.
#
# Chạy từ thư mục Finguard/, MÁY PHẢI CẮM SẠC (chạy pin chậm ~4 lần):
#   bash scripts/run_r23.sh r2
#   bash scripts/run_r23.sh r3
#   STOP_AFTER=2 bash scripts/run_r23.sh r3     # chia thành phiên ngắn hơn
# Gọi lại ĐÚNG lệnh đó để chạy tiếp từ epoch kế (<checkpoint>.resume.pt).
set -euo pipefail

STEP="${1:-}"
case "$STEP" in
  r2) ARCH=v1 ;;
  r3) ARCH=v2 ;;
  *) echo "Dùng: $0 r2|r3" >&2; exit 2 ;;
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
  --arch "$ARCH" --scenarios mule_chain,cross_bank_fx,cycle \
  --resume --stop-after "${STOP_AFTER:-4}" 2>&1 | tee -a "logs/${STEP}_hi_small.log"
echo "=== $(date '+%F %T') Kết thúc phiên $STEP" | tee -a "logs/${STEP}_hi_small.log"
