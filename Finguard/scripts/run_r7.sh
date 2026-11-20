#!/usr/bin/env bash
# Chuỗi đo còn lại của lộ trình đợt 2 trên MỘT seed (42), chạy tuần tự trên máy này:
#   1. Dựng đặc trưng kịch bản cho HI-Medium (chỉ 3 mô-đun đã chốt ở R1)      ~30–60 phút
#   2. Cổng 1 với seed 42 trên HI-Small: P2/P3 + bảng xếp hạng + tổng hợp      ~45 phút
#   3. R7 = cấu hình R5 trên HI-Medium, 4 epoch                                ~16–20 giờ
#   4. Đánh giá R7 trên val HI-Medium: P2/P3 (drre_contribution, cũng để lại tệp làm nóng)
#      rồi bảng xếp hạng P1/P4 cùng 28 cột                                     ~3–4 giờ
# P5 (đường học Small -> Medium) = so R5 (HI-Small) với R7 (HI-Medium) ở bước 4.
#
# MÁY CẮM SẠC. Gọi lại ĐÚNG lệnh này nếu bị ngắt: bước đã xong bỏ qua, R7 chạy tiếp từ
# epoch đã lưu (mỗi epoch Medium ~4–5 giờ: phiên ngắn hơn một epoch thì mất cả epoch dở).
#   bash scripts/run_r7.sh
#   STOP_AFTER=2 bash scripts/run_r7.sh     # dừng R7 sau 2 epoch của phiên này
set -euo pipefail

source "$(dirname "$0")/_env.sh"   # đặt ROOT, PY (Windows lẫn Linux)
SMALL=data/processed_aml_v2
MEDIUM=data/processed_hi_medium
CK=models/drre_r7_hi_medium.pt
OUT=reports/r7
mkdir -p logs "$OUT"
LOG=logs/r7_chain.log
SCEN=mule_chain,cross_bank_fx,cycle

step() {
  local name="$1"; shift
  if [ -f "$OUT/$name.done" ]; then echo "=== Bỏ qua $name (đã xong)" | tee -a "$LOG"; return; fi
  echo "=== $(date '+%F %T') Bắt đầu $name" | tee -a "$LOG"
  "$@" 2>&1 | tee -a "$LOG"
  touch "$OUT/$name.done"
  echo "=== $(date '+%F %T') Xong $name" | tee -a "$LOG"
}

step scenarios_hi_medium "$PY" -X utf8 scripts/build_scenarios.py --keep-awake \
  --processed-dir "$MEDIUM" --raw-csv "$ROOT/HI-Medium_Trans.csv" --scenarios "$SCEN"

step gate1_s42 env SEEDS=42 bash scripts/run_gate1.sh

# --state-cache 2100000 giữ trạng thái cho toàn bộ ~2,08 triệu tài khoản HI-Medium.
step train_r7 "$PY" -X utf8 -m src.drre.train --keep-awake --source amlworld \
  --processed-dir "$MEDIUM" --checkpoint "$CK" \
  --epochs "${EPOCHS:-4}" --batch-size 32768 \
  --carry-state --state-cache 2100000 --seed 42 \
  --arch v2 --scenarios "$SCEN" \
  --graph past --graph-lookback-hours 24 --graph-block 2048 \
  --graph-arch edge --graph-fanout 10 --graph-layers 3 \
  --resume --stop-after "${STOP_AFTER:-4}"
# train.py thoát bình thường cả khi dừng theo STOP_AFTER: chỉ coi là xong khi đủ epoch.
if ! grep -q "Epoch ${EPOCHS:-4} | Val AUPRC" logs/r7_chain.log; then
  rm -f "$OUT/train_r7.done"
  echo "=== R7 chưa đủ ${EPOCHS:-4} epoch — chạy lại đúng lệnh này để train tiếp." | tee -a "$LOG"
  exit 0
fi

# SKIP_CONTRIB=1: hoãn P2/P3 (P2 ở đây rò rỉ, đo đúng bằng p2_stacking.py; P3 tùy chọn).
# Bảng xếp hạng vẫn dùng lại lượt quét train đã lưu ở models/drre_r7_hi_medium.warm_val.pt.
if [ -n "${SKIP_CONTRIB:-}" ]; then
  echo "=== Hoãn contrib_r7 (SKIP_CONTRIB)" | tee -a "$LOG"
else
  step contrib_r7 "$PY" -X utf8 scripts/drre_contribution.py --keep-awake \
    --checkpoint "$CK" --processed-dir "$MEDIUM" --out-dir "$OUT"
fi
step leaderboard_r7 "$PY" -X utf8 scripts/leaderboard.py --keep-awake --processed-dir "$MEDIUM" \
  --drre "R7=$CK" --features-from "$CK" --out-dir "$OUT/leaderboard"
echo "=== $(date '+%F %T') R7 HOÀN TẤT" | tee -a "$LOG"
