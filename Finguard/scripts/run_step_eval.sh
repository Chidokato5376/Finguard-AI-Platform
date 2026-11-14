#!/usr/bin/env bash
# Đánh giá một bậc của thang chạy thử so với bậc trước, trên val HI-Small
# (lộ trình đợt 2, mục 4):
#   1. evaluate: bậc mới (chính) vs bậc trước (--compare-checkpoint) — KTC bootstrap
#      ghép cặp cho hiệu số AUPRC, bắt@k, ma trận nhầm lẫn, theo kênh, theo kiểu hình.
#      Baseline cổ điển ở đây chỉ có 20 đặc trưng gốc; đối chứng cùng 28 đặc
#      trưng là R1 "+core" (reports/r1/hi_small*/results.json) và Cổng 1.
#   2. embedding_health cho cả hai: embedding có bị sụp không (chẩn đoán 2C).
#
# Chạy từ Finguard/, MÁY CẮM SẠC. Gọi lại đúng lệnh nếu bị ngắt: bước đã xong bỏ qua.
#   bash scripts/run_step_eval.sh            # R3 vs R2 (mặc định)
#   bash scripts/run_step_eval.sh r4 r3      # R4 vs R3 -> reports/r4
set -euo pipefail

NEW="${1:-r3}"
OLD="${2:-r2}"
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PY="$ROOT/.venv/Scripts/python.exe"
OUT="reports/$NEW"
mkdir -p logs "$OUT"
LOG="logs/${NEW}_eval.log"
DATA=data/processed_aml_v2
CK_NEW="models/drre_${NEW}_hi_small.pt"
CK_OLD="models/drre_${OLD}_hi_small.pt"

step() {
  local name="$1"; shift
  if [ -f "$OUT/$name.done" ]; then
    echo "=== Bỏ qua $name (đã xong)" | tee -a "$LOG"; return
  fi
  echo "=== $(date '+%F %T') Bắt đầu $name" | tee -a "$LOG"
  "$@" 2>&1 | tee -a "$LOG"
  touch "$OUT/$name.done"
  echo "=== $(date '+%F %T') Xong $name" | tee -a "$LOG"
}

step "eval_${NEW}_vs_${OLD}" "$PY" -X utf8 scripts/evaluate.py --keep-awake --source amlworld \
  --split val --checkpoint "$CK_NEW" --compare-checkpoint "$CK_OLD" \
  --processed-dir "$DATA" --compare-baseline gbdt_can_bang --by-typology --out-dir "$OUT"
step "emb_${NEW}" "$PY" -X utf8 scripts/embedding_health.py --checkpoint "$CK_NEW" \
  --processed-dir "$DATA" --out "$OUT/embedding_${NEW}.json"
echo "=== $(date '+%F %T') Đánh giá $NEW HOÀN TẤT" | tee -a "$LOG"
