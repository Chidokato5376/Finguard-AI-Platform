#!/usr/bin/env bash
# Tiền xử lý toàn bộ một bộ AMLWorld Medium (HI hoặc LI, ~31 triệu dòng) theo
# đúng giao thức sạch đã dùng cho HI-Small: lọc ngày lành --min-daily-volume
# 100000 (các ngày cuối mà máy sinh rửa tiền "chèn" vào được tách ra
# amlworld_drift.parquet — lỗi dữ liệu số 4) và gắn nhãn kiểu hình từ
# {bộ}_Patterns.txt. Bộ lọc độc lập với nhãn.
#
# Cần >= 32 GB RAM: HI-Medium đo được đỉnh 20,4 GB, chạy 12 phút (31/10).
# Chạy từ thư mục Finguard/:  bash scripts/run_aml_medium.sh HI   (hoặc LI)
set -euo pipefail

KIND="${1:-HI}"
case "$KIND" in HI|LI) ;; *) echo "Dùng: $0 HI|LI" >&2; exit 2 ;; esac
TAG="$(echo "$KIND" | tr '[:upper:]' '[:lower:]')"
source "$(dirname "$0")/_env.sh"   # đặt ROOT, PY (Windows lẫn Linux)
OUT="data/processed_${TAG}_medium"
mkdir -p "$OUT" logs

"$PY" -m src.data.preprocessing --source amlworld \
  --raw-path "$ROOT/${KIND}-Medium_Trans.csv" \
  --patterns-path "$ROOT/${KIND}-Medium_Patterns.txt" \
  --min-daily-volume 100000 \
  --output "$OUT" 2>&1 | tee "logs/preprocess_${TAG}_medium.log"
