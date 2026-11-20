#!/usr/bin/env bash
# Huấn luyện DRRE trên AMLWorld Medium, MỖI LẦN GỌI CHẠY MỘT PHIÊN (mặc định
# 1 epoch ≈ 3,5–4 giờ trên i7-13620H). Gọi lại ĐÚNG lệnh này để chạy phiên
# tiếp theo — lần đầu chưa có <checkpoint>.resume.pt thì tự bắt đầu mới.
# Kết quả giống hệt chạy liền một mạch (tests/test_drre_train.py::
# test_resume_matches_uninterrupted_run).
#
# Chạy từ thư mục Finguard/:
#   bash scripts/train_aml_medium.sh HI          # 1 epoch, mục tiêu 5 epoch
#   EPOCHS=8 bash scripts/train_aml_medium.sh HI # nâng mục tiêu khi val còn lên
#   STOP_AFTER=2 bash scripts/train_aml_medium.sh HI  # phiên dài 2 epoch
#
# --state-cache 2100000 giữ trạng thái Behavior Memory cho TOÀN BỘ 2,08 triệu
# tài khoản HI-Medium (≈5,4 GB cho hai store), giống HI-Small (500k > 494k tài
# khoản) — bộ đệm nhỏ hơn sẽ đẩy tài khoản ra và làm phép so sánh lệch.
set -euo pipefail

KIND="${1:-HI}"
case "$KIND" in HI|LI) ;; *) echo "Dùng: $0 HI|LI" >&2; exit 2 ;; esac
TAG="$(echo "$KIND" | tr '[:upper:]' '[:lower:]')"
source "$(dirname "$0")/_env.sh"   # đặt ROOT, PY (Windows lẫn Linux)
mkdir -p logs

# --keep-awake: chính tiến trình huấn luyện xin Windows không ngủ khi rảnh;
# yêu cầu tự hết khi tiến trình thoát. Gập máy / chọn Sleep vẫn ngủ — khi đó
# chỉ mất epoch đang chạy dở.
"$PY" -m src.drre.train --keep-awake --source amlworld \
  --processed-dir "data/processed_${TAG}_medium" \
  --checkpoint "models/drre_${TAG}_medium.pt" \
  --epochs "${EPOCHS:-5}" --batch-size 32768 \
  --carry-state --state-cache 2100000 --seed 42 \
  --resume --stop-after "${STOP_AFTER:-1}" 2>&1 | tee -a "logs/train_${TAG}_medium.log"
