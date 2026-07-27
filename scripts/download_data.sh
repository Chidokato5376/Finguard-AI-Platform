#!/usr/bin/env bash
# Tải toàn bộ dữ liệu cần thiết cho FinGuard AI (PaySim + IEEE-CIS).
#
# CHẠY TRÊN MÁY CỦA BẠN — script này KHÔNG chạy trong môi trường chat,
# vì cần token Kaggle thật của bạn và quyền truy cập mạng ra kaggle.com.
#
# Yêu cầu trước khi chạy:
#   1. Đã cài: pip install kaggle
#   2. Đã thiết lập token — chọn MỘT trong hai cách:
#        export KAGGLE_API_TOKEN=KGAT_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
#      hoặc:
#        ~/.kaggle/access_token chứa token (chmod 600)
#   3. Đã Accept Rules tại:
#      https://www.kaggle.com/competitions/ieee-fraud-detection/rules
#
# Cách chạy:
#   chmod +x scripts/download_data.sh
#   ./scripts/download_data.sh

set -euo pipefail

RAW_DIR="data/raw"
mkdir -p "$RAW_DIR"

echo "== Kiểm tra xác thực Kaggle =="
if ! kaggle competitions list >/dev/null 2>&1; then
  echo "LỖI: kaggle CLI chưa xác thực được. Kiểm tra lại KAGGLE_API_TOKEN hoặc ~/.kaggle/access_token." >&2
  exit 1
fi
echo "OK — đã xác thực."

echo ""
echo "== 1/2: Tải PaySim =="
kaggle datasets download -d ealaxi/paysim1 -p "$RAW_DIR" --unzip

# Tên file gốc của PaySim khá dài — đổi tên cho khớp config/config.yaml
PAYSIM_ORIGINAL=$(find "$RAW_DIR" -maxdepth 1 -name "PS_*.csv" | head -n 1)
if [ -n "$PAYSIM_ORIGINAL" ]; then
  mv "$PAYSIM_ORIGINAL" "$RAW_DIR/paysim.csv"
  echo "Đã lưu: $RAW_DIR/paysim.csv"
else
  echo "CẢNH BÁO: không tìm thấy file PS_*.csv sau khi giải nén — kiểm tra thủ công $RAW_DIR" >&2
fi

echo ""
echo "== 2/2: Tải IEEE-CIS Fraud Detection =="
if kaggle competitions download -c ieee-fraud-detection -p "$RAW_DIR" 2>/dev/null; then
  mkdir -p "$RAW_DIR/ieee_cis"
  unzip -o "$RAW_DIR/ieee-fraud-detection.zip" -d "$RAW_DIR/ieee_cis" >/dev/null
  rm "$RAW_DIR/ieee-fraud-detection.zip"
  echo "Đã lưu: $RAW_DIR/ieee_cis/train_transaction.csv, train_identity.csv, ..."
else
  echo "CẢNH BÁO: tải IEEE-CIS thất bại — có thể bạn chưa Accept Rules tại:" >&2
  echo "  https://www.kaggle.com/competitions/ieee-fraud-detection/rules" >&2
  echo "Bỏ qua bước này, PaySim vẫn đã tải xong ở trên." >&2
fi

echo ""
echo "== Hoàn tất =="
echo "Kiểm tra kết quả:"
ls -la "$RAW_DIR"
