# shellcheck shell=bash
# Nạp (source) bởi mọi script chạy trong scripts/. Đặt:
#   FINGUARD = thư mục Finguard/ (nơi chạy lệnh)
#   ROOT     = thư mục chứa repo (cha của finguard/): nơi để .venv và các CSV gốc AMLWorld
#   PY       = Python của môi trường ảo
# Chạy được cả Windows (Git Bash: .venv/Scripts/python.exe) lẫn Linux/macOS (.venv/bin/python).
# Ghi đè được: PY=/duong/dan/python bash scripts/run_r23.sh r5   hoặc   ROOT=/data bash ...
FINGUARD="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${ROOT:-$(cd "$FINGUARD/../.." && pwd)}"
if [ -z "${PY:-}" ]; then
  for candidate in "$ROOT/.venv/Scripts/python.exe" "$ROOT/.venv/bin/python" \
                   "$FINGUARD/.venv/Scripts/python.exe" "$FINGUARD/.venv/bin/python"; do
    if [ -x "$candidate" ]; then PY="$candidate"; break; fi
  done
fi
if [ -z "${PY:-}" ]; then
  echo "Không tìm thấy Python của môi trường ảo (.venv) ở $ROOT hoặc $FINGUARD." >&2
  echo "Tạo venv theo docs/huong-dan-chay-may-moi.md, hoặc đặt biến PY=<đường dẫn python>." >&2
  exit 1
fi
export PY ROOT FINGUARD
# Mọi số của đợt 2 đo trên CPU. Đường GPU của DRRE chưa được kiểm thử, nên mặc định ẨN GPU
# (máy thuê có GPU vẫn chạy CPU, kết quả so được với các bậc trước). Muốn thử GPU: FINGUARD_GPU=0
export CUDA_VISIBLE_DEVICES="${FINGUARD_GPU:-}"
