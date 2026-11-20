# Hướng dẫn chạy Cổng 1 trên máy mới hoặc máy thuê

Nhóm CMET (AQ2026-111) · 15/11/2026

## 0. Tóm tắt

| Mục | Nội dung |
|---|---|
| Việc cần chạy | **Cổng 1** (docs/lo-trinh-train-dot-2.md): DRRE cấu hình R5 với seed 43 và 44 (seed 42 = R5 đã có), đo P2/P3 cho từng seed, bảng xếp hạng P1/P4, tổng hợp theo seed |
| Lệnh | `bash scripts/run_gate1.sh` (chạy lại đúng lệnh nếu bị ngắt — bước đã xong bỏ qua, train chạy tiếp từ epoch đã lưu) |
| Thời gian (i7-13620H, cắm sạc) | ~3,3 giờ/seed train + ~25 phút/seed P2/P3 + ~20 phút bảng xếp hạng ≈ **8 giờ** cho 2 seed |
| Đầu vào | Mã từ GitHub (nhánh `feat/drre-dot-2`) + tệp `gate1_inputs.tar` (775 MB: dữ liệu `data/processed_aml_v2/` + checkpoint `models/drre_r5_hi_small.pt`) |
| Đầu ra cần gửi lại | `gate1_results.tar` (mục 4) |
| Cấu hình tối thiểu | CPU 8 lõi trở lên, **RAM 32 GB** (16 GB có thể thiếu), ổ trống 10 GB. **Không cần GPU** |

Tệp `gate1_inputs.tar` đã được tạo sẵn trên máy hiện tại ở
`D:\Risk Management - HVTC Competition\finguard\Finguard\transfer\gate1_inputs.tar`, kèm mã
SHA-256 trong `gate1_inputs.sha256`:

```
3250e417c410f97f864ddd61985ccdd6fd49b98b716850572a21b33674ce8793
```

Chép tệp này sang máy mới bằng USB, ổ cứng di động hoặc Google Drive. Dữ liệu là bộ giao
dịch **tổng hợp công khai** IBM AMLWorld, không có dữ liệu khách hàng thật.

---

## 1. Phương án A — một máy Windows khác

### 1.1 Cài phần mềm (một lần)

1. **Git for Windows**: <https://git-scm.com/download/win> — cài mặc định (có sẵn Git Bash).
2. **Python 3.10** (đúng 3.10, 64-bit): <https://www.python.org/downloads/release/python-31011/>
   → "Windows installer (64-bit)". Trong trình cài, tích **"Add python.exe to PATH"**.
3. **Chống ngủ máy** (bắt buộc, chạy qua đêm): Settings → System → Power & battery →
   *Screen and sleep*: "When plugged in, put my device to sleep after" = **Never**. Nếu là
   laptop: Control Panel → Power Options → *Choose what closing the lid does* → "When plugged
   in" = **Do nothing**. Luôn **cắm sạc** (chạy pin chậm ~4 lần).

### 1.2 Lấy mã và tạo môi trường

Mở **Git Bash** (không phải PowerShell), chạy từng dòng:

```bash
mkdir -p /d/FinGuard && cd /d/FinGuard
git clone https://github.com/Chidokato5376/FinGuard-AI-Platform.git finguard
cd finguard && git checkout feat/drre-dot-2 && cd ..
py -3.10 -m venv .venv
cd finguard/Finguard
PYV=../../.venv/Scripts/python.exe
$PYV -m pip install --upgrade pip
$PYV -m pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu
grep -v qiskit-machine-learning requirements.txt > /tmp/req.txt
grep -v qiskit-machine-learning constraints.txt  > /tmp/con.txt
$PYV -m pip install -r /tmp/req.txt -c /tmp/con.txt
$PYV -m pip install --no-deps qiskit-machine-learning==0.8.4
$PYV -m pytest -q -p no:warnings
```

Dòng cuối phải báo **402 passed** (khoảng 1–3 phút). Bố cục thư mục phải đúng như sau, vì
các script tìm `.venv` ở thư mục chứa repo:

```
D:\FinGuard\
├── .venv\
└── finguard\
    └── Finguard\      <- mọi lệnh chạy từ đây
```

### 1.3 Đặt dữ liệu

Chép `gate1_inputs.tar` vào `D:\FinGuard\finguard\Finguard\`, rồi trong Git Bash:

```bash
cd /d/FinGuard/finguard/Finguard
sha256sum gate1_inputs.tar        # phải ra đúng mã ở mục 0
tar -xf gate1_inputs.tar          # tạo data/processed_aml_v2/ và models/drre_r5_hi_small.pt
```

### 1.4 Chạy

```bash
cd /d/FinGuard/finguard/Finguard
bash scripts/run_gate1.sh
```

**Giữ cửa sổ Git Bash mở** suốt quá trình (đóng cửa sổ là dừng). Theo dõi ở một cửa sổ Git
Bash khác:

```bash
cd /d/FinGuard/finguard/Finguard
grep -hE "Val AUPRC|=== " logs/gate1.log logs/r5_s4*_hi_small.log | tail -20
```

Mỗi epoch in một dòng `Val AUPRC = ...` (khoảng 45–50 phút/epoch). Bị ngắt (mất điện, lỡ
tắt): chạy lại **đúng** lệnh `bash scripts/run_gate1.sh` — chỉ mất phần epoch đang dở.

---

## 2. Phương án B — thuê máy chủ (server) Linux

### 2.1 Nên thuê gì

- Bài toán này chạy **CPU**: phần nặng là quét tuần tự từng cửa sổ giao dịch và dựng đồ thị
  bằng numpy, GPU gần như không giúp (và đường GPU chưa được kiểm thử — script mặc định ẩn
  GPU). **Không thuê máy GPU.**
- Cấu hình: **8 vCPU loại dedicated** (không chia sẻ), **32 GB RAM**, ổ 40–50 GB,
  hệ điều hành **Ubuntu 22.04 LTS** (Python 3.10 sẵn, cùng phiên bản với máy nhóm).
- **Tốc độ — nói thật:** 8 vCPU trên cloud ≈ 4 lõi vật lý, nên một lượt train **không nhanh
  hơn** laptop i7-13620H, có khi chậm hơn 1–1,5 lần. Lợi ích của máy thuê là **chạy song
  song** (mỗi máy một seed) và **giải phóng laptop** — không phải tốc độ từng lượt.
- Gợi ý: thuê **2 máy**, máy 1 chạy seed 43, máy 2 chạy seed 44 (mục 2.4) → xong trong
  ~4 giờ thay vì ~8 giờ.

### 2.2 Nhà cung cấp gợi ý

| Nhà cung cấp | Vì sao | Lưu ý |
|---|---|---|
| **Google Cloud** (Compute Engine) | Tài khoản mới được **300 USD dùng thử trong 90 ngày**, không bị tính tiền trong thời gian dùng thử — với nhu cầu này gần như **miễn phí**. Có vùng Singapore (`asia-southeast1`) | Cần thẻ để xác minh. Trong thời gian dùng thử không xin tăng quota được. Hết 90 ngày hoặc hết 300 USD thì máy bị dừng và dữ liệu trên máy có thể bị xóa — tải kết quả về trước. Loại máy: `e2-standard-8` hoặc `n2-standard-8` (8 vCPU, 32 GB) |
| **Hetzner Cloud** | Rẻ, tính tiền theo giờ, có Singapore, giao diện đơn giản | Đã **tăng giá mạnh từ 15/06/2026** (dòng CCX/CPX). Một nguồn thứ cấp ghi CCX33 (8 vCPU dedicated, 32 GB) khoảng 0,22 USD/giờ (kiểm tra 07/2026) — xem giá chính thức trước khi thuê. Có thể yêu cầu xác minh danh tính khi đăng ký |
| **BizFly Cloud** (Việt Nam) | Thanh toán VND, hóa đơn trong nước, hỗ trợ tiếng Việt; có **Spot Instance** tính theo giây, rẻ hơn tới 90% so với máy thường | Máy Spot có thể bị thu hồi giữa chừng — vẫn dùng được vì script chạy tiếp từ epoch đã lưu, nhưng mất phần epoch dở. Xem bảng giá/công cụ tính giá trên trang |
| Vast.ai, RunPod | Chợ thuê GPU giá rẻ | **Không cần** cho bài này (CPU) |

Nguồn: [Google Cloud free trial](https://docs.cloud.google.com/free),
[Hetzner Cloud](https://www.hetzner.com/cloud),
[tin tăng giá Hetzner 06/2026](https://wz-it.com/en/blog/hetzner-price-increase-june-2026-cpx-ccx-alternatives/),
[giá CCX33 (thứ cấp)](https://spendark.com/instances/ccx33/),
[BizFly Cloud Server](https://bizflycloud.vn/en/cloud-server/bang-gia),
[BizFly Spot](https://bizflycloud.vn/en/cloud-server/spot). Giá thay đổi thường xuyên —
luôn xem trang chính thức.

**Ước tính chi phí:** 2 máy × ~4–5 giờ ≈ 10 giờ máy. Với máy 8 vCPU/32 GB (khoảng 0,2–0,45
USD/giờ tùy nhà cung cấp, chưa kể thuế) là vài USD; dùng thử Google Cloud thì 0 đồng.

### 2.3 Từng bước (lần đầu dùng server)

**Bước 1 — Tạo khóa SSH trên Windows** (khóa thay cho mật khẩu, an toàn hơn). Mở
**PowerShell**:

```powershell
ssh-keygen -t ed25519 -C finguard
```

Nhấn Enter ở mọi câu hỏi (có thể đặt passphrase nếu muốn). Xem khóa **công khai** để dán lên
nhà cung cấp:

```powershell
type $env:USERPROFILE\.ssh\id_ed25519.pub
```

Chỉ bao giờ dán tệp **`.pub`**. Tệp không có đuôi `.pub` là khóa bí mật — không gửi cho ai.

**Bước 2 — Tạo máy chủ** trên trang quản trị của nhà cung cấp:

- Image/OS: **Ubuntu 22.04 LTS**; cấu hình như mục 2.1; vùng gần Việt Nam (Singapore).
- SSH key: dán nội dung tệp `.pub` ở bước 1.
  - Google Cloud: Compute Engine → VM instances → Create instance → Machine type
    `e2-standard-8`; Boot disk → Ubuntu 22.04 LTS, 50 GB; Security → *Manage access* →
    *Add manually generated SSH keys* → dán khóa. Tên đăng nhập = chữ cuối của khóa
    (`finguard`).
  - Hetzner: Servers → Add server → chọn Location, Image Ubuntu 22.04, Type dedicated vCPU
    8 lõi/32 GB → SSH keys → dán khóa. Tên đăng nhập = `root`.
- Ghi lại **địa chỉ IP công khai** (Public/External IP).

**Bước 3 — Kết nối** từ PowerShell (lần đầu gõ `yes` để xác nhận máy):

```powershell
ssh finguard@<IP>        # Google Cloud
ssh root@<IP>            # Hetzner
```

Từ giờ mọi lệnh gõ trong cửa sổ này chạy **trên máy chủ**. Gõ `exit` để thoát.

**Bước 4 — Cài môi trường trên máy chủ:**

```bash
sudo apt update && sudo apt install -y git python3.10-venv python3-pip tmux
mkdir -p ~/fg && cd ~/fg
git clone https://github.com/Chidokato5376/FinGuard-AI-Platform.git finguard
cd finguard && git checkout feat/drre-dot-2 && cd ..
python3.10 -m venv .venv
cd finguard/Finguard
PYV=../../.venv/bin/python
$PYV -m pip install --upgrade pip
$PYV -m pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu
grep -v qiskit-machine-learning requirements.txt > /tmp/req.txt
grep -v qiskit-machine-learning constraints.txt  > /tmp/con.txt
$PYV -m pip install -r /tmp/req.txt -c /tmp/con.txt
$PYV -m pip install --no-deps qiskit-machine-learning==0.8.4
$PYV -m pytest -q -p no:warnings          # phải báo 402 passed
```

(Với `root` trên Hetzner thì bỏ chữ `sudo` cũng được.)

**Bước 5 — Đưa dữ liệu lên.** Mở một cửa sổ **PowerShell mới trên laptop** (không phải cửa
sổ SSH):

```powershell
scp "D:\Risk Management - HVTC Competition\finguard\Finguard\transfer\gate1_inputs.tar" finguard@<IP>:~/fg/finguard/Finguard/
```

Rồi quay lại cửa sổ SSH:

```bash
cd ~/fg/finguard/Finguard
sha256sum gate1_inputs.tar        # đúng mã ở mục 0
tar -xf gate1_inputs.tar
```

**Bước 6 — Chạy trong `tmux`** để tắt laptop mà máy chủ vẫn chạy:

```bash
tmux new -s gate1
cd ~/fg/finguard/Finguard
bash scripts/run_gate1.sh
```

Thoát khỏi tmux mà **không** dừng tiến trình: nhấn `Ctrl+B`, thả ra, rồi nhấn `D`. Lúc này có
thể đóng PowerShell, tắt laptop. Lần sau xem lại:

```bash
ssh finguard@<IP>
tmux attach -t gate1
```

Xem nhanh tiến độ mà không vào tmux:

```bash
cd ~/fg/finguard/Finguard && grep -hE "Val AUPRC|=== " logs/gate1.log logs/r5_s4*_hi_small.log | tail -20
```

### 2.4 Chạy song song hai máy

Làm bước 1–5 cho cả hai máy, rồi ở bước 6:

```bash
SEEDS="43" bash scripts/run_gate1.sh     # máy 1
```

```bash
SEEDS="44" bash scripts/run_gate1.sh     # máy 2
```

Mỗi máy tự đo P2/P3 cho seed của nó; phần tổng hợp đủ 3 seed mình làm trên laptop khi nhận
kết quả.

### 2.5 Tải kết quả về và XÓA máy

Trên máy chủ (cửa sổ SSH):

```bash
cd ~/fg/finguard/Finguard
tar -cf ~/gate1_results.tar --exclude='*/scores/*' reports/gate1 reports/runs logs models/drre_r5_s4*_hi_small.pt
```

Trên laptop (PowerShell):

```powershell
scp finguard@<IP>:~/gate1_results.tar "D:\Risk Management - HVTC Competition\finguard\Finguard\transfer\"
```

Sau khi đã có tệp trên laptop: **xóa (Delete/Destroy) máy chủ** trên trang quản trị. Chỉ tắt
(Stop) là **vẫn bị tính tiền ổ đĩa** (Google Cloud) hoặc **vẫn tính tiền cả máy** (Hetzner).

### 2.6 An toàn

- Chỉ đăng nhập bằng khóa SSH; không lưu mật khẩu, token GitHub hay thông tin cá nhân lên máy
  thuê (repo công khai nên `git clone` không cần đăng nhập).
- Không mở thêm cổng mạng nào; dự án không chạy dịch vụ web trên máy thuê.
- Xóa máy ngay khi xong.

---

## 3. Gửi lại kết quả

Trên **máy Windows khác** (phương án A), trong Git Bash:

```bash
cd /d/FinGuard/finguard/Finguard
tar -cf gate1_results.tar --exclude='*/scores/*' reports/gate1 reports/runs logs models/drre_r5_s4*_hi_small.pt
```

Chép `gate1_results.tar` (vài chục MB) vào
`D:\Risk Management - HVTC Competition\finguard\Finguard\transfer\` trên laptop nhóm rồi báo
lại — phần gộp, tổng hợp 3 seed và kết luận Cổng 1 làm tiếp trên laptop.

**Không** `git push` từ máy mới/máy thuê, để tránh lệch nhánh; mọi commit làm từ laptop nhóm.

---

## 4. Sự cố thường gặp

| Hiện tượng | Cách xử lý |
|---|---|
| `Không tìm thấy Python của môi trường ảo (.venv)` | Sai bố cục thư mục (mục 1.2). Hoặc chỉ định thẳng: `PY=/duong/dan/.venv/bin/python bash scripts/run_gate1.sh` |
| `$'\r': command not found` | Tệp `.sh` bị đổi sang CRLF. Chạy: `git config core.autocrlf false && git checkout -- .` rồi chạy lại |
| `Thiếu data/processed_aml_v2` | Chưa giải nén `gate1_inputs.tar` trong `Finguard/` (mục 1.3 / bước 5) |
| `MemoryError` hoặc máy chủ tự khởi động lại | Thiếu RAM — dùng máy 32 GB |
| `pytest` không ra 402 passed | Sai phiên bản thư viện: cài lại đúng các lệnh `pip` ở trên, đúng Python 3.10 |
| Một epoch lâu hơn ~1,5 giờ | Laptop đang chạy pin, hoặc máy thuê loại vCPU *shared* — đổi sang *dedicated* |
| `Val AUPRC` không xuất hiện sau 2 giờ | Xem lỗi: `tail -50 logs/r5_s43_hi_small.log` (thay 43 bằng seed đang chạy) |
