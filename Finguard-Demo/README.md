# FinGuard AI — Dashboard Demo

> **Bản demo công khai của nền tảng Quản trị Rủi ro Giao dịch FinGuard AI**

[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20Demo-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://finguard-ai-demo-cfjsgsbg3qxf4zkv7j2xde.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Mode](https://img.shields.io/badge/Scoring-Heuristic-blue?style=flat-square)]()

**🔗 Live Demo:** [finguard-ai-demo.streamlit.app](https://finguard-ai-demo-cfjsgsbg3qxf4zkv7j2xde.streamlit.app/)
**📦 Repository chính (mã nguồn đầy đủ):** [Finguard-AI](https://github.com/Chidokato5376/Finguard-AI)

> Submission — **AI-Quantum Challenge 2026**
> Tác giả: Phạm Tiến Dũng — Khoa Toán Kinh tế, Đại học Kinh tế Quốc dân (NEU)

---

## Table of Contents

1. [Giới thiệu](#1-giới-thiệu)
2. [Bốn khối chức năng của Dashboard](#2-bốn-khối-chức-năng-của-dashboard)
3. [Bộ chấm điểm & Tính minh bạch](#3-bộ-chấm-điểm--tính-minh-bạch)
4. [Dữ liệu](#4-dữ-liệu)
5. [Cấu trúc & Chạy cục bộ](#5-cấu-trúc--chạy-cục-bộ)
6. [Giới hạn của bản demo](#6-giới-hạn-của-bản-demo)
7. [Giấy phép](#7-giấy-phép)

---

## 1. Giới thiệu

Đây là **bản chụp gọn để triển khai công khai** của FinGuard AI — nền tảng quản trị rủi ro giao dịch cho ngân hàng số, dựa trên Biểu diễn Rủi ro Động (DRRE), Explainable AI nội sinh và tối ưu hóa lượng tử cho bài toán phân bổ nguồn lực xử lý cảnh báo (QAPE).

Bản demo được cắt gọn có chủ đích để chạy ổn định trên hạ tầng miễn phí:

| Đặc điểm | Bản demo (repo này) | Repo chính |
|---|---|---|
| Bộ chấm điểm | Heuristic Scorer (z-score) | Heuristic + DRRE (học sâu) |
| Dependencies | 8 gói nhẹ (không torch/qiskit) | Đầy đủ: PyTorch, PyTorch Geometric, Qiskit |
| Dữ liệu kèm sẵn | `synthetic_vn` (đã tiền xử lý) | PaySim + IEEE-CIS + synthetic_vn |
| Nhánh lượng tử (QAOA/VQC) | Không (quá nặng cho tier free) | Đầy đủ, có benchmark |
| Huấn luyện mô hình | Không | Có (`src/drre/train.py`) |

Toàn bộ mã nguồn kiến trúc (DRRE, QAPE, quantum kernel, VQC, training loop, 49 unit test) nằm ở **[repository chính](https://github.com/Chidokato5376/Finguard-AI)**.

---

## 2. Bốn khối chức năng của Dashboard

Dashboard mô phỏng quy trình làm việc thực của một chuyên viên quản trị rủi ro: **giám sát → chấm điểm → hiểu nguyên nhân → nhìn mạng lưới → phân bổ nguồn lực**.

### Khối 1 — Hàng đợi cảnh báo ưu tiên

Bảng giao dịch được chấm điểm rủi ro liên tục **0–100** (tô màu theo mức độ), phân ba tier:

| Khoảng điểm | Tier | Ý nghĩa vận hành |
|---|---|---|
| 0 – 30 | Low Risk | Xử lý bình thường |
| 31 – 70 | Medium Risk | Cần xác thực bổ sung hoặc theo dõi |
| 71 – 100 | Critical | Can thiệp ngay: xác minh, tạm hoãn, chuyển đội xử lý gian lận |

**Ý nghĩa rủi ro:** chấm điểm *liên tục* thay vì nhãn nhị phân cho phép **xếp hạng ưu tiên** thay vì chỉ quyết định chặn/cho qua — rủi ro là một mức độ, không phải trạng thái đúng/sai.

### Khối 2 — Giải thích quyết định (Explainable AI)

Với mỗi giao dịch được chọn, hệ thống chỉ rõ *vì sao* điểm rủi ro cao:

- **Z-score biên độ giao dịch** — độ lệch số tiền so với **lịch sử của chính tài khoản đó** (không dùng ngưỡng chung cho mọi khách hàng)
- **Người nhận lần đầu xuất hiện** — tín hiệu rủi ro cổ điển của lừa đảo đầu tư / chuyển tiền bị dụ
- **Số giao dịch lịch sử của tài khoản** — cơ sở đánh giá độ tin cậy của z-score
- **Công thức đầy đủ** được công bố ngay trên giao diện

**Ý nghĩa rủi ro:** ngân hàng phải *giải trình* được với khách hàng và cơ quan quản lý. Lời giải thích giúp chuyên viên **kiểm chứng** thay vì tin mù vào AI.

### Khối 3 — Sơ đồ mạng lưới tài khoản liên quan

Đồ thị quan hệ quanh tài khoản được chọn (chấm đỏ = tài khoản trung tâm).

**Ý nghĩa rủi ro:** nhiều loại gian lận **không lộ ra khi nhìn từng giao dịch riêng lẻ** — chỉ thấy qua cấu trúc mạng lưới:
- **Money Mule** — tiền chảy qua chuỗi tài khoản trung gian rồi rút ra; mỗi giao dịch nhìn riêng có vẻ bình thường
- **Fraud Ring** — cụm tài khoản hoạt động phối hợp

### Khối 4 — Phân bổ ca trực (QAPE)

Chia cảnh báo thành **"xử lý ngay"** vs **"chờ ca sau"** dưới ràng buộc ngân sách xử lý.

**Ý nghĩa rủi ro:** một đội xử lý rủi ro nhận hàng nghìn cảnh báo nhưng chỉ có hữu hạn nhân sự và thời gian (mỗi cảnh báo tốn ~10–20 phút xác minh). QAPE giải bài toán: *với ngân sách giờ công hữu hạn, chọn tập cảnh báo nào để kiểm soát được nhiều rủi ro nhất*, đồng thời **bắt buộc xử lý mọi cảnh báo có điểm ≥ 90**. Trong repo chính, bài toán này được hình thức hóa thành QUBO và giải bằng QAOA.

> Thử kéo thanh **"Ngân sách xử lý"** trên demo — số cảnh báo được chọn thay đổi theo, minh họa trực quan bài toán tối ưu dưới ràng buộc.

---

## 3. Bộ chấm điểm & Tính minh bạch

Bản demo chạy **Heuristic Scorer** — không phải mô hình học sâu. Đây là lựa chọn có chủ đích, không phải hạn chế kỹ thuật:

**Công thức (công bố đầy đủ):**

```
z = (log1p(amount) − mean_lịch_sử_tài_khoản) / std_lịch_sử_tài_khoản
new_counterparty_bonus = +0.8  nếu người nhận mới VÀ tài khoản đã có ≥ 3 giao dịch lịch sử
risk_score = 100 · σ(z + new_counterparty_bonus − 1.2)
```

Chi tiết: [`src/dashboard/scoring_service.py`](src/dashboard/scoring_service.py).

**Nguyên tắc trung thực trong thiết kế:**

- Điểm rủi ro chỉ tính từ **dữ liệu quá khứ tại thời điểm giao dịch** (rolling expanding window) — không nhìn thấy tương lai.
- Nếu tồn tại `models/drre_checkpoint.pt` đã huấn luyện, dashboard **tự động** chuyển sang DRRE thật; nếu không, dùng Heuristic và **ghi rõ trên giao diện**.
- Hệ thống **không bao giờ** dùng trọng số DRRE khởi tạo ngẫu nhiên để chấm điểm — một điểm số trông "giống thật" nhưng vô nghĩa là rủi ro trình bày sai lệch nghiêm trọng.
- Cột `method` trong bảng cảnh báo **luôn** hiển thị backend đang dùng (`heuristic_fallback` / `drre_trained`).

---

## 4. Dữ liệu

Dashboard demo dùng bộ **`synthetic_vn`** — dữ liệu **tổng hợp (mô phỏng)**, hiệu chỉnh theo thống kê công khai của NHNN/Napas.

| Thuộc tính | Giá trị |
|---|---|
| Quy mô | ~5.273 giao dịch · 500 tài khoản · khung thời gian 90 ngày |
| Chia tập (theo thời gian) | train 3.691 · val 790 · test 792 |
| Tỷ lệ nhãn dương (test) | ~1,39% |
| Kịch bản gian lận | Money Mule (18) · Chiếm quyền tài khoản (17) · Lừa đảo đầu tư (12) |

**Hiệu chỉnh:** số tiền trung bình ~6,1 triệu VND (neo theo Napas Q1/2025, nguồn NHNN), phân phối log-normal; tỷ lệ người trưởng thành có tài khoản 88,9% (NHNN cuối 2025).

> ⚠️ **KHÔNG phải dữ liệu thật.** Đây là dữ liệu mô phỏng, **không phải giao dịch ngân hàng thật** và **không chứa dữ liệu khách hàng**. Hai giả định phải nêu rõ: (1) **tỷ lệ gian lận 1% là tham số giả định** — không có nguồn công khai nào công bố con số này ở mức chi tiết cần cho mô phỏng; (2) `device_id` và `ip_country` là trường **tự sinh** (đã gắn cờ `is_synthetic_field`). Bộ dữ liệu này dùng để *minh họa luồng vận hành* và *kiểm thử concept drift*, **không** dùng làm bằng chứng về thị trường Việt Nam hay làm chỉ số hiệu năng chính.

Lý do dùng dữ liệu mô phỏng: theo **Nghị định 13/2023/NĐ-CP** về bảo vệ dữ liệu cá nhân và quy định bảo mật thông tin khách hàng ngành ngân hàng, Việt Nam chưa có bộ dữ liệu giao dịch gian lận công khai ở cấp bản ghi.

---

## 5. Cấu trúc & Chạy cục bộ

```
finguard-demo/
├── README.md
├── requirements.txt              # 8 gói nhẹ (không torch/qiskit)
├── .streamlit/config.toml        # Theme
├── config/config.yaml            # Ngưỡng rủi ro, tham số QAPE
├── data/processed/               # synthetic_vn (train/val/test .parquet) — commit kèm
└── src/
    ├── dashboard/                # app.py + data/scoring/qape/graph services
    ├── data/                     # schema, loaders, preprocessing, synthetic generator
    ├── qape/                     # QUBO + solvers (Greedy/ILP dùng trong demo)
    ├── scoring/ · explainability/ · evaluation/ · drre/ · quantum/
```

### Chạy trên máy

```bash
git clone https://github.com/Chidokato5376/FinGuard-AI-Platform.git
cd FinGuard-AI-Platform
```
```bash
python -m venv .venv
.venv\Scripts\activate
```
```bash
pip install -r requirements.txt
```
```bash
streamlit run Finguard-Demo/src/dashboard/app.py
```

Dữ liệu `synthetic_vn` đã được commit kèm trong `data/processed/`, nên dashboard hiển thị ngay không cần bước tiền xử lý.

### Triển khai (Streamlit Community Cloud)

| Thiết lập | Giá trị |
|---|---|
| Main file path | **`Finguard-Demo/src/dashboard/app.py`** ← mã nguồn nằm trong thư mục con |
| Python version | 3.10+ |
| Dependencies | `requirements.txt` (ở **gốc** repository) |
| Secrets / biến môi trường | Không cần |

---

## 6. Giới hạn của bản demo

| Giới hạn | Chi tiết |
|---|---|
| **Không có mô hình học sâu** | Chạy Heuristic Scorer; DRRE + nhánh lượng tử nằm ở [repo chính](https://github.com/Chidokato5376/Finguard-AI) |
| **Không có QAOA/Simulated Annealing** | Dashboard chỉ dùng Greedy + ILP (đủ nhanh cho tương tác); QAOA quá chậm cho giao diện realtime |
| **Dữ liệu mô phỏng** | Xem cảnh báo ở §4 |
| **App ngủ khi không dùng** | Tier miễn phí: app tự ngủ sau một thời gian không có truy cập, tự thức dậy (~30 s) khi có người vào |
| **Đồ thị chỉ dựng từ lô hiện tại** | Sơ đồ mạng lưới phản ánh lô giao dịch đang hiển thị, không phải toàn bộ lịch sử |

**Nguyên tắc human-in-the-loop:** FinGuard AI đóng vai trò **AI Risk Officer** — đề xuất và cung cấp căn cứ, **không tự động thực thi** các hành động ảnh hưởng trực tiếp đến khách hàng. Quyết định cuối cùng thuộc về chuyên viên quản trị rủi ro.

---

## 7. Giấy phép

MIT License — xem [LICENSE](LICENSE).

Giấy phép áp dụng cho mã nguồn và tài liệu. Bộ sinh dữ liệu tổng hợp được cung cấp cho mục đích tái lập kết quả dưới cùng giấy phép.

---

*Cập nhật lần cuối: 07/2026 · Đại học Kinh tế Quốc dân, Hà Nội*
