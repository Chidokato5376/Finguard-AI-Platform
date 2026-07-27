# FinGuard AI — Transaction Risk Management Platform

> **Nền tảng Quản trị Rủi ro Giao dịch dựa trên Biểu diễn Rủi ro Động (DRRE), Explainable AI và Tối ưu hóa Lượng tử cho Ngân hàng số**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Qiskit](https://img.shields.io/badge/Qiskit-1.x-6929C4?style=flat-square&logo=qiskit&logoColor=white)](https://qiskit.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20Demo-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://finguard-ai-demo-cfjsgsbg3qxf4zkv7j2xde.streamlit.app/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Research%20Prototype-orange?style=flat-square)]()

> Submission — **AI-Quantum Challenge 2026**
> Tác giả: Phạm Tiến Dũng — Khoa Toán Kinh tế, Đại học Kinh tế Quốc dân (NEU)

**🔗 Live Demo:** [finguard-ai-demo.streamlit.app](https://finguard-ai-demo-cfjsgsbg3qxf4zkv7j2xde.streamlit.app/)

---

## 📂 Nội dung repository

| Đường dẫn | Mô tả |
|---|---|
| [**`Finguard/`**](Finguard/) | **Dự án chính** — kiến trúc DRRE đầy đủ, QAPE/QAOA, nhánh lượng tử, training, bộ test. → [README chi tiết](Finguard/README.md) |
| [**`Finguard-Demo/`**](Finguard-Demo/) | **Bản demo công khai** — dashboard Streamlit chạy ở chế độ Heuristic, kèm sẵn dữ liệu mô phỏng. → [README chi tiết](Finguard-Demo/README.md) |
| `requirements.txt` · `.streamlit/` | Cấu hình triển khai của bản demo (Streamlit Cloud yêu cầu đặt ở gốc repository) |
| `LICENSE` | Giấy phép MIT |

### Khác biệt giữa hai thư mục

| | `Finguard/` (chính) | `Finguard-Demo/` (demo) |
|---|---|---|
| Bộ chấm điểm | Heuristic **+ DRRE** (học sâu) | Heuristic Scorer |
| Dependencies | Đầy đủ: PyTorch, PyTorch Geometric, Qiskit | 8 gói nhẹ (không torch/qiskit) |
| Nhánh lượng tử (QAOA/VQC) | Có, kèm benchmark | Không (quá nặng cho hạ tầng miễn phí) |
| Huấn luyện mô hình | Có (`src/drre/train.py`) | Không |
| Dữ liệu kèm sẵn | Không (tải riêng, xem README) | `synthetic_vn` đã tiền xử lý |

---

## Tổng quan

FinGuard AI giải quyết bốn hạn chế của các hệ thống chống gian lận hiện hành: thiếu khả năng thích nghi với thủ đoạn mới, bỏ sót ngữ cảnh hành vi và quan hệ mạng lưới, thiếu tính minh bạch, và áp một ngưỡng rủi ro chung cho mọi khách hàng.

Dự án định vị lại bài toán: **rủi ro không phải một nhãn tĩnh gán một lần, mà là một trạng thái embedding cập nhật liên tục** theo lịch sử hành vi và quan hệ mạng lưới của từng khách hàng.

### Ba đóng góp kỹ thuật

| # | Thành phần | Vai trò |
|---|---|---|
| 1 | **DRRE** (Dynamic Risk Representation Engine) | Hợp nhất Behavior Memory (GRU + K-slot), Context-aware Attention, Graph Encoder (GraphSAGE/GAT) và Adaptive Fusion thành một Risk Embedding động |
| 2 | **QAPE** (Quantum Alert Prioritization Engine) | Hình thức hóa bài toán phân bổ nguồn lực xử lý cảnh báo thành QUBO, giải bằng QAOA — đặt lượng tử vào đúng lớp bài toán tối ưu tổ hợp |
| 3 | **Explainable AI nội sinh** | Giải thích sinh trực tiếp từ trọng số attention (α) và gate (β) của kiến trúc |

Triết lý xuyên suốt: FinGuard AI là một **AI Risk Officer** hỗ trợ ra quyết định, **không thay thế** con người ở khâu quyết định cuối cùng — yêu cầu bắt buộc về compliance trong ngành ngân hàng.

---

## Bắt đầu nhanh

### Chạy bản demo (nhanh nhất — dữ liệu đã kèm sẵn)

```bash
git clone https://github.com/Chidokato5376/FinGuard-AI-Platform.git
```
```bash
cd FinGuard-AI-Platform
```
```bash
pip install -r requirements.txt
```
```bash
streamlit run Finguard-Demo/src/dashboard/app.py
```

### Chạy dự án chính (đầy đủ DRRE + lượng tử)

```bash
cd FinGuard-AI-Platform/Finguard
```
```bash
pip install -r requirements.txt
```
```bash
python -m src.data.synthetic_generator --output data/raw/synthetic_vn.csv
```
```bash
python -m src.data.preprocessing --source synthetic_vn --raw-path data/raw/synthetic_vn.csv --output data/processed
```
```bash
streamlit run src/dashboard/app.py
```

> 📖 Hướng dẫn đầy đủ — chuẩn bị dữ liệu PaySim / IEEE-CIS, huấn luyện DRRE, benchmark QAPE, và **cảnh báo quan trọng về phiên bản thư viện lượng tử** — xem [`Finguard/README.md`](Finguard/README.md).

---

## Triển khai bản demo (Streamlit Community Cloud)

| Thiết lập | Giá trị |
|---|---|
| Repository | `Chidokato5376/FinGuard-AI-Platform` |
| Branch | `main` |
| **Main file path** | **`Finguard-Demo/src/dashboard/app.py`** |
| Dependencies | `requirements.txt` (ở gốc repository) |
| Secrets / biến môi trường | Không cần |

---

## Ghi chú về dữ liệu

Dashboard demo dùng dữ liệu **tổng hợp** (`synthetic_vn`), hiệu chỉnh theo thống kê công khai của NHNN/Napas — **không phải giao dịch ngân hàng thật, không chứa dữ liệu khách hàng**. Dự án chính dùng thêm PaySim và IEEE-CIS (dữ liệu công khai trên Kaggle, không phân phối lại qua repository này).

Lý do không dùng dữ liệu thật của Việt Nam: theo **Nghị định 13/2023/NĐ-CP** về bảo vệ dữ liệu cá nhân và quy định bảo mật thông tin khách hàng ngành ngân hàng, Việt Nam chưa có bộ dữ liệu giao dịch gian lận công khai ở cấp bản ghi. Chi tiết về nguồn, cách hiệu chỉnh và các giả định: xem [`Finguard/README.md` §4](Finguard/README.md).

---

*Đại học Kinh tế Quốc dân, Hà Nội*
