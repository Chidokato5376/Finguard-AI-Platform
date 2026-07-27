# FinGuard AI — Dynamic Risk Representation & Quantum-Enhanced Transaction Risk Management

> **Nền tảng Quản trị Rủi ro Giao dịch dựa trên Biểu diễn Rủi ro Động, tích hợp Explainable AI và Tối ưu hóa Lượng tử cho Ngân hàng số**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Qiskit](https://img.shields.io/badge/Qiskit-1.x-6929C4?style=flat-square&logo=qiskit&logoColor=white)](https://qiskit.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Research%20Prototype-orange?style=flat-square)]()

> Submission — **AI-Quantum Challenge 2026**
> Tác giả: Phạm Tiến Dũng — Khoa Toán Kinh tế, Đại học Kinh tế Quốc dân (NEU)

**🔗 Live Demo:** [finguard-ai-demo.streamlit.app](https://finguard-ai-demo-cfjsgsbg3qxf4zkv7j2xde.streamlit.app/) · **Repo demo:** [Finguard-AI-Demo](https://github.com/Chidokato5376/Finguard-AI-Demo)

---

## Table of Contents

1. [Tóm tắt](#1-tóm-tắt)
2. [Bối cảnh & Động lực nghiên cứu](#2-bối-cảnh--động-lực-nghiên-cứu)
3. [Câu hỏi nghiên cứu](#3-câu-hỏi-nghiên-cứu)
4. [Dữ liệu](#4-dữ-liệu)
5. [Kiến trúc & Phương pháp](#5-kiến-trúc--phương-pháp)
6. [Cài đặt](#6-cài-đặt)
7. [Cách chạy dự án](#7-cách-chạy-dự-án)
8. [Cấu trúc repository](#8-cấu-trúc-repository)
9. [Kết quả](#9-kết-quả)
10. [Giá trị nghiệp vụ & Tuân thủ](#10-giá-trị-nghiệp-vụ--tuân-thủ)
11. [Rủi ro mô hình & Giới hạn](#11-rủi-ro-mô-hình--giới-hạn)

---

## 1. Tóm tắt

FinGuard AI là một nền tảng quản trị rủi ro giao dịch cho ngân hàng số, giải quyết đồng thời bốn hạn chế của các hệ thống chống gian lận hiện hành: **thiếu khả năng thích nghi** với thủ đoạn mới, **bỏ sót ngữ cảnh hành vi và quan hệ mạng lưới**, **thiếu tính minh bạch** trong quyết định của AI, và **áp một ngưỡng rủi ro chung** cho toàn bộ khách hàng.

Dự án định vị lại bài toán: rủi ro không phải một nhãn tĩnh gán một lần, mà là một **trạng thái embedding được cập nhật liên tục** theo lịch sử hành vi và ngữ cảnh quan hệ mạng lưới của từng khách hàng. Ba đóng góp kỹ thuật:

1. **Dynamic Risk Representation Engine (DRRE)** — kiến trúc học biểu diễn hợp nhất Behavior Memory (GRU + K-slot), Context-aware Attention, Graph Encoder (GraphSAGE/GAT) và Adaptive Fusion (gating network), huấn luyện bằng hàm mất mát kết hợp tự-giám-sát + tương phản + bán-giám-sát.
2. **Quantum Alert Prioritization Engine (QAPE)** — hình thức hóa bài toán *chọn tập cảnh báo ưu tiên xử lý dưới ràng buộc nguồn lực* thành QUBO và giải bằng QAOA, đặt tính toán lượng tử vào đúng lớp bài toán tối ưu tổ hợp thay vì gắn hình thức vào tầng dự đoán.
3. **Explainable AI nội sinh** — lời giải thích sinh trực tiếp từ trọng số attention (α) và gate (β) của kiến trúc, trả lời được câu hỏi bậc cao: *hệ thống đang dựa vào lịch sử hành vi hay quan hệ mạng lưới để ra quyết định?*

Triết lý xuyên suốt: FinGuard AI là một **AI Risk Officer** hỗ trợ ra quyết định, **không thay thế** con người ở khâu quyết định cuối cùng — yêu cầu bắt buộc về compliance trong ngành ngân hàng.

Dự án hiện ở trạng thái **research prototype**: toàn bộ tám module đã được cài đặt và chạy được end-to-end, kèm bộ unit test cho từng thành phần. Kết quả benchmark định lượng đang trong quá trình hoàn thiện — xem [§9](#9-kết-quả).

---

## 2. Bối cảnh & Động lực nghiên cứu

### 2.1. Quy mô vấn đề tại Việt Nam

Thiệt hại do lừa đảo trực tuyến tại Việt Nam năm 2024 **ước tính 18.900 tỷ đồng** (Hiệp hội An ninh mạng quốc gia — NCA, khảo sát trên 59.000 người dùng, 12/2024). Trung bình **cứ 220 người dùng điện thoại thông minh có 1 người là nạn nhân**; **70,72%** người được hỏi từng nhận lời mời đầu tư tài chính trá hình.

Các thủ đoạn phổ biến: giả mạo cán bộ công an/viện kiểm sát, deepfake giả người thân, giả mạo nhân viên ngân hàng, lừa đảo đầu tư, lừa đảo tiền điện tử, **Money Mule** (tài khoản trung gian), chiếm quyền điều khiển Mobile Banking, chuyển tiền xuyên biên giới nhằm gây khó khăn truy vết.

### 2.2. Hai thái cực của hệ thống hiện hành

| Cách tiếp cận | Điểm mạnh | Hạn chế cốt lõi |
|---|---|---|
| **Rule-based** (tập luật cố định) | Minh bạch, dễ kiểm toán | Không thích nghi thủ đoạn mới; tỷ lệ cảnh báo sai cao |
| **AI hộp đen** (ensemble XGBoost/LightGBM/Transformer) | Độ chính xác dự báo cao | Không giải trình được; áp ngưỡng chung cho mọi khách hàng |

Vấn đề ngưỡng chung là then chốt: một giao dịch 50 triệu đồng có thể là bất thường nghiêm trọng với khách hàng thường chuyển dưới 3 triệu, nhưng hoàn toàn bình thường với một doanh nghiệp giao dịch hàng trăm triệu mỗi ngày.

### 2.3. Khoảng trống nghiên cứu

Khảo sát các công bố trong nước (Tạp chí Ngân hàng, Tạp chí Kinh tế – Luật & Ngân hàng HVNH, Tạp chí Quản lý nhà nước) cho thấy:

| Loại khoảng trống | Mô tả |
|---|---|
| **Khoảng trống thực nghiệm** | Các nghiên cứu trong nước về phát hiện gian lận đều dừng ở bài toán **phân loại nhị phân** trên dữ liệu công khai, đo bằng Accuracy/AUC-ROC — chỉ số mất ý nghĩa khi tỷ lệ gian lận dưới 1% |
| **Khoảng trống phương pháp** | Chưa có công bố trong nước áp dụng phân tích đồ thị (GNN), học biểu diễn theo chuỗi hành vi, hay Explainable AI nội sinh cho quản trị rủi ro giao dịch |
| **Khoảng trống lượng tử** | Các công bố về lượng tử–tài chính tại Việt Nam hiện dừng ở mức **tổng quan tiềm năng**; chưa có công bố nào hình thức hóa một bài toán quản trị rủi ro giao dịch cụ thể thành QUBO và giải bằng thuật toán lượng tử |
| **Khoảng trống quy trình** | Phần lớn nghiên cứu dừng ở bước *phát hiện*, chưa xây dựng quy trình đầy đủ: nhận diện → đánh giá → giải thích → cảnh báo → hỗ trợ quyết định → phân bổ nguồn lực xử lý |

---

## 3. Câu hỏi nghiên cứu

**RQ1.** Một kiến trúc học biểu diễn rủi ro động (DRRE) — hợp nhất chuỗi hành vi cá nhân và quan hệ mạng lưới tài khoản — có cải thiện năng lực phân biệt gian lận so với baseline chấm điểm theo độ lệch thống kê và ensemble tree-based, đặc biệt trên phân khúc khách hàng mới (cold-start)?

**RQ2.** Lời giải thích nội sinh từ trọng số attention/gate có cung cấp thông tin nghiệp vụ hữu ích hơn so với diễn giải post-hoc (SHAP) trên đặc trưng tĩnh, và có vượt qua được kiểm định tính trung thực (faithfulness) không?

**RQ3.** Thuật toán tối ưu lượng tử (QAOA) có khả thi cho bài toán phân bổ nguồn lực xử lý cảnh báo ở quy mô vận hành, và tại ngưỡng quy mô nào nó mới có tiềm năng vượt bộ giải cổ điển chính xác (ILP)?

---

## 4. Dữ liệu

> ⚠️ **Ghi chú về dữ liệu Việt Nam:** Do ràng buộc pháp lý theo **Nghị định 13/2023/NĐ-CP** về bảo vệ dữ liệu cá nhân và quy định bảo mật thông tin khách hàng của ngành ngân hàng, **Việt Nam chưa có bộ dữ liệu giao dịch gian lận công khai ở cấp bản ghi**. Hệ thống SIMO của NHNN là hệ thống nội bộ liên ngân hàng, không công khai. Dự án khắc phục bằng dữ liệu công khai quốc tế kết hợp dữ liệu tổng hợp hiệu chỉnh theo thống kê đã công bố, và nêu rõ mọi giới hạn khi diễn giải kết quả.

### 4.1. Nguồn dữ liệu

| Nguồn | Quy mô | Vai trò | Giấy phép |
|---|---|---|---|
| [**PaySim**](https://www.kaggle.com/datasets/ealaxi/paysim1) (Kaggle, `ealaxi`) | ~6,36 triệu giao dịch | Behavior Memory + Graph Encoder (có sender/receiver, trục thời gian `step`) | Public (Kaggle) |
| [**IEEE-CIS Fraud Detection**](https://www.kaggle.com/competitions/ieee-fraud-detection) (Vesta) | ~590.000 giao dịch · 394 + 41 cột | Đa dạng đặc trưng giao dịch, đầu dò bán giám sát (ℒ_BCE), kiểm định chéo | Public (competition rules) |
| **Synthetic VN** (tự sinh) | Tùy cấu hình (mặc định ~5.000 giao dịch) | Kiểm thử concept drift, kịch bản đặc thù VN, dữ liệu demo | Nội bộ (MIT) |

### 4.2. Unified Schema

Ba nguồn được chuẩn hóa về một schema chung trước khi vào pipeline. Định nghĩa chính thức: [`src/data/schema.py`](src/data/schema.py) (dataclass `TransactionRecord`).

| Trường | Kiểu | Mô tả |
|---|---|---|
| `account_id` | string | Định danh tài khoản gửi — khóa tích lũy Behavior Memory |
| `counterparty_id` | string | Định danh tài khoản nhận — đỉnh thứ hai trong đồ thị giao dịch |
| `timestamp` / `step` | datetime / int | Trục thời gian cho GRU và decay hành vi |
| `amount` | float | Số tiền giao dịch |
| `channel` | categorical | nội địa / quốc tế / ví điện tử |
| `device_id` | string, optional | Tín hiệu thiết bị — **PaySim không có gốc**, xem §4.4 |
| `ip_country` | categorical, optional | Tín hiệu định vị — **PaySim/IEEE-CIS không có gốc**, xem §4.4 |
| `label` | binary, optional | Nhãn gian lận — chỉ dùng cho ℒ_BCE, không bắt buộc cho ℒ_pred/ℒ_contrast |
| `source` | enum | Nguồn gốc bản ghi — bắt buộc để audit tính xác thực từng trường |
| `is_synthetic_field` | tuple | Danh sách trường được **sinh bổ sung**, không phải dữ liệu gốc |

### 4.3. Dữ liệu tổng hợp Việt Nam (`synthetic_vn`)

Sinh bởi [`src/data/synthetic_generator.py`](src/data/synthetic_generator.py), hiệu chỉnh theo thống kê công khai:

- **Số tiền trung bình ~6,1 triệu VND** — neo theo Napas Q1/2025 (2.453.426.745 giao dịch, tổng giá trị 14.972.645 tỷ đồng; nguồn: NHNN), phân phối log-normal.
- **Tỷ lệ người trưởng thành có tài khoản 88,9%** — theo NHNN cuối 2025.

Ba kịch bản gian lận, mỗi kịch bản kiểm thử một thành phần khác nhau của DRRE:

| Kịch bản | Đặc điểm | Thành phần được kiểm thử |
|---|---|---|
| **Money Mule** | Tiền chảy qua chuỗi tài khoản trung gian mới, 20–200 triệu, trong vài phút | Graph Encoder + cold-start |
| **Chiếm quyền tài khoản (ATO)** | 5–10 GD nhỏ bình thường, rồi 1 GD lớn gấp 15–40× từ thiết bị lạ + IP nước ngoài | Context-aware Attention (δ_t) |
| **Lừa đảo đầu tư/tình cảm** | Nạn nhân *tự* chuyển 10–50× tới người nhận mới, **cùng thiết bị/IP** | Ca khó nhất — chỉ Graph + độ lệch biên độ |

### 4.4. Giới hạn dữ liệu đã biết

- **Mất cân bằng nhãn nghiêm trọng** (< 1% gian lận): xử lý bằng contrastive learning + self-supervised prediction loss làm tín hiệu chính, nhãn chỉ dùng fine-tune. **Không dùng SMOTE** — kỹ thuật này phá vỡ cấu trúc chuỗi thời gian theo khách hàng.
- **PaySim là dữ liệu mô phỏng**, không phản ánh 100% hành vi người dùng Việt Nam. Ngoài ra PaySim **rất thưa** — phần lớn tài khoản chỉ xuất hiện một lần, làm suy yếu cả tín hiệu chuỗi hành vi lẫn message passing trên đồ thị.
- **PaySim không có `device_id`/`ip_country` gốc** — nếu sinh bổ sung, bắt buộc gắn cờ `is_synthetic_field` và không được trình bày như dữ liệu thật.
- **IEEE-CIS không có `counterparty_id`** → không dùng làm nguồn cho Graph Encoder; `account_id` là **pseudo-ID** suy luận từ tổ hợp `card1/card2/card3/card5/addr1/P_emaildomain` (heuristic phổ biến, có thể gộp/tách nhầm khách hàng). 339 cột `V*` của Vesta **không được công bố ý nghĩa** → không dùng cho diễn giải nghiệp vụ.
- **Tỷ lệ gian lận trong `synthetic_vn` là THAM SỐ GIẢ ĐỊNH** (mặc định 1%) — không có nguồn công khai nào công bố tỷ lệ này ở mức chi tiết cần cho mô phỏng. Bắt buộc khai báo là giả định khi báo cáo.
- **Rủi ro temporal leakage**: bắt buộc time-based split theo *giá trị* thời gian (không theo chỉ số dòng, không chia ngẫu nhiên) — xem [`src/data/preprocessing.py`](src/data/preprocessing.py).

---

## 5. Kiến trúc & Phương pháp

### 5.1. Tám module chức năng

| # | Module | Vai trò | Mã nguồn |
|---|---|---|---|
| 1 | Transaction Monitoring | Thu nhận, chuẩn hóa giao dịch theo Unified Schema | `src/data/` |
| 2 | Behavior Analytics | Cung cấp lịch sử hành vi thô cho Behavior Memory | `src/data/` |
| **3′** | **Dynamic Risk Representation Engine (DRRE)** | Hợp nhất giao dịch + lịch sử hành vi + quan hệ mạng lưới thành Risk Embedding | `src/drre/` |
| 4 | Graph Intelligence | Xây dựng đồ thị quan hệ tài khoản cho Graph Encoder | `src/dashboard/graph_utils.py` |
| 5 | Risk Scoring Engine | Ánh xạ Risk Embedding → điểm rủi ro liên tục 0–100 | `src/scoring/` |
| 6 | Explainable AI | Giải thích qua trọng số attention (α) / gate (β) / cạnh đồ thị | `src/explainability/` |
| 7 | Decision Support | Đề xuất hành động xử lý theo mức rủi ro | `src/dashboard/` |
| 8 | **Quantum Alert Prioritization (QAPE)** | Chọn tập cảnh báo ưu tiên dưới ràng buộc nguồn lực bằng QAOA | `src/qape/` |

### 5.2. Luồng xử lý DRRE (Module 3′)

```
x_t (giao dịch hiện tại)
   │
   ├──▶ Behavior Memory (GRUCell + K-slot decay) ──▶ h_t, M_t
   │                                                    │
   │                              Context-aware Attention (α_k, δ_t)
   │                                                    │
   │                                                    ▼
   │                                             h̃_t (ngữ cảnh hành vi)
   │                                                    │
   Graph (Module 4) ──▶ Graph Encoder (GraphSAGE/GAT) ──▶ g_t
   │                                                    │
   └────────────────────┬───────────────────────────────┘
                        ▼
          Adaptive Fusion — gating network (β ∈ ℝ³)
                        │
                        ▼
              e_t (Risk Embedding)
                    │        │
                    ▼        ▼
        Risk Score (0–100)   Explanation (α, β, δ_t)
                    │
                    ▼
              QAPE (Module 8) — QUBO + QAOA
```

### 5.3. Công thức toán học

**Behavior Memory** (Mục 4.1) — trạng thái hành vi cập nhật tuần tự, kèm K slot đại diện các mẫu hành vi điển hình:

$$h_t^c = \text{GRUCell}(x_t, h_{t-1}^c)$$

$$m_k \leftarrow (1 - \alpha_k)\, m_k + \alpha_k f_{\text{enc}}(x_t), \quad \alpha_k \propto \text{similarity}(x_t, m_k)$$

**Context-aware Attention** (Mục 4.2) — so khớp giao dịch với chính các mẫu hành vi của khách hàng đó, **loại bỏ khái niệm ngưỡng chung**:

$$\alpha_k = \text{softmax}_k\!\left(\frac{q_t \cdot m_k}{\sqrt{d}}\right), \quad q_t = W_q x_t, \quad \tilde{h}_t^c = \sum_k \alpha_k m_k$$

$$\delta_t = 1 - \max_k(\alpha_k) \quad \text{(độ lệch ngữ cảnh — diễn giải được)}$$

**Adaptive Risk Fusion** (Mục 4.4) — trọng số hợp nhất *học được*, giải trực tiếp bài toán cold-start:

$$\beta = \text{softmax}\!\left(W_\beta \cdot [\,x_t \,\|\, \tilde{h}_t^c \,\|\, g_t^c\,]\right) \in \mathbb{R}^3$$

$$e_t^c = \beta_1 \phi_1(x_t) + \beta_2 \phi_2(\tilde{h}_t^c) + \beta_3 \phi_3(g_t^c)$$

**Hàm mất mát kết hợp** (Mục 4.6):

$$\mathcal{L} = \mathcal{L}_{\text{pred}} + \lambda_1 \mathcal{L}_{\text{contrast}} + \lambda_2 \mathcal{L}_{\text{BCE}}$$

> ⚠️ **Ghi chú kỹ thuật quan trọng:** ℒ_pred so sánh dự đoán của encoder với embedding tại t+1. Nếu target được sinh bởi *chính* encoder đang huấn luyện, nghiệm tầm thường là ánh xạ mọi input về một điểm cố định — **representation collapse** (hiện tượng kinh điển, xem BYOL/SimSiam). Implementation này bắt buộc dùng **EMA target network** với `torch.no_grad()`; hàm `prediction_loss()` chủ động raise `ValueError` nếu phát hiện `e_target.requires_grad = True`. Xem [`src/drre/losses.py`](src/drre/losses.py).

**Risk Scoring** (Module 5): $r_t = 100 \cdot \sigma(w \cdot e_t^c + b)$

| Khoảng điểm | Phân loại | Ý nghĩa vận hành |
|---|---|---|
| 0 – 30 | Low Risk | Xử lý bình thường |
| 31 – 70 | Medium Risk | Cần xác thực bổ sung hoặc theo dõi |
| 71 – 100 | Critical | Can thiệp ngay: xác minh, tạm hoãn, chuyển đội xử lý gian lận |

### 5.4. QAPE — Bài toán phân bổ nguồn lực xử lý cảnh báo

**Phát biểu gốc** (Mục 6.1): với $n$ cảnh báo trong một lô, mỗi cảnh báo $i$ có điểm rủi ro $r_i$, chi phí xử lý $c_i$, mức khẩn cấp $u_i$, biến quyết định $x_i \in \{0,1\}$:

$$\max \sum_i (r_i u_i) x_i \quad \text{s.t.} \quad \sum_i c_i x_i \le C, \quad x_i = 1 \;\; \forall i : r_i \ge 90$$

**Chuyển sang QUBO** bằng penalty method, giải bằng QAOA (ansatz $p$ lớp, optimizer COBYLA):

$$H_C = -\sum_i (r_i u_i) x_i + \lambda_1\Big(\sum_i c_i x_i - C\Big)^2 + \lambda_2 \sum_{i : r_i \ge 90} (1 - x_i)$$

> **Ghi chú thiết kế về penalty method:** penalty chỉ ràng buộc "mọi cảnh báo Critical phải được chọn" ở dạng *mềm*, không phải ràng buộc cứng như ILP; hệ số λ₁ ảnh hưởng tới việc ràng buộc có được tôn trọng hay không theo cách **không đơn điệu** — đặc tính điển hình của penalty method. Vì vậy implementation áp dụng bước `_repair_critical_constraint()` **sau** khi giải, đảm bảo tính đúng đắn nghiệp vụ không phụ thuộc vào việc tuning λ₁ có may mắn hay không. Xem [`src/qape/solvers.py`](src/qape/solvers.py).

### 5.5. Nhánh lượng tử tăng cường biểu diễn (tùy chọn)

| Phương án | Cơ chế | Trạng thái |
|---|---|---|
| **Quantum Kernel** (ưu tiên) | `FidelityQuantumKernel` + ZZFeatureMap, thay kernel RBF trong SVM/kNN trên không gian embedding đã nén PCA | Đã cài đặt, **có gate benchmark bắt buộc** |
| **VQC** (nâng cao) | `EstimatorQNN` + ZFeatureMap + RealAmplitudes, huấn luyện đồng thời với phần cổ điển qua `TorchConnector` | Đã cài đặt, tùy chọn |

Cả hai nhánh **chỉ được bật** (`config.yaml: quantum.enabled`) sau khi `benchmark_vs_classical()` xác nhận cải thiện AUPRC thực sự so với baseline cổ điển tương ứng. Đây là ràng buộc thiết kế có chủ đích nhằm tránh gắn lượng tử một cách hình thức.

### 5.6. Chỉ số đánh giá

| Chỉ số | Vai trò | Lý do lựa chọn |
|---|---|---|
| **AUPRC** (Average Precision) | **Chỉ số chính** | Với tỷ lệ gian lận < 1%, Accuracy và AUC-ROC gây ngộ nhận nghiêm trọng — một mô hình dự đoán "không gian lận" cho mọi giao dịch vẫn đạt Accuracy 99% |
| Cold-start AUPRC | Kiểm chứng luận điểm "không dùng ngưỡng chung" | Đo riêng trên phân khúc khách hàng mới |
| Concept-drift Recall drop | Độ bền trước thủ đoạn mới | Huấn luyện trên tập không có pattern X, đánh giá trên tập có X |
| QAOA approximation ratio | Chất lượng nghiệm lượng tử | So với nghiệm ILP tối ưu |
| Constraint violation rate | Tính khả thi nghiệm | Kiểm tra theo ràng buộc **gốc**, không phải bản đã penalize |
| Decision Agreement Rate | Chất lượng giải thích | Tỷ lệ chuyên viên đồng thuận với đề xuất hệ thống |

---

## 6. Cài đặt

### Yêu cầu

- Python 3.10 hoặc cao hơn
- pip 23.0+
- Git

### Bước 1 — Clone repository

```bash
git clone https://github.com/Chidokato5376/Finguard-AI.git
```
```bash
cd Finguard-AI
```

### Bước 2 — Tạo môi trường ảo

```bash
python -m venv .venv
```
```bash
.venv\Scripts\activate
```

### Bước 3 — Cài dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ **BẮT BUỘC — cài toàn bộ bằng MỘT lệnh `-r requirements.txt`.** Cài lẻ từng gói `qiskit-*` trong các lệnh riêng biệt sẽ âm thầm phá vỡ ràng buộc phiên bản đã cài trước đó: pip chỉ resolve ràng buộc của *lệnh đang chạy*, không tự bảo vệ ràng buộc của các gói đã cài. Hậu quả từng gặp trong quá trình phát triển: `qiskit` bị nâng từ 1.4.6 lên 2.5.0 **không kèm bất kỳ lỗi cài đặt nào**, làm QAOA từ chạy ~2 giây tụt xuống không hoàn thành sau 150 giây.
>
> Sau khi cài, kiểm tra bằng `pip list` — kỳ vọng: `numpy 1.26.x` · `torch 2.2.x` · `qiskit 1.4.x` · `qiskit-machine-learning 0.8.x` · `qiskit-algorithms 0.4.0`.

> ⚠️ **Về cảnh báo numpy:** `qiskit-machine-learning 0.8.4` **khai báo** yêu cầu `numpy>=2.0`, trong khi `torch 2.2.2` được biên dịch với `numpy<2` và sẽ báo `RuntimeError: Numpy is not available` nếu gặp numpy 2.x. Cấu hình đã kiểm chứng là **numpy 1.26.4** (module VQC dùng `TorchConnector` cần cả torch và qiskit-machine-learning hoạt động đồng thời). Nếu pip in cảnh báo *"qiskit-machine-learning 0.8.4 requires numpy>=2.0, but you have numpy 1.26.4"* — **bỏ qua**. Nếu `pip list` cho thấy `numpy 2.x`, chạy `pip install "numpy<2"` để khôi phục.

### Bước 4 — Kiểm tra cài đặt

```bash
pytest tests/ -q
```

> Môi trường phát triển tham chiếu: Windows 11, Python 3.10, PyTorch 2.2.2 (CPU), Qiskit 1.4.6. Random seed = 42 xuyên suốt. Trên Windows, đặt biến môi trường `PYTHONUTF8=1` để log tiếng Việt hiển thị đúng:
> ```bash
> set PYTHONUTF8=1
> ```

---

## 7. Cách chạy dự án

Có hai đường đi tùy mục tiêu:

| Mục tiêu | Đường đi | Thời gian |
|---|---|---|
| **Xem demo nhanh** — dashboard hoạt động với dữ liệu mô phỏng | §7.1 → §7.4 | ~2 phút |
| **Chạy đầy đủ** — dữ liệu thật + huấn luyện DRRE + benchmark lượng tử | §7.1 → §7.5 | vài giờ (tùy quy mô) |

### 7.1. Chuẩn bị dữ liệu

Dashboard chỉ hiển thị các nguồn đã có trong `data/processed/`, nên **bước tiền xử lý là bắt buộc** trước khi chạy dashboard.

**Cách A — Dữ liệu tổng hợp (không cần tài khoản Kaggle, khuyên dùng để thử nhanh):**

```bash
python -m src.data.synthetic_generator --output data/raw/synthetic_vn.csv
```
```bash
python -m src.data.preprocessing --source synthetic_vn --raw-path data/raw/synthetic_vn.csv --output data/processed
```

**Cách B — PaySim (dữ liệu mô phỏng quy mô lớn):** tải [PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1) vào `data/raw/`, rồi:

```bash
python -m src.data.preprocessing --source paysim --raw-path data/raw/PS_20174392719_1491204439457_log.csv --output data/processed
```

> 💡 Bộ PaySim đầy đủ có ~6,36 triệu dòng, khiến dashboard tải chậm. Để thử nhanh, tạo một subset trước:
> ```bash
> python -c "import pandas as pd; pd.read_csv('data/raw/PS_20174392719_1491204439457_log.csv', nrows=300000).to_csv('data/raw/paysim_300k.csv', index=False)"
> ```

**Cách C — IEEE-CIS:** cần chấp nhận [Competition Rules](https://www.kaggle.com/competitions/ieee-fraud-detection/rules) trên Kaggle trước khi tải `train_transaction.csv` + `train_identity.csv`:

```bash
python -m src.data.preprocessing --source ieee_cis --raw-path data/raw/train_transaction.csv --identity-path data/raw/train_identity.csv --output data/processed
```

> ⚠️ IEEE-CIS **không có** `counterparty_id` → không dùng được cho Graph Encoder và panel sơ đồ mạng lưới trên dashboard (xem §4.4).

Kết quả bước này: `data/processed/{source}_train.parquet`, `_val.parquet`, `_test.parquet` + file index tài khoản và mapping categorical.

### 7.2. Sinh dữ liệu kiểm thử concept drift (tùy chọn)

Sinh **cặp** dữ liệu known/drift để đo độ bền trước thủ đoạn mới — huấn luyện trên tập *không có* một loại gian lận, rồi đánh giá trên tập *có* loại đó:

```bash
python -m src.data.synthetic_generator --output-dir data/raw --concept-drift-pattern account_takeover
```

Giá trị hợp lệ cho `--concept-drift-pattern`: `money_mule_chain`, `account_takeover`, `investment_romance_scam`.

### 7.3. Huấn luyện DRRE (tùy chọn — dashboard chạy được mà không cần bước này)

```bash
python -m src.drre.train --source synthetic_vn --epochs 15 --batch-size 256 --checkpoint models/drre_synth.pt
```

| Tham số | Ý nghĩa |
|---|---|
| `--source` | `synthetic_vn` \| `paysim` \| `ieee_cis` (phải đã tiền xử lý ở §7.1) |
| `--epochs` | Số vòng huấn luyện |
| `--batch-size` | Kích thước **cửa sổ thời gian** (không phải batch ngẫu nhiên) |
| `--checkpoint` | Đường dẫn lưu checkpoint |
| `--num-negatives` | Số mẫu âm cho ℒ_contrast (mặc định 5) |

Vòng huấn luyện tự lưu checkpoint có **Val AUPRC tốt nhất** và in log mỗi epoch.

> ⚠️ **Ghi chú hiệu năng:** Behavior Memory chạy tuần tự từng giao dịch bằng vòng lặp Python — đúng về mặt toán học nhưng chậm, và **chưa được tối ưu cho quy mô lớn**. Với PaySim, dùng subset (§7.1) thay vì bộ đầy đủ. Nếu log báo *"cửa sổ không có cặp ℒ_pred"* với tỷ lệ cao, tăng `--batch-size`.

### 7.4. Chạy dashboard

```bash
streamlit run src/dashboard/app.py
```

Mở `http://localhost:8501`. Dashboard gồm bốn khối:

| Khối | Nội dung | Ý nghĩa quản trị rủi ro |
|---|---|---|
| **1. Hàng đợi cảnh báo ưu tiên** | Bảng giao dịch chấm điểm 0–100, phân tier Low/Medium/Critical, ngân sách xử lý | Xếp hạng ưu tiên thay vì quyết định chặn/cho qua nhị phân |
| **2. Giải thích quyết định** | Yếu tố đóng góp vào điểm rủi ro (z-score biên độ, người nhận mới, lịch sử tài khoản) hoặc trọng số α/β/δ_t nếu dùng DRRE | Đáp ứng nghĩa vụ giải trình; chuyên viên kiểm chứng thay vì tin mù vào AI |
| **3. Sơ đồ mạng lưới tài khoản** | Đồ thị quan hệ quanh tài khoản được chọn | Phát hiện Money Mule / Fraud Ring — mẫu hình không lộ ra khi nhìn giao dịch riêng lẻ |
| **4. Phân bổ ca trực (QAPE)** | Chia cảnh báo "xử lý ngay" vs "chờ ca sau" theo ngân sách | Bài toán vận hành thật: nguồn lực hữu hạn, phải chọn |

**Chọn bộ chấm điểm:**

| Mục tiêu | Thao tác |
|---|---|
| **Heuristic Scorer** — minh bạch, không cần huấn luyện | Đảm bảo **không** tồn tại `models/drre_checkpoint.pt` |
| **DRRE đã huấn luyện** | Copy checkpoint mong muốn thành `models/drre_checkpoint.pt`, rồi refresh dashboard |

```bash
copy models\drre_synth.pt models\drre_checkpoint.pt
```

Cột `method` trong bảng cảnh báo **luôn** hiển thị backend đang dùng (`heuristic_fallback` / `drre_trained`) — thiết kế có chủ đích để không bao giờ trình bày sai nguồn gốc điểm số. Hệ thống **không bao giờ** dùng trọng số DRRE khởi tạo ngẫu nhiên để chấm điểm.

> 💡 Nên chọn nguồn dữ liệu và checkpoint **khớp nhau** (huấn luyện trên `synthetic_vn` thì xem `synthetic_vn`) — áp model của một miền dữ liệu lên miền khác cho kết quả không có ý nghĩa.

### 7.5. Benchmark QAPE (Greedy vs ILP vs Simulated Annealing vs QAOA)

**Trên dữ liệu cảnh báo tổng hợp** — kiểm tra logic solver độc lập với chất lượng mô hình:

```bash
python -m src.qape.benchmark --n-alerts 15 --budget 50
```

**Trên dữ liệu thật đã tiền xử lý** — dùng checkpoint DRRE nếu có, tự rơi về Heuristic nếu chưa:

```bash
python -m src.qape.benchmark --source synthetic_vn --split test --top-n 15 --budget 50
```

**Bỏ QAOA để chạy nhanh** (chỉ Greedy/ILP/SA):

```bash
python -m src.qape.benchmark --n-alerts 30 --budget 50 --no-qaoa
```

| Tham số | Ý nghĩa |
|---|---|
| `--n-alerts` | Số cảnh báo tổng hợp (mặc định 15, để QAOA khả thi trong giới hạn qubit) |
| `--budget` | Ngân sách xử lý C |
| `--cost-scale` | Hệ số quy đổi chi phí về số nguyên (ảnh hưởng mạnh tới số qubit slack) |
| `--no-qaoa` | Bỏ qua QAOA |

> ⚠️ **Giới hạn quy mô QAOA:** mô phỏng statevector tốn bộ nhớ theo cấp số nhân ($2^n$). `QAOA_MAX_QUBITS_DEFAULT` được đặt theo đo đạc thực tế để tránh treo máy; vượt ngưỡng, hàm `solve_qaoa()` raise lỗi rõ ràng thay vì chạy vô định. Số qubit thực tế **lớn hơn** số cảnh báo do biến slack sinh từ ràng buộc ngân sách (~log₂ của budget × cost_scale).

### 7.6. Chạy toàn bộ test

```bash
pytest tests/ -q
```

Bộ test bao phủ: schema & audit nguồn trường dữ liệu, synthetic generator, IEEE-CIS loader, các hàm mất mát (gồm kiểm tra chống representation collapse), training loop, QUBO/solvers cổ điển, solvers lượng tử, benchmark trên dữ liệu thật, scoring service, và VQC.

---

## 8. Cấu trúc repository

```
finguard/
│
├── README.md                        # File này
├── LICENSE                          # MIT License
├── requirements.txt                 # Dependencies (⚠ khối qiskit* ghim có chủ đích)
├── pyproject.toml                   # Cấu hình package & tooling
│
├── config/
│   └── config.yaml                  # Hyperparameters, ngưỡng rủi ro, tham số QAPE/quantum
│
├── data/
│   ├── raw/                         # Dữ liệu gốc (KHÔNG commit — xem .gitignore)
│   ├── processed/                   # Parquet đã chuẩn hóa + time-based split
│   └── README.md                    # Data dictionary, nguồn, giới hạn
│
├── docs/
│   └── architecture.md              # Đặc tả kỹ thuật đầy đủ (8 module, công thức)
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_behavior_memory_training.ipynb
│   ├── 03_graph_encoder.ipynb
│   ├── 04_drre_end_to_end.ipynb
│   └── 05_qape_benchmark.ipynb
│
├── src/
│   ├── data/
│   │   ├── schema.py                # Unified Schema (TransactionRecord + audit nguồn trường)
│   │   ├── loaders.py               # load_paysim / load_ieee_cis / load_synthetic_vn
│   │   ├── preprocessing.py         # Chuẩn hóa, encode, TIME-BASED SPLIT (chống leakage)
│   │   └── synthetic_generator.py   # Sinh dữ liệu VN + cặp known/drift cho concept drift
│   ├── drre/
│   │   ├── behavior_memory.py       # GRUCell + K-slot memory với decay
│   │   ├── attention.py             # Context-aware Attention (α_k, δ_t)
│   │   ├── graph_encoder.py         # GraphSAGE / GAT wrapper (PyTorch Geometric)
│   │   ├── fusion.py                # Adaptive Risk Fusion (gating network β)
│   │   ├── losses.py                # ℒ_pred (EMA target) + ℒ_contrast + ℒ_BCE
│   │   ├── model.py                 # Lắp ráp DRRE (Module 3′)
│   │   └── train.py                 # Training loop theo cửa sổ thời gian
│   ├── scoring/
│   │   └── risk_scoring.py          # Module 5 — điểm 0–100 + phân tier
│   ├── explainability/
│   │   └── explain.py               # Module 6 — giải thích từ α/β
│   ├── qape/
│   │   ├── qubo.py                  # Alert + QuadraticProgram + penalized QUBO
│   │   ├── solvers.py               # Greedy | ILP | Simulated Annealing | QAOA
│   │   └── benchmark.py             # Benchmark 4 phương pháp + sweep λ₁
│   ├── quantum/
│   │   ├── quantum_kernel.py        # FidelityQuantumKernel + benchmark_vs_classical
│   │   └── vqc.py                   # EstimatorQNN + TorchConnector (tùy chọn)
│   ├── evaluation/
│   │   └── metrics.py               # AUPRC, cold-start, concept drift, approx ratio
│   └── dashboard/
│       ├── app.py                   # Streamlit UI (4 khối)
│       ├── data_service.py          # Đọc parquet, dựng đồ thị hiển thị
│       ├── scoring_service.py       # Heuristic / DRRE (tự chọn, minh bạch backend)
│       ├── qape_service.py          # Nối scoring → Alert → Greedy/ILP
│       └── graph_utils.py           # Dựng edge_index cho Graph Encoder
│
└── tests/                           # Unit test cho từng module
    ├── test_data_schema.py
    ├── test_synthetic_generator.py
    ├── test_ieee_cis_loader.py
    ├── test_losses.py
    ├── test_drre_train.py
    ├── test_qape.py
    ├── test_qape_quantum_solvers.py
    ├── test_benchmark_real_data.py
    ├── test_scoring_service.py
    └── test_vqc.py
```

---

## 9. Kết quả

*Đang hoàn thiện.* Bộ benchmark định lượng sẽ được cập nhật sau khi hoàn tất các hạng mục ưu tiên trong lộ trình phát triển — đặc biệt là mở rộng bộ đặc trưng đầu vào cho DRRE và bổ sung baseline đối chứng mạnh.

Danh sách KPI dự kiến công bố (xem `docs/architecture.md` §11 và [§5.6](#56-chỉ-số-đánh-giá) của file này):

- **AUPRC** trên tập kiểm định — chỉ số chính, so sánh DRRE với baseline heuristic và ensemble tree-based
- **AUPRC trên phân khúc cold-start** — kiểm chứng luận điểm "không dùng ngưỡng chung"
- **Mức suy giảm Recall khi có concept drift** — độ bền trước thủ đoạn mới
- **QAOA approximation ratio** và **tỷ lệ vi phạm ràng buộc** — so với nghiệm ILP tối ưu
- **Decision Agreement Rate** — tỷ lệ chuyên viên đồng thuận với đề xuất của hệ thống

> **Nguyên tắc báo cáo:** khi công bố, kết quả sẽ được trình bày đầy đủ kể cả khi không thuận lợi. Điều 11 Thể lệ cuộc thi ưu tiên kết quả thực hơn độ phức tạp kiến trúc — một mô hình phức tạp chưa vượt được baseline đơn giản là một phát hiện cần công bố kèm chẩn đoán nguyên nhân, không phải điều cần che giấu.

---

## 10. Giá trị nghiệp vụ & Tuân thủ

### 10.1. Giá trị nghiệp vụ

| Thiết kế | Tác động vận hành |
|---|---|
| **Chấm điểm liên tục thay vì nhãn nhị phân** | Cho phép xếp hạng ưu tiên và định giá rủi ro theo mức độ, thay vì quyết định chặn/cho qua |
| **Đánh giá theo ngữ cảnh cá nhân** | Giảm cảnh báo sai với khách hàng có biên độ giao dịch lớn; tăng độ nhạy với khách hàng có hành vi ổn định |
| **Khai thác quan hệ mạng lưới** | Phát hiện gian lận có tổ chức (Money Mule, Fraud Ring) mà phân tích từng giao dịch đơn lẻ không thể hiện được |
| **Phân bổ nguồn lực có ràng buộc (QAPE)** | Trả lời câu hỏi vận hành thực: *với ngân sách giờ công của một ca trực, nên xử lý tập cảnh báo nào?* |
| **Giải thích minh bạch** | Hỗ trợ nghĩa vụ giải trình với khách hàng và cơ quan quản lý; cho phép chuyên viên kiểm chứng thay vì tin mù vào AI |
| **Kiến trúc mở, tách module** | Tích hợp Core Banking qua API chuẩn hóa; thay thế từng mô hình mà không sửa toàn hệ thống |

### 10.2. Đối tượng ứng dụng

| Đối tượng | Giá trị mang lại |
|---|---|
| Ngân hàng thương mại | Giám sát giao dịch và cảnh báo gian lận theo thời gian thực, giảm tải cho đội xử lý rủi ro |
| Ví điện tử & FinTech | Phát hiện giao dịch bất thường trên nền tảng thanh toán số |
| Đơn vị trung gian thanh toán | Chấm điểm rủi ro trước khi xử lý giao dịch |
| Cơ quan quản lý | Nghiên cứu xu hướng gian lận, hỗ trợ hệ thống cảnh báo sớm ở cấp hệ thống |
| Môi trường đào tạo | Nền tảng thực hành về AI, quản trị rủi ro và phân tích dữ liệu tài chính |

### 10.3. Tuân thủ pháp lý

- **Nghị định 13/2023/NĐ-CP** về bảo vệ dữ liệu cá nhân — lý do dự án không sử dụng dữ liệu giao dịch thật của khách hàng Việt Nam.
- **Nguyên tắc human-in-the-loop** — hệ thống *đề xuất*, không tự động thực thi các hành động ảnh hưởng trực tiếp đến khách hàng (tạm hoãn, từ chối giao dịch).
- **Yêu cầu giải trình** — mọi cảnh báo đi kèm lời giải thích có thể kiểm chứng; dashboard luôn công bố backend chấm điểm đang dùng.

---

## 11. Rủi ro mô hình & Giới hạn

| Loại rủi ro | Mô tả | Biện pháp giảm thiểu |
|---|---|---|
| **Đại diện dữ liệu** | PaySim là dữ liệu mô phỏng; `synthetic_vn` có tham số giả định | Nêu rõ mọi giả định; không dùng làm bằng chứng về thị trường Việt Nam |
| **Độ thưa dữ liệu** | PaySim có mật độ giao dịch/tài khoản rất thấp, làm suy yếu tín hiệu chuỗi hành vi và message passing | Ưu tiên nguồn có mật độ cao hơn cho Behavior Memory; nêu rõ khi diễn giải |
| **Representation collapse** | ℒ_pred có nghiệm tầm thường nếu thiếu stop-gradient | EMA target network + kiểm tra chủ động `requires_grad` trong `prediction_loss()` |
| **Temporal leakage** | Rất dễ mắc khi chia tập cho bài toán chuỗi | Time-based split theo *giá trị* thời gian, không theo chỉ số dòng, không chia ngẫu nhiên |
| **Faithfulness của attention** | Trọng số attention không đảm bảo phản ánh đúng cơ chế quyết định (Jain & Wallace, 2019) | Định vị giải thích attention/gate là **bổ sung** cho SHAP, không thay thế; cần kiểm định faithfulness trước khi tuyên bố |
| **Heterophily của đồ thị gian lận** | Kẻ gian lận cố ý kết nối tài khoản bình thường để nguỵ trang; message passing trung bình hóa làm loãng tín hiệu | Cần cơ chế chống camouflage (neighbor sampling theo similarity); baseline XGBoost trên đặc trưng đồ thị thủ công là đối chứng bắt buộc |
| **Penalty method cho ràng buộc cứng** | λ₁ ảnh hưởng không đơn điệu tới việc ràng buộc Critical có được tôn trọng | Bước `_repair_critical_constraint()` sau khi giải; hướng dài hạn là constraint-preserving QAOA |
| **Xung đột phiên bản quantum stack** | Nâng cấp `qiskit` ngoài ý muốn phá vỡ QAOA **không kèm cảnh báo** | Ghim phiên bản chặt trong `requirements.txt`; quy tắc cài một lệnh duy nhất (§6) |
| **Hiệu năng huấn luyện** | Vòng lặp Python tuần tự chưa tối ưu cho dữ liệu quy mô lớn | Dùng subset; vector hóa theo account là hướng cải tiến |
| **Chưa có SLA latency** | Tuyên bố "giám sát thời gian thực" chưa được đo | Cần công bố p95 inference time trước khi đưa claim này vào bản pitch |
| **Quantum ở quy mô demo** | Simulator giới hạn số qubit; chưa có bằng chứng quantum advantage | Định vị rõ là **proof of concept** về khả năng tích hợp, không phải tuyên bố vượt trội |

---

*Cập nhật lần cuối: 07/2026 · Đại học Kinh tế Quốc dân, Hà Nội*
