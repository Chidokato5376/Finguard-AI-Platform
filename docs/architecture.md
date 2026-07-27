# Đặc tả kỹ thuật — FinGuard AI

Tài liệu này là bản rút gọn kỹ thuật của đặc tả đầy đủ (submission AI-Quantum Challenge 2026), dùng làm tài liệu tham chiếu khi đọc code trong `src/`. Mỗi mục ánh xạ trực tiếp sang một module trong repo.

## Sơ đồ luồng dữ liệu (DRRE)

```
x_t (Module 1) + M_t^c (Module 2)
        │
        ▼
Behavior Memory (GRU)  ──►  Context-aware Attention  ──┐
        │                                                │
Graph (Module 4) ──► Graph-based Relationship Encoder ──┤──► Adaptive Risk Fusion ──► e_t^c
        │                                                │            │
        └──────────── (optional) Quantum-enhanced Embedding ──────────┘
                                                                        │
                                                     ┌──────────────────┴──────────────────┐
                                                     ▼                                      ▼
                                      Module 5 — Risk Scoring Engine          Module 6 — Explainable AI
                                                     │                                      │
                                                     └──────────────► Module 7 — Decision Support
                                                                                             │
                                                                        Module 8 — QAPE (batch, 15-min window)
```

## Module 3′ — Dynamic Risk Representation Engine

**Behavior Memory** (`src/drre/behavior_memory.py`)

```
h_t^c = GRUCell(x_t, h_{t-1}^c)
M_t^c = [m_1, ..., m_K]
m_k ← (1 - α_k)·m_k + α_k·f_enc(x_t),   α_k ∝ similarity(x_t, m_k)
```

**Context-aware Attention** (`src/drre/attention.py`)

```
α_k = softmax_k(q_t · m_k / √d),   q_t = W_q x_t
h̃_t^c = Σ_k α_k · m_k
δ_t = ‖x_t − h̃_t^c‖₂   hoặc   δ_t = 1 − max_k(α_k)
```

**Graph-based Relationship Encoder** (`src/drre/graph_encoder.py`)

```
g_t^c = AGGREGATE({ f_msg(g_u^(l-1)) : u ∈ N(c) }),   l = 1,...,L
```

**Adaptive Risk Fusion** (`src/drre/fusion.py`)

```
β = softmax(W_β · [x_t ‖ h̃_t^c ‖ g_t^c]) ∈ R³
e_t^c = β₁·φ₁(x_t) + β₂·φ₂(h̃_t^c) + β₃·φ₃(g_t^c)
```

Cơ chế cold-start: khách hàng mới (Behavior Memory rỗng) → gate dồn trọng số sang tín hiệu đồ thị + giao dịch thô; khách hàng lâu năm nhưng giao dịch cô lập → dồn trọng số sang Behavior Memory.

**Mục tiêu huấn luyện** (`src/drre/losses.py`)

```
L_pred     = ‖ ê_{t+1}^c − e_{t+1}^c ‖²₂
L_contrast = −log[ exp(sim(e_i,e_i⁺)/τ) / Σ_j exp(sim(e_i,e_j⁻)/τ) ]
L          = L_pred + λ₁·L_contrast + λ₂·L_BCE
```

> **⚠ TODO kỹ thuật bắt buộc trước khi train**: `L_pred` so sánh output của encoder với chính nó tại t+1. Nếu không có target network riêng (EMA của encoder, theo pattern BYOL/SimSiam) hoặc stop-gradient, nghiệm tầm thường (representation collapse — mọi embedding hội tụ về một điểm) sẽ làm loss → 0 mà biểu diễn vô nghĩa. `config.yaml: drre.loss.target_network` đã đặt mặc định là `"ema"` — **không tắt** cơ chế này khi implement `src/drre/losses.py`.

## Module 4.5 — Quantum-enhanced Embedding (tùy chọn, mặc định tắt)

```
K_Q(e_i, e_j) = | ⟨0| U†(e_j) U(e_i) |0⟩ |²          (Quantum Kernel)
φ₄(e_t^c) = ⟨ψ(e_t^c; θ)| Ô |ψ(e_t^c; θ)⟩              (VQC, phương án nâng cao)
```

Giới hạn: simulator Qiskit mô phỏng hiệu quả tới ~10–16 qubit → chỉ nhận bản nén PCA của embedding. **Trước khi bật `quantum.enabled: true`**, phải chạy `src/evaluation/metrics.py` so sánh AUPRC có/không có nhánh này trên cùng tập kiểm định — nếu không cải thiện đáng kể, giữ tắt và báo cáo trung thực trong `README.md §9`.

## Module 5 — Risk Scoring Engine

```
r_t = 100 · σ(w · e_t^c + b)
```

| Khoảng điểm | Phân loại | Hành động (Module 7) |
|---|---|---|
| 0–30 | Low Risk | Xử lý bình thường |
| 31–70 | Medium Risk | Xác thực bổ sung / theo dõi |
| 71–100 | Critical | Can thiệp ngay: OTP, tạm hoãn, chuyển đội xử lý gian lận |

## Module 6 — Explainable AI

Giải thích nội sinh từ ba nguồn, không phải lớp diễn giải gắn thêm:

| Nguồn | Ví dụ diễn giải |
|---|---|
| Attention α_k | "Giao dịch không khớp với bất kỳ mẫu hành vi lịch sử nào (max α_k = 0.11)" |
| Gate β | "Hệ thống dựa chủ yếu vào tín hiệu mạng lưới (β_graph = 0.71) vì khách hàng mới" |
| Attention trên đồ thị (GAT) | "Tài khoản liên quan mạnh nhất là tài khoản X (trọng số cạnh 0.58)" |

SHAP chỉ dùng như đầu dò phụ trên nhánh bán giám sát, **không phải nguồn giải thích chính**.

## Module 8 — Quantum Alert Prioritization Engine (QAPE)

**Phát biểu bài toán** (`src/qape/qubo.py`)

```
max  Σ_i (r_i · u_i) · x_i
s.t. Σ_i c_i · x_i ≤ C
     x_i = 1  ∀i : r_i ≥ 90
```

**QUBO (penalty method)**

```
H_C = −Σ_i (r_i·u_i)·x_i + λ₁·(Σ_i c_i·x_i − C)² + λ₂·Σ_{i: r_i≥90} (1 − x_i)
```

**Giới hạn quy mô — trình bày trung thực**: QAOA trên simulator Qiskit khả thi tới ~20–30 qubit (~20–30 cảnh báo/lô). Ở quy mô này, **ILP cổ điển (OR-Tools/PuLP) gần như chắc chắn nhanh và tốt hơn QAOA trên simulator**. Đây là proof of concept cho khả năng tích hợp khi bài toán mở rộng lên quy mô toàn ngân hàng — **không claim quantum advantage ở quy mô demo**.

**Benchmark bắt buộc** (`src/qape/benchmark.py`): Greedy vs ILP (exact, reference) vs Simulated Annealing (cùng QUBO) vs QAOA. Chỉ số: approximation ratio so với ILP, tỷ lệ vi phạm ràng buộc, thời gian chạy, số vòng lặp hội tụ.

> **⚠ Sensitivity analysis bắt buộc**: λ₁, λ₂ trong QUBO hiện đang hiệu chỉnh thủ công (`config.yaml: qape.qubo_penalty`). Phải chạy sweep trên vài giá trị và báo cáo độ nhạy của approximation ratio — nếu không, kết quả benchmark dễ bị nghi ngờ là "chọn λ để ra kết quả đẹp".

## KPI (Module `src/evaluation/`)

- AUPRC (không dùng Accuracy/AUC-ROC do mất cân bằng nhãn nặng)
- So sánh DRRE với ensemble XGBoost/LightGBM/Autoencoder (baseline phiên bản trước)
- Hiệu năng trên cold-start subset (khách hàng mới)
- Độ ổn định khi concept drift (chèn fraud pattern mới, đo suy giảm Recall)
- QAOA approximation ratio + tỷ lệ vi phạm ràng buộc so với ILP
- Decision Agreement Rate (tỷ lệ chuyên viên đồng thuận với đề xuất hệ thống)

## Ràng buộc vận hành chưa được kiểm chứng — cần đo trước khi công bố

- **Latency thời gian thực**: pipeline suy luận mỗi giao dịch gồm GRU update → attention K-slot → L-hop graph message passing → gating fusion → (tùy chọn) quantum kernel. Chưa có SLA hay chiến lược serving (batch/streaming, neighbor caching). Đo p95 inference time trước khi giữ nguyên tuyên bố "giám sát thời gian thực" trong pitch.
- **Tuân thủ dữ liệu cá nhân**: đồ thị quan hệ tài khoản + dữ liệu thiết bị/IP thuộc phạm vi Nghị định 13/2023 về bảo vệ dữ liệu cá nhân — cần một dòng compliance song song với Basel/IFRS 9 đã có.
