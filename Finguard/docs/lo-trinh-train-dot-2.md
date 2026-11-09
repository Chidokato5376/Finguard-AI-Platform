# Lộ trình huấn luyện đợt 2 — DRRE trước, mô hình khác sau

Nhóm CMET (AQ2026-111) · lập 08/11/2026 · bắt đầu sau chung kết (11/11)

Mục tiêu của đợt này: sửa toàn bộ lỗi logic của DRRE và bổ sung đặc trưng theo
từng kịch bản rửa tiền, rồi chứng minh bằng số đo **DRRE đóng góp gì mà mô hình
cổ điển không làm được**. Chỉ sau khi có kết luận đó mới tính tới việc thêm mô
hình khác (XGBoost cho ACH, LightGBM cho kênh còn lại).

---

## 1. Tám lỗi cần sửa

| # | Lỗi | Nơi | Giai đoạn sửa |
|---|---|---|---|
| 1 | Chấm theo **tài khoản gửi** trong khi bài toán là chấm **giao dịch** (cạnh) | `src/drre/model.py`, `train.py` | 2B |
| 2 | φ₁, φ₂, φ₃ và đầu chấm điểm đều **tuyến tính**; hợp nhất chỉ **cộng** | `src/drre/fusion.py`, `src/scoring/risk_scoring.py` | 2A |
| 3 | Ba nhánh dùng **chung đầu vào** → cổng sụp (β_graph 0,924) | `src/drre/model.py` | 2A |
| 4 | δ_t (độ lệch thói quen) tính ra nhưng **không vào điểm** | `src/drre/attention.py`, `fusion.py` | 2A |
| 5 | Đồ thị một chiều, không có đặc trưng cạnh, nút chỉ-nhận mang vector 0, dựng cho cả cửa sổ nên **nhìn trước tối đa ~25 phút** | `src/dashboard/graph_utils.py`, `src/drre/graph_encoder.py` | 2B |
| 6 | 83% hàm mất mát cho hai nhiệm vụ phụ đã bão hòa; mỗi cập nhật chỉ thấy ~31 vụ; thứ tự cửa sổ cố định | `src/drre/losses.py`, `train.py` | 2C |
| 7 | Đầu vào không chuẩn hóa; kênh đi vào như số 0–6 | `src/drre/model.py`, `train.py` | 2A |
| 8 | `evaluate.py` chấm DRRE **không mang trạng thái** bộ nhớ (đặt về 0 mỗi lô) | `src/dashboard/scoring_service.py`, `scripts/evaluate.py` | 0 |

---

## 2. Giao thức chung cho mọi lần chạy

| Mục | Quy định |
|---|---|
| Chia tập | Theo thời gian 70/15/15. Trong train: **15% cuối là tập giữ lại** (dừng sớm, huấn luyện đầu chấm điểm pha 2, hiệu chuẩn). Val: chọn mô hình và ngưỡng. Test: **chạy đúng một lần** ở cuối |
| Vũ trụ chấm điểm | Bỏ Wire, Reinvestment khỏi phép chấm và mất mát có nhãn; vẫn giữ để dựng lịch sử và đồ thị |
| Bộ dữ liệu | Phát triển trên **HI-Small**; xác nhận trên **HI-Medium**; kiểm tra chuyển giao trên LI |
| Seed | Phát triển: 1 seed (42). Mỗi cổng kiểm tra: 3 seed (42, 43, 44), báo trung vị kèm dải |
| Đối chứng | XGBoost (hoặc GBDT hiện tại) nhận **đúng cùng đặc trưng thủ công** như DRRE |
| Chọn checkpoint | Trung bình trượt val AUPRC (lưu ý: không bao giờ chọn epoch đầu và cuối) |
| Thang đo bắt buộc | AUPRC + độ nâng; bắt@500/1000/2000; **ma trận nhầm lẫn ở ngưỡng chọn trên val**; theo kênh; theo kiểu hình; KTC bootstrap ghép cặp so với đối chứng |
| Truy nguyên | Commit mã trước mỗi lần chạy; mỗi lần chạy có manifest; ghi bảng kết quả vào `reports/` |
| Máy | Kiểm tra **cắm sạc** (`BatteryStatus = 2`) trước mỗi phiên; dùng `--keep-awake`; chia phiên bằng `STOP_AFTER` |

---

## 3. Các giai đoạn

### Giai đoạn 0 — Sửa cách đo (0,5 ngày)

| Việc | Đầu ra |
|---|---|
| 0.1 `evaluate.py` chấm DRRE bằng `run_encoder_over_window` + `BehaviorStateStore`: quét train để nạp trạng thái, rồi chấm val | Số DRRE công bằng |
| 0.2 Thêm vào `evaluate.py`: ma trận nhầm lẫn ở ngưỡng chọn trên val, bảng theo kênh, F1 lớp rửa tiền (để so với bài báo IBM) | Báo cáo đủ thang đo |
| 0.3 **R0**: đánh giá lại DRRE v1 (HI-Small và HI-Medium epoch 3) bằng cách đo mới | Mốc xuất phát chính thức |

### Giai đoạn 1 — Mô-đun đặc trưng theo kịch bản (2–3 ngày)

Mỗi kịch bản là một mô-đun độc lập `src/data/scenarios/<tên>.py`, có test nhân
quả (thêm dòng tương lai thì giá trị cũ không đổi) và cờ bật/tắt.

| Mô-đun | Đặc trưng thủ công (quanh 1 bước) | Phần giao cho DRRE |
|---|---|---|
| `fan_out` | Số người nhận khác nhau / người nhận mới trong 24 giờ; độ đều số tiền chuyển đi | Người nhận có cùng chuyển tiếp về một nơi |
| `fan_in` | Số người gửi khác nhau tới người nhận trong 24 giờ; tổng tiền nhận 24 giờ; tỷ lệ người gửi mới | Nguồn chung phía trên |
| `gather_scatter` | Tỷ lệ trung chuyển 24 giờ; số phút từ nhận tới chuyển; bậc vào × bậc ra 24 giờ | — |
| `scatter_gather` | — | Đường 2 bước qua trung gian về một điểm gom |
| `cycle` | `reciprocity` | Vòng 3–6 bước |
| `mule_chain` | `forward_ratio`, `amount_similarity_in`, số phút từ lần nhận gần nhất, tuổi tài khoản | Độ sâu chuỗi |
| `synchronized` (BIPARTITE/STACK) | Số người gửi cùng chuyển cho một nhóm người nhận trong cửa sổ ngắn; độ giống số tiền | Hành động đồng bộ nhiều lớp |
| `hub` | `hub_degree` hai chiều | Vị trí trung tâm nhiều bước |
| `cross_bank_fx` | `cross_bank`, số ngân hàng của đối tác, `log_amount_usd` (tỷ giá chỉ dựng từ train) | — |
| `cash_structuring` | Tổng tiền mặt 24 giờ so với ngưỡng; số giao dịch sát ngưỡng; số kênh đã dùng | — |
| `integration` | Lệch khỏi chu kỳ đều đặn; khoản lớn lần đầu cho doanh nghiệp | Nhịp hành vi dài hạn |
| (thay thế) | Đặc trưng **cửa sổ 1 giờ / 24 giờ / 7 ngày** thay cho cộng dồn từ đầu | — |
| (bỏ) | `is_round_amount`, `hour_sin` | — |

**R1**: đo đóng góp từng mô-đun bằng đối chứng (nhanh, mỗi lần vài phút trên
HI-Small). Giữ mô-đun khi KTC 95% của mức tăng AUPRC nằm trên 0, **hoặc** khi
nó tăng recall có ý nghĩa cho đúng kiểu hình nó nhắm tới. Đo trên cả HI và LI để
tránh học theo máy sinh dữ liệu.

### Giai đoạn 2 — Sửa DRRE

**2A — Tầng chấm điểm (1–1,5 ngày):** chuẩn hóa đầu vào (học trên train, lưu
trong checkpoint); embedding 8 chiều cho kênh; φ và đầu chấm điểm thành MLP 2
lớp; hợp nhất bằng ghép nối [φ₁ ‖ φ₂ ‖ φ₃ ‖ φ₁⊙φ₃] rồi MLP (β chỉ để giải
thích); đưa δ_t và |x_t − h̃_t| vào; **tách đầu vào theo nhánh** (giao dịch /
thói quen / mạng lưới theo bảng mô-đun).

**2B — Nhánh đồ thị (2–3 ngày):** đồ thị **chỉ gồm cạnh quá khứ** của N giờ
trước (cửa sổ trượt, dựng theo khối ~2.048 giao dịch); truyền tin hai chiều với
hai bộ trọng số; đặc trưng cạnh (số tiền, kênh, khoảng thời gian) qua GINEConv
hoặc GAT có edge_attr; đặc trưng nút gồm cả phía nhận; **đọc ra theo cạnh**:
MLP([x_t ‖ g_gửi ‖ g_nhận ‖ h̃_gửi ‖ δ_t]); 3 lớp truyền tin. Test: thêm cạnh
tương lai thì điểm cũ không đổi.

**2C — Cách huấn luyện (1 ngày):** pha 1 tự giám sát 1–2 epoch; pha 2 **đóng
băng bộ mã hóa, trích embedding toàn bộ train, huấn luyện đầu chấm điểm trên
mọi vụ cùng lúc**; bộ đệm phát lại vụ rửa tiền; cộng dồn gradient 8–16 cửa sổ;
so sánh BCE trọng số 930 / √930 / focal loss.

Mỗi thay đổi định dạng checkpoint tăng `RESUME_FORMAT`; kiến trúc mới đặt sau
cờ `--arch v2`, DRRE v1 vẫn chạy được để đối chiếu.

### Giai đoạn 3 — Kiểm chứng (2 ngày + ~50 giờ máy)

| Phép đo | So sánh | Câu hỏi |
|---|---|---|
| P1 | DRRE v2 so với XGBoost, cùng đặc trưng | Kiến trúc DRRE có hơn mô hình cổ điển không |
| P2 | XGBoost so với XGBoost + embedding DRRE v2 | DRRE có mang thông tin mà đặc trưng thủ công không có không |
| P3 | Tỷ trọng đóng góp của nhóm DRRE (xáo trộn nhóm, SHAP theo nhóm) | DRRE gánh bao nhiêu phần quyết định |
| P4 | P1, P2 theo kiểu hình, theo kênh, tài khoản mới / lâu năm | DRRE mạnh ở đâu |
| P5 | Đường học Small → Medium | DRRE có hưởng lợi từ dữ liệu lớn không |

**Điều kiện thành công riêng của DRRE:** recall của SCATTER-GATHER, CYCLE,
STACK, BIPARTITE cao hơn XGBoost có ý nghĩa thống kê.

### Giai đoạn 4 — Quyết định và chốt

| Kết quả | Hướng đi |
|---|---|
| P1 ngang hoặc thắng | DRRE v2 là mô hình chính |
| P1 thua nhưng thắng ở kiểu nhiều bước (P4) | DRRE chuyên trách các kiểu đó |
| P2 dương, P1 thua | Kết hợp: mô hình cổ điển chấm điểm, DRRE cấp biểu diễn; báo cáo P2, P3 |
| P2 không dương | Báo cáo trung thực; quay lại chẩn đoán |

Sau đó mới thêm M1 (XGBoost cho ACH) và M2 (LightGBM cho kênh còn lại) nếu cần,
chọn ngưỡng theo phân khúc trên val, rồi **chạy tập test một lần**.

---

## 4. Thang chạy thử (ablation)

Mỗi bậc chỉ thêm một thay đổi so với bậc trước, để biết đóng góp của từng thay đổi.

| Mã | Cấu hình | Dữ liệu | Seed | Epoch | Máy (ước lượng, cắm sạc) |
|---|---|---|---|---|---|
| R0 | DRRE v1, đo lại đúng cách | HI-Small, HI-Medium | — | — | ~1,5 giờ |
| R1 | Đối chứng + từng mô-đun kịch bản | HI-Small, LI-Small | 1 | — | ~2 giờ |
| R2 | DRRE v1 + đặc trưng đợt 2 | HI-Small | 1 | 4 | ~1,5 giờ |
| R3 | + 2A | HI-Small | 1 | 4 | ~2 giờ |
| R4 | + 2B1 (đồ thị chỉ quá khứ) | HI-Small | 1 | 4 | ~2,5 giờ |
| R5 | + 2B2–2B5 (hai chiều, đặc trưng cạnh, đọc theo cạnh) | HI-Small | 1 | 4 | ~3 giờ |
| R6 | + 2C (cách huấn luyện) | HI-Small | 1 | 4 | ~3 giờ |
| **Cổng 1** | R6 so với đối chứng, P1–P4 | HI-Small | **3** | 4 | ~9 giờ |
| R7 | Cấu hình cuối | **HI-Medium** | **3** | 4 | ~45–55 giờ (chạy qua đêm nhiều phiên) |
| R8 | Chuyển giao HI → LI | LI-Medium | 1 | — | ~1 giờ |
| R9 | Kết hợp / M1 / M2 (nếu cần) | HI-Medium | 3 | — | ~2 giờ |
| **R10** | **Tập test, một lần** | HI-Medium | — | — | ~1 giờ |

---

## 5. Lịch đề xuất

| Thời gian | Việc |
|---|---|
| 09/11 | Chỉ R0 (đo lại DRRE v1) và kết quả P2/P3 cho slide; **không** sửa kiến trúc trước chung kết |
| 10/11 | Chung kết |
| 11–15/11 | Giai đoạn 0, 1, 2A; chạy R0–R3 |
| 16–20/11 | Giai đoạn 2B, 2C; chạy R4–R6; Cổng 1 |
| 21–28/11 | R7 trên HI-Medium (qua đêm), R8, R9, quyết định giai đoạn 4, R10 |

## 6. Ước lượng kết quả (phán đoán, không phải số đo)

| Sau bậc | DRRE val AUPRC trên HI-Medium |
|---|---|
| Hiện tại (đo trong lúc huấn luyện, epoch 3) | 0,400 |
| R4 (bỏ nhìn trước tương lai) | 0,37 – 0,40 |
| R3 + R4 (tầng chấm điểm đã sửa) | 0,43 – 0,48 |
| R5 (đồ thị đúng nghĩa) | 0,46 – 0,52 |
| R6 (cách huấn luyện) | 0,47 – 0,54 |
| + đặc trưng đợt 2 (đối chứng cũng tăng, ~0,54–0,58) | 0,50 – 0,58 |
