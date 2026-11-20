# Bài toán: dạng, đầu vào, đầu ra

Tài liệu này phát biểu rõ hai bài toán của FinGuard AI **trước khi** nói tới mô
hình. Đây là bước quyết định: chọn sai dạng bài toán thì mọi lựa chọn phía sau
— thước đo, mô hình đối chứng, cách chia dữ liệu — đều sai theo.

---

## Bài toán 1 — DRRE: chấm điểm rủi ro giao dịch

### Dạng bài toán

**Phân loại nhị phân trên dữ liệu mất cân bằng cực đoan, có cấu trúc chuỗi
thời gian và cấu trúc đồ thị.**

Bốn tính chất này, mỗi tính chất ràng buộc một quyết định kỹ thuật:

| Tính chất | Đo được | Ràng buộc nó đặt ra |
|---|---|---|
| Nhị phân | nhãn ∈ {hợp lệ, rửa tiền} | Không dùng hồi quy tuyến tính (xem dưới) |
| Mất cân bằng cực đoan | tỷ lệ nền 0,080–0,119% tùy tập | Thước đo phải là AUPRC, không phải accuracy; mô hình phải cân bằng lớp |
| Chuỗi thời gian | mỗi tài khoản có lịch sử giao dịch | Chia tập theo thời gian, không bao giờ random; đặc trưng phải nhân quả |
| Đồ thị | định danh thật ở **cả hai đầu** giao dịch | Có chỗ cho nhánh đồ thị; 8 kiểu hình rửa tiền đều là hình dạng mạng lưới |

### Đầu vào

Một giao dịch đã chuẩn hóa về **Unified Schema** (`src/data/schema.py`), gồm
10 trường: `account_id`, `counterparty_id`, `timestamp`, `amount`, `channel`,
`device_id`, `ip_country`, `label`, `source`, `is_synthetic_field`.

Từ 10 trường đó, `src/data/features.py` sinh ra bộ đặc trưng **nhân quả** —
mỗi đặc trưng chỉ dùng các giao dịch **trước** giao dịch đang xét:

| Nguồn | Số đặc trưng | Bỏ nhóm nào và vì sao |
|---|---|---|
| `amlworld` / `transxion` | 14 | bỏ `is_fx` — rò rỉ ngược, 0 nhãn dương trên 72.170 giao dịch đổi tiền |
| `paysim` | 11 | bỏ nhóm đồng hồ — giờ 4–8 có 100% gian lận; bỏ `is_fx` (không có) |
| `ieee_cis` | 10 | bỏ nhóm đối tác — bộ này không có `counterparty_id` |
| `synthetic_vn` | 14 | — |

Mô hình **không** nhận: số dư tài khoản (rò rỉ), cờ gian lận của máy sinh
(chính là đáp án), và nhãn kiểu hình `laundering_pattern` (cũng là nhãn, chỉ
dùng để đánh giá).

### Đầu ra

Một **điểm rủi ro 0–100** cho mỗi giao dịch (`src/scoring/risk_scoring.py`),
kèm bốn tín hiệu giải thích mà mô hình tự tính ra: `max_attention_weight`
(giao dịch có khớp mẫu hành vi nào không) và ba trọng số gate
`beta_transaction` / `beta_behavior` / `beta_graph` (hệ thống đang dựa vào
nguồn thông tin nào).

### Quyết định mà đầu ra này phục vụ

Đây là phần hay bị bỏ qua, và nó quyết định thước đo. Ngân hàng **không**
dùng điểm số để tự động chặn. Họ có một đội điều tra với năng lực hữu hạn:
mỗi ngày xem được khoảng N cảnh báo. Câu hỏi thật là:

> Trong N giao dịch nghi nhất mà hệ thống đưa lên, có bao nhiêu vụ rửa tiền thật?

Vì vậy thước đo chính là **AUPRC** (chất lượng xếp hạng ở đầu danh sách) và
**bắt được bao nhiêu trong top N** với N = 500 / 1.000 / 2.000
(`DEFAULT_ALERT_BUDGETS`). Các con số này khớp trực tiếp với ràng buộc năng
lực trong bài toán QAPE bên dưới.

### Vì sao KHÔNG dùng hồi quy tuyến tính

Hồi quy tuyến tính dự đoán một đại lượng **liên tục**. Đầu ra ở đây là nhị
phân, nên nó không áp dụng — ép dùng sẽ cho giá trị ngoài khoảng [0, 1] và
giả định phương sai đồng nhất bị vi phạm hoàn toàn khi tỷ lệ nền là 0,1%.

Hồi quy tuyến tính **sẽ** là lựa chọn đúng cho một bài toán khác mà dự án
chưa làm: dự báo **số tiền tổn thất** của một vụ đã xác định là gian lận.
Đó là đại lượng liên tục. Nếu nhóm muốn bổ sung, đó là chỗ của nó.

### Thước đo: dùng gì và tránh gì

| Thước đo | Dùng | Lý do |
|---|---|---|
| **AUPRC** | chính | Nhạy với chất lượng đầu danh sách, đúng cái ta cần |
| **bắt@500/1000/2000** | chính | Dịch thẳng ra ngôn ngữ vận hành |
| Recall theo kiểu hình | chính | 8 kiểu hình là hình dạng đồ thị → đo trực tiếp đóng góp của nhánh đồ thị |
| ECE (hiệu chuẩn) | phụ | Điểm số có đáng tin như một xác suất không |
| Bootstrap KTC 95% | bắt buộc | Chênh lệch AUPRC thô không cho biết có vượt nhiễu lấy mẫu hay không |
| ROC-AUC | **chỉ tham khảo** | Đã đo: hai lần chạy ROC-AUC 0,790 vs 0,772 gần như bằng nhau trong khi AUPRC lệch 13 lần. Không dùng làm bằng chứng chất lượng |
| Accuracy | **không bao giờ** | Đoán "mọi giao dịch đều hợp lệ" đã đạt 99,9% |

**Mốc nền phải báo kèm mọi con số.** AUPRC phụ thuộc trực tiếp vào tỷ lệ nền,
và ba tập của AMLWorld có tỷ lệ nền lệch nhau 1,48 lần (train 0,0804%, val
0,0998%, test 0,1190%). Một con số AUPRC không kèm tỷ lệ nền của chính tập đó
là một con số không đọc được.

---

## Bài toán 2 — QAPE: xếp lịch đội điều tra

### Dạng bài toán

**Tối ưu tổ hợp có ràng buộc** — cụ thể là bài toán cái túi (knapsack) với
ràng buộc cứng, được chuyển sang QUBO rồi sang Ising để chạy QAOA.

### Đầu vào

- Danh sách cảnh báo đã chấm điểm từ DRRE, mỗi cảnh báo có điểm rủi ro và
  chi phí điều tra ước tính;
- Năng lực của đội trong ca làm việc (tổng chi phí tối đa);
- Ràng buộc cứng: cảnh báo có `is_critical` (điểm ≥ 90) **bắt buộc** phải
  được chọn.

### Đầu ra

Tập con cảnh báo được xử lý trong ca, tối đa hóa tổng rủi ro được xử lý.

### Thước đo

Tỷ lệ xấp xỉ so với nghiệm tối ưu của ILP
(`compute_qaoa_approximation_ratio`), và tỷ lệ trùng quyết định giữa các bộ
giải. Đối chứng: Greedy, ILP (CBC), Simulated Annealing, Quy hoạch động.

---

## Hai bài toán nối với nhau ở đâu

Ngưỡng `is_critical = điểm ≥ 90` là điểm nối duy nhất. Nó biến đầu ra liên
tục của bài toán 1 thành ràng buộc cứng của bài toán 2. Đã đo: ràng buộc này
làm ca 2 giờ trở thành **bất khả thi** trên lô thật — tức là ngưỡng 90 đang
tạo ra nhiều cảnh báo bắt buộc hơn năng lực đội. Đó là một kết quả có ý nghĩa
nghiệp vụ, không phải lỗi.

---

## Tóm tắt một dòng

> Bài toán 1 là **xếp hạng** 5 triệu giao dịch sao cho 2.000 giao dịch nghi
> nhất chứa càng nhiều vụ rửa tiền càng tốt. Bài toán 2 là **chọn** tập con
> nào trong số đó để đội điều tra xử lý trong ca. Không bài nào là bài toán
> "dự đoán một con số", nên không bài nào dùng hồi quy tuyến tính.
