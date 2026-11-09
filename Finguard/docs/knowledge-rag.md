# Kho tri thức + truy xuất vector (RAG) cho giải thích cảnh báo

Mô hình chấm ra một con số. Con số đó không trả lời được câu hỏi mà chuyên
viên rủi ro thật sự hỏi: *"giao dịch này giống kiểu rửa tiền nào, và tôi phải
tìm thêm bằng chứng gì"*. Module `src/knowledge/` nối hai thứ đó.

---

## Kiến trúc

```
đặc trưng của giao dịch (số)
   │
   ├─► chuẩn hóa z-score theo tập train          retriever.z_scores
   │
   ├─► ĐƯỜNG SỐ:  khớp kiểu hình                 retriever.match_by_signals
   │      ├── bộ phân loại HỌC từ train  (khi có nhãn kiểu hình)
   │      └── chữ ký VIẾT TAY           (khi chưa có nhãn nào)
   │
   ├─► ĐƯỜNG CHỮ: câu truy vấn tiếng Việt        retriever.signals_to_query
   │      └── truy xuất vector trên 8 văn bản    retriever.search
   │
   └─► ghép thành lời giải thích                 narrative.build_narrative
```

Hai đường bổ cho nhau: đường số chạy tự động cho mọi cảnh báo nhưng không trả
lời được câu hỏi mở; đường chữ trả lời được câu hỏi mở nhưng cần ai đó gõ câu
hỏi. `signals_to_query` nối chúng — biến tín hiệu số thành câu tiếng Việt.

---

## Vì sao TF-IDF chứ không phải embedding

Kho tri thức chỉ có **8 văn bản**. Ở quy mô đó, một mô hình ngôn ngữ lớn
không mang lại gì thêm so với TF-IDF ở mức ký tự (`char_wb`, 3–5 gram), trong
khi TF-IDF:

- không cần tách từ tiếng Việt (vốn là bài toán riêng);
- không cần tải ~90 MB trọng số, chạy được offline trên máy yếu;
- cho kết quả **tái lập 100%** giữa các lần chạy.

Đường embedding vẫn có sẵn (`encoder="embedding"`) nếu môi trường cài
`sentence-transformers`. `available_encoders()` cho biết có dùng được không.

**Đo được:** đường chữ trả đúng kiểu hình ở hạng 1 cho cả 4 câu hỏi thử
(FAN-IN, FAN-OUT, CYCLE, STACK) — xem `tests/test_knowledge.py`.

---

## Đường số: hai cách khớp, và vì sao giữ cả hai

Đo trên **461 giao dịch val AMLWorld có nhãn kiểu hình thật**, bộ phân loại
học trên **1.529 dòng train**:

| Kiểu hình | n | chữ ký tay top-1 / top-3 | học top-1 / top-3 |
|---|---|---|---|
| STACK | 105 | 1,0% / 29,5% | **32,4% / 76,2%** |
| SCATTER-GATHER | 93 | 11,8% / **83,9%** | 18,3% / 76,3% |
| GATHER-SCATTER | 70 | 0,0% / 25,7% | 5,7% / 42,9% |
| CYCLE | 51 | 0,0% / 9,8% | **29,4% / 54,9%** |
| FAN-IN | 44 | **72,7% / 90,9%** | 54,5% / 70,5% |
| FAN-OUT | 41 | **61,0%** / 82,9% | 53,7% / 82,9% |
| BIPARTITE | 35 | 0,0% / 2,9% | **22,9% / 65,7%** |
| RANDOM | 22 | 0,0% / 31,8% | 4,5% / 50,0% |
| **TỔNG** | **461** | **15,0% / 46,4%** | **27,1% / 66,8%** |
| *mốc đoán bừa 1/8* | | *12,5% / 37,5%* | |
| *mốc luôn đoán STACK* | | *22,8%* | |

Đọc bảng này cho đúng:

- **Đường học thắng tổng thể** và vượt được cả mốc "luôn đoán kiểu phổ biến
  nhất" (27,1% > 22,8%), còn chữ ký tay thì **thua mốc đó** (15,0%).
- Nhưng chữ ký tay **mạnh hơn ở đúng hai kiểu hình một bước** (FAN-IN 72,7%
  vs 54,5%; FAN-OUT 61,0% vs 53,7%) — vì fan-in và fan-out đúng là bậc đỉnh,
  thứ mà đặc trưng một giao dịch đo trực tiếp được.
- Chữ ký tay bó tay với STACK, CYCLE, BIPARTITE (0–1%) vì ba kiểu này cần
  ngữ cảnh đồ thị **nhiều bước**. Bộ phân loại học được chúng từ các tương
  tác giữa đặc trưng mà tay không viết ra được.
- **Cả hai đều gần như bất lực với GATHER-SCATTER (5,7%) và RANDOM (4,5%)** —
  đúng như `Typology.distinguishable=False` đã khai báo TRƯỚC khi đo. Hai
  kiểu gom-phân tán chỉ khác nhau ở THỨ TỰ, mà một dòng giao dịch đơn lẻ
  không mang thông tin đó.

**Con số dùng được trong thực tế là top-3 = 66,8%**: đưa cho chuyên viên ba
kiểu hình khả dĩ thì hai phần ba số lần có kiểu đúng trong đó. Top-1 không
phải cách dùng đúng của công cụ này.

Vì vậy giữ cả hai đường: chữ ký tay là đường chạy khi **chưa có nhãn kiểu
hình nào** (bộ dữ liệu mới, hoặc ngân hàng thật chưa gán nhãn typology), và
nó **đọc được bằng mắt**.

---

## Một lỗi đã mắc và cách phát hiện

Bản đầu của `match_by_signals` chỉ chặn vector đúng bằng 0. Nhưng **cosine bỏ
qua độ lớn**, nên một giao dịch hoàn toàn bình thường (mọi z quanh −0,01) vẫn
khớp SCATTER-GATHER ở **0,75** — tức là trên dashboard, *mọi* giao dịch đều
được gán một kiểu hình nghe rất chắc chắn.

Lỗi này do `tests/test_knowledge.py::test_khong_co_tin_hieu_thi_khong_khop_gi_het`
bắt được, không phải do đọc lại code. Bản vá: đòi có **ít nhất một tín hiệu
vượt `SIGNAL_THRESHOLD`** (1,0 độ lệch chuẩn) mới khớp — cùng ngưỡng mà
`signals_to_query` dùng để quyết định một tín hiệu có đáng nhắc tới không.

Sau khi vá, số đo của chữ ký tay giảm nhẹ (top-1 15,2% → 15,0%, top-3 47,9%
→ 46,4%) vì những lần "khớp" trên tín hiệu yếu trước đây nay bị loại.

---

## Hai cam kết

**1. Không bịa.** Mọi câu trong lời giải thích đến từ một trong hai nguồn:
tín hiệu **đã đo** trên chính giao dịch đó, hoặc văn bản **đã có** trong
`typology_kb`. Không có bước nào sinh ra khẳng định mới.
Kiểm bởi `test_loi_giai_thich_chi_lay_noi_dung_tu_kho_tri_thuc`.

**2. Không gửi dữ liệu thật ra ngoài.** Đường LLM (tùy chọn, **mặc định
tắt**) chỉ nhận các cụm từ mô tả tín hiệu và văn bản kiểu hình — không số
tiền, không định danh tài khoản, không dấu thời gian. Cùng cam kết với
`src/augmentation/llm_adapter.py`.
Kiểm bởi `test_prompt_llm_khong_chua_dinh_danh_hay_so_tien`.

Khi tín hiệu yếu, câu trả lời đúng là **"không khớp kiểu hình nào rõ rệt"**
chứ không phải chọn đại kiểu hình có điểm cao nhất.

---

## Giới hạn, nói trước

- Mô tả 8 kiểu hình là **kiến thức nghiệp vụ phổ thông**, viết lại cho dự án.
  Không phải trích dẫn tài liệu pháp lý, không dùng làm căn cứ tuân thủ.
- Kho chỉ phủ 8 kiểu hình AMLWorld gán nhãn. Rửa tiền thật còn nhiều kiểu
  khác.
- Tham chiếu z-score trên dashboard fit trên **toàn bộ bộ dữ liệu đang nạp**,
  không phải tập train riêng — với dashboard đó là ngữ nghĩa đúng ("lệch bao
  nhiêu so với mặt bằng lô đang xem"), nhưng khi **đánh giá mô hình** thì vẫn
  phải `fit_reference` trên train.
- Chưa có OCR. Phạm vi hiện tại là **vector + RAG trên văn bản tri thức**
  (mức B). Xử lý chứng từ ảnh/PDF là bước sau.
