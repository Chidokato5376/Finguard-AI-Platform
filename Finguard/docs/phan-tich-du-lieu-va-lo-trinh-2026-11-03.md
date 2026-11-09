# FinGuard AI — Phân tích dữ liệu, câu hỏi kinh doanh và lộ trình

Nhóm CMET (AQ2026-111) · cập nhật 03/11/2026

Mọi con số về dữ liệu dưới đây được đếm lại trực tiếp từ file gốc và file
parquet đã tiền xử lý ngày 03/11/2026, không chép từ báo cáo cũ. Chỗ nào số
đếm được lệch với mô tả trên Kaggle hoặc bài báo gốc đều ghi rõ.

---

## 0. Kết luận một trang

1. **Bộ dữ liệu chính (IBM AMLWorld) giải bài toán CHỐNG RỬA TIỀN (AML), không
   phải bài toán LỪA ĐẢO KHÁCH HÀNG (fraud).** Hồ sơ vòng 1 phát biểu bài toán
   theo hướng lừa đảo (deepfake, chiếm quyền Mobile Banking, money mule), nhưng
   AMLWorld không có thiết bị, IP hay nạn nhân. Nó chỉ có dòng tiền giữa các
   tài khoản ở nhiều ngân hàng, được gắn nhãn "rửa tiền / hợp lệ". Hai bài toán
   này có người dùng, quyết định và thước đo khác nhau (mục 2.1).
2. **May mắn là phần lõi đã làm khớp với AML.** Thước đo "bắt được bao nhiêu vụ
   trong N cảnh báo/ngày", đội điều tra có năng lực hữu hạn (QAPE) và phát hiện
   đường dây đều là ngôn ngữ của một phòng giám sát giao dịch AML. Đề nghị
   **chốt định vị dự án là AML**, còn các kịch bản lừa đảo Việt Nam chỉ là phần
   mở rộng trên dữ liệu tổng hợp.
3. **Câu hỏi kinh doanh trung tâm:** *với đội điều tra chỉ xem được N cảnh
   báo/ngày, hệ thống có chọn ra được nhiều vụ rửa tiền hơn cách làm hiện tại
   hay không, và có giải thích được vì sao không?* Hiện đã trả lời được nửa đầu
   so với các baseline học máy, **chưa trả lời được "so với cách làm hiện tại"**
   vì dự án chưa có bộ luật AML thật để so (mục 2.3).
4. **Số liệu kinh doanh mạnh nhất hiện có** (HI-Medium, tập val, 4,6 triệu giao
   dịch): trong 2.000 cảnh báo đầu, GBDT bắt đúng **1.843 vụ (92,2% cảnh báo là
   thật)**, trong khi xếp theo số tiền chỉ bắt 16 vụ (99,2% cảnh báo sai).
5. **Thời gian còn lại rất ít.** Theo kế hoạch triển khai, chung kết là
   **10/11/2026, còn 7 ngày**. Lộ trình ở mục 5 ưu tiên chốt số liệu, chạy tập
   test một lần và chuẩn bị bảo vệ, thay vì phát triển thêm.

---

## 1. Hiện tại nhóm đang có gì và dùng để làm gì

### 1.1 Dữ liệu

| Bộ dữ liệu | Quy mô | Nguồn | Bài toán gốc của bộ dữ liệu | Vai trò trong dự án |
|---|---|---|---|---|
| **IBM AMLWorld** (6 bộ) | 5–180 triệu giao dịch/bộ, tổng 430,9 triệu | Kaggle, IBM, CDLA-Sharing 1.0 | Phát hiện giao dịch rửa tiền trong cả hệ sinh thái nhiều ngân hàng | **Dữ liệu chính** từ 18/10. Đã dùng HI-Small, LI-Small, HI-Medium, LI-Medium |
| PaySim | 6.362.620 giao dịch | Kaggle, CC BY-SA 4.0 | Phát hiện gian lận chiếm quyền ví di động rồi rút tiền | Dùng tới 17/10; đã ngừng trích AUPRC vì 2 lỗi sinh dữ liệu |
| IEEE-CIS (bản rút gọn) | 300.000 giao dịch, 11 cột | Kaggle (cuộc thi Vesta) | Phát hiện gian lận thẻ trong thương mại điện tử | Dữ liệu thật duy nhất; dùng để đo DRRE tới 17/10 |
| Synthetic VN | 5.273 giao dịch | Nhóm tự sinh | Kịch bản lừa đảo đặc thù Việt Nam | Kiểm thử, demo |
| Bộ tăng cường | 3.675 giao dịch | Nhóm tự sinh bằng luật | 4 kịch bản: chia nhỏ giao dịch, chuỗi mule, chiếm quyền, lừa đầu tư/tình cảm | Đã sinh, đạt kiểm định chất lượng, **chưa dùng để huấn luyện** |

### 1.2 Mã nguồn và kết quả

| Thành phần | Mục đích | Trạng thái |
|---|---|---|
| Tiền xử lý (`src/data/`) | Đưa mọi nguồn về một schema, sinh 20 đặc trưng nhân quả, chia tập theo thời gian, lọc ngày hỏng | Chạy được toàn bộ HI/LI-Medium (31,9 triệu dòng, 12 phút) |
| DRRE (`src/drre/`) | Mô hình chính: GRU nhớ hành vi + attention + GNN đồ thị + cổng hợp nhất → điểm rủi ro 0–100 | Thua GBDT trên mọi bộ (mục 4). Đang huấn luyện lại trên HI-Medium |
| 11 baseline (`scripts/evaluate.py`) | Đối chứng: ngẫu nhiên, theo số tiền, theo kênh, Isolation Forest, logistic, GBDT, cây, RF | GBDT cân bằng là mô hình tốt nhất hiện có |
| Đo lường | AUPRC, bắt@500/1000/2000, recall theo kiểu hình, KTC bootstrap, manifest truy nguyên | Chắc chắn; là điểm mạnh nhất của dự án |
| QAPE (`src/qape/`) | Chọn cảnh báo nào cho đội điều tra xử lý trong ca (knapsack → QUBO → QAOA) | QAOA thua DP/ILP; lượng tử đang tắt đúng theo thiết kế |
| XAI + RAG | Giải thích cảnh báo bằng trọng số attention/cổng và tra cứu tri thức | Có, chưa được đánh giá với người dùng |
| Demo (`finguard-demo/`, `Finguard-Demo/`) | Giao diện giám sát giao dịch và ưu tiên cảnh báo | Có từ 22/09 |
| Kiểm thử | 276 test tự động | Đạt hết; **46 file thay đổi chưa commit** |

---

## 2. Câu hỏi kinh doanh: mô hình phải trả lời điều gì

### 2.1 Lừa đảo (fraud) và rửa tiền (AML) là hai bài toán khác nhau

| | Lừa đảo khách hàng (fraud) | Chống rửa tiền (AML) |
|---|---|---|
| Ai bị hại | Khách hàng bị lừa hoặc bị chiếm tài khoản | Hệ thống tài chính; ngân hàng chịu rủi ro pháp lý |
| Người dùng kết quả | Hệ thống thanh toán (chặn tự động) | Chuyên viên giám sát, điều tra AML |
| Thời điểm quyết định | Trước khi tiền đi, tính bằng mili giây | Sau giao dịch, xử lý theo lô hằng ngày |
| Quyết định | Chặn / cho qua / xác thực thêm | Điều tra / bỏ qua / lập báo cáo giao dịch đáng ngờ |
| Căn cứ pháp lý | Bảo vệ khách hàng | Luật Phòng, chống rửa tiền 2022 |
| Tín hiệu chính | Thiết bị lạ, IP lạ, hành vi lệch thói quen | Hình dạng dòng tiền: gom, tỏa, vòng, trung chuyển |
| Thước đo đúng | Tỷ lệ chặn đúng ở ngưỡng chặn, thiệt hại tránh được | Số vụ thật trong N cảnh báo, số đường dây bị lộ |
| Bộ dữ liệu phù hợp | PaySim, IEEE-CIS, Synthetic VN | **AMLWorld** |

Hồ sơ vòng 1 đặt vấn đề ở cột trái, còn dữ liệu chính và cách đo đang ở cột
phải. Đây là điểm hội đồng dễ hỏi nhất: *"Đề bài nói deepfake và chiếm quyền
Mobile Banking, sao kết quả lại đo trên rửa tiền?"*

**Đề nghị:** chốt AML làm bài toán chính. Lý do:

- Dữ liệu tốt nhất (AMLWorld) và kết quả tốt nhất (recall đường dây 91%) đều ở
  phía AML.
- QAPE (xếp lịch đội điều tra) chỉ có nghĩa trong AML. Bài toán chặn lừa đảo
  thời gian thực không có "đội điều tra xử lý theo ca".
- Money mule nằm ở giao điểm hai bài toán: tài khoản trung chuyển tiền lừa đảo
  chính là bước layering của rửa tiền. Đây là cầu nối để giữ được câu chuyện
  Việt Nam trong hồ sơ.

### 2.2 Bảy câu hỏi kinh doanh và thước đo tương ứng

| # | Câu hỏi của ngân hàng | Thước đo | Đã trả lời chưa |
|---|---|---|---|
| Q1 | Đội xem được N cảnh báo/ngày thì bắt được bao nhiêu vụ thật? | bắt@N, tỷ lệ cảnh báo đúng | **Có** — 1.843/2.000 trên HI-Medium val (GBDT) |
| Q2 | Có lộ ra được cả đường dây, hay chỉ từng giao dịch lẻ? | recall đường dây @N | **Có** — 75,6% đường dây @500, 91,1% @2.000 trên HI-Small val (lần chạy khác: 82–87%) |
| Q3 | Có tốt hơn hệ thống luật đang dùng không? | bắt@N so với bộ luật AML | **Chưa** — chưa có bộ luật AML thật |
| Q4 | Mang sang ngân hàng khác hoặc giai đoạn khác có còn dùng được? | độ nâng khi chuyển giao | **Một phần** — HI→LI 299×; Medium chưa thử |
| Q5 | Chia việc cho đội điều tra thế nào là tối ưu? | tỷ lệ so với nghiệm tối ưu | **Có** — DP đạt 99,87% trong 0,13 giây; QAOA không có lợi thế |
| Q6 | Chuyên viên có hiểu vì sao bị cảnh báo không? | đánh giá định tính | **Chưa đo** — có XAI nhưng chưa thử với người dùng |
| Q7 | Chặn được bao nhiêu tiền? | % tiền trong top N (đã quy đổi USD) | **Không đo tin cậy được** — 10 giao dịch giữ 90,9% tiền |

### 2.3 Câu trả lời cho Q1, dịch ra ngôn ngữ vận hành

HI-Medium, tập val: 4.603.545 giao dịch trong vũ trụ chấm điểm, 5.489 vụ rửa
tiền, tỷ lệ nền 0,119%.

| Cách chọn 2.000 cảnh báo | Vụ thật bắt được | Cảnh báo sai | Phủ được bao nhiêu % vụ |
|---|---|---|---|
| Ngẫu nhiên | 2 | 99,9% | 0,04% |
| Số tiền lớn nhất (gần với luật ngưỡng) | 16 | 99,2% | 0,3% |
| DRRE (checkpoint HI-Small) | 996 | 50,2% | 18,1% |
| Cây quyết định | 1.789 | 10,6% | 32,6% |
| **GBDT cân bằng, 20 đặc trưng** | **1.843** | **7,9%** | **33,6%** |

Hai lưu ý trước khi trình bày:

- 92% cảnh báo đúng nghe quá đẹp so với thực tế ngành, nơi đa số cảnh báo AML là
  sai. Lý do chính là AMLWorld là dữ liệu tổng hợp, và kênh ACH chứa ~86% vụ
  rửa tiền (mục 3.2). Con số này chứng minh phương pháp, không hứa hẹn hiệu quả
  ở ngân hàng thật.
- 2.000 cảnh báo chỉ phủ 34% số vụ của 2,4 ngày. Muốn phủ hết cần ngân sách lớn
  hơn. Đây chính là đầu vào cho QAPE.

---

## 3. Đọc kỹ mô tả gốc các bộ dữ liệu trên Kaggle

### 3.1 IBM AMLWorld — bài toán bộ dữ liệu được tạo ra để giải

Theo mô tả của tác giả (Erik Altman, IBM) trên Kaggle và bài báo NeurIPS 2023
(arXiv 2306.16424):

- **Mục đích:** cung cấp dữ liệu có nhãn đầy đủ để huấn luyện và kiểm thử mô
  hình AML. Dữ liệu thật vừa bị hạn chế truy cập vừa không bao giờ có nhãn đầy
  đủ, vì phần lớn vụ rửa tiền thật không bị phát hiện.
- **Cách sinh:** một thế giới ảo gồm cá nhân, doanh nghiệp và ngân hàng giao
  dịch với nhau (mua hàng, trả lương, trả nợ…). Một phần nhỏ làm việc phi pháp
  (buôn lậu, cờ bạc, tống tiền) rồi tìm cách che giấu nguồn tiền.
- **Mô phỏng đủ ba giai đoạn rửa tiền:** sắp đặt (placement), phân tán
  (layering), hợp nhất (integration). Máy sinh lần theo đồng tiền bẩn qua nhiều
  chặng, nên gắn nhãn được cả giao dịch cách nguồn bẩn rất xa.
- **Điểm đặc biệt:** dữ liệu chứa toàn bộ hệ sinh thái nhiều ngân hàng, trong
  khi một ngân hàng thật chỉ thấy giao dịch của mình. Tác giả gợi ý học trên
  toàn cảnh rồi suy luận cho một ngân hàng.
- **HI / LI:** HI có tỷ lệ rửa tiền cao hơn, LI thấp hơn. Sáu bộ **độc lập**,
  bộ Small không phải tập con của bộ Medium.
- **Cách chia tập khuyến nghị:** theo thời gian, **60% / 20% / 20%**.
- **Thước đo trong bài báo:** F1 của lớp thiểu số (rửa tiền), không phải AUPRC.
- **File Patterns:** chỉ liệt kê các vụ thuộc 8 kiểu hình của AMLSim. Tác giả
  nói rõ **không phải mọi vụ rửa tiền đều theo 8 kiểu hình đó**.
- **Giao dịch sau khoảng thời gian chính toàn là rửa tiền:** tác giả đã ghi
  nhận trên Kaggle (thảo luận của Marco Pianta).

Hệ quả cho dự án:

| Phát hiện khi đọc mô tả | Hệ quả |
|---|---|
| Lỗi "giao dịch ngày muộn toàn là rửa tiền" đã được IBM công bố | Báo cáo 31/10 coi đây là lỗi 4 do nhóm **tự phát hiện**. Cần sửa thành "đã xác nhận độc lập một lỗi IBM đã ghi nhận", nếu không hội đồng có thể coi là nhận vơ |
| Bài báo dùng F1 lớp thiểu số và chia 60/20/20; dự án dùng AUPRC và chia 70/15/15 | Muốn so với số công bố (lỗ hổng 7.4 của báo cáo 31/10) phải tính thêm F1, và phải nói rõ hai cách chia khác nhau |
| Không có thiết bị, IP, số dư hay nạn nhân | AMLWorld không kiểm chứng được các kịch bản deepfake hay chiếm quyền trong hồ sơ vòng 1 |
| Số vụ rửa tiền trong bảng của bài báo (HI-Small 3,6K) lệch với Kaggle (5,1K) | Nhóm đếm lại từ file: **5.177**. Nên trích số tự đếm |

Kết quả công bố trong bài báo, để định vị kết quả của nhóm (F1 lớp thiểu số,
%, trung bình ± độ lệch chuẩn):

| Mô hình | HI-Small | LI-Small | HI-Medium | LI-Medium |
|---|---|---|---|---|
| GIN | 28,70 ± 1,13 | 7,90 ± 2,78 | 42,30 ± 0,44 | 3,86 ± 3,62 |
| PNA | 56,77 ± 2,41 | 16,45 ± 1,46 | 59,71 ± 1,91 | 27,73 ± 1,65 |
| LightGBM + đặc trưng đồ thị | 62,86 ± 0,25 | 20,83 ± 1,50 | 59,48 ± 0,15 | 20,85 ± 0,38 |
| **XGBoost + đặc trưng đồ thị** | **63,23 ± 0,17** | **27,30 ± 0,33** | **65,70 ± 0,26** | **28,16 ± 0,14** |

Điều đáng chú ý: **ngay trong bài báo gốc, mô hình cây + đặc trưng đồ thị cũng
thắng mạng nơ-ron đồ thị.** Kết quả "GBDT thắng DRRE" của nhóm khớp với kết luận
đã công bố, nên có thể trình bày như một kết quả nhất quán với tài liệu thay vì
một thất bại riêng.

### 3.2 PaySim

- **Bài toán:** phát hiện gian lận trên dịch vụ tiền di động ở một nước châu
  Phi. Kẻ gian chiếm quyền tài khoản khách, chuyển tiền sang tài khoản khác rồi
  rút tiền mặt.
- **Quy mô:** 744 bước (mỗi bước 1 giờ, 30 ngày), bằng 1/4 dữ liệu gốc.
- **Tác giả cảnh báo:** giao dịch gian lận bị hủy, nên **không được dùng** bốn
  cột số dư làm đặc trưng.
- **Liên quan dự án:** gần với kịch bản "chiếm quyền Mobile Banking" nhất, nhưng
  99,85% người gửi chỉ xuất hiện đúng một lần, nên không có lịch sử hành vi cho
  Behavior Memory học.

### 3.3 IEEE-CIS Fraud Detection

- **Bài toán:** dự đoán xác suất một giao dịch thẻ trực tuyến (thương mại điện
  tử, Vesta) là gian lận. Cuộc thi chấm bằng ROC-AUC.
- **Bản đầy đủ:** 590.540 giao dịch huấn luyện, ~3,5% gian lận, gần 400 cột ẩn
  danh cộng bảng định danh thiết bị.
- **Bản dự án đang dùng:** 300.000 dòng đầu, **chỉ 11 cột**. Đã bỏ gần như toàn
  bộ cột giàu thông tin (V1–V339, định danh thiết bị) và không có người nhận.
  Kết quả trên bộ này chỉ phản ánh bản rút gọn.

---

## 4. Chi tiết từng bộ dữ liệu

### 4.1 AMLWorld — sáu file gốc (đếm lại 03/11/2026)

"Bình thường" là giao dịch có `Is Laundering = 0`, "bất thường" là `= 1`.

| Bộ | Số ngày chính | Giao dịch | Bình thường | **Rửa tiền** | Tỷ lệ | 1 vụ / N giao dịch | Tài khoản | Chủ thể |
|---|---|---|---|---|---|---|---|---|
| HI-Small | 10 | 5.078.345 | 5.073.168 | **5.177** | 0,102% | 981 | 515.080 | 166.207 |
| LI-Small | 10 | 6.924.049 | 6.920.484 | **3.565** | 0,051% | 1.942 | 705.903 | 224.931 |
| HI-Medium | 16 | 31.898.238 | 31.863.008 | **35.230** | 0,110% | 905 | 2.076.999 | 668.138 |
| LI-Medium | 16 | 31.251.483 | 31.235.442 | **16.041** | 0,051% | 1.948 | 2.032.061 | 654.133 |
| HI-Large | 97 | 179.702.229 | 179.476.683 | **225.546** | 0,126% | 797 | 2.099.541 | 670.231 |
| LI-Large | 97 | 176.066.557 | 175.965.953 | **100.604** | 0,057% | 1.750 | 2.054.390 | 656.221 |
| **Tổng** | | **430.920.901** | | **386.163** | | | | |

"Chủ thể" là số cá nhân/doanh nghiệp đứng tên tài khoản, lấy từ
`*_accounts.csv`: trung bình mỗi chủ thể có khoảng 3 tài khoản.

Báo cáo 31/10 ghi HI-Large "~127 triệu dòng" và tổng "~323,6 triệu". Đếm thật
là **179,7 triệu** và **430,9 triệu**. Dự án đang dùng 4 bộ đầu, tức 75,2 triệu
dòng (17% tổng số).

**Phân bố theo hình thức thanh toán** (HI-Medium, toàn bộ file):

| Hình thức | Giao dịch | Rửa tiền | Tỷ lệ trong kênh | % tổng số vụ |
|---|---|---|---|---|
| **ACH** | 3.868.410 | **30.746** | 0,795% | **87,3%** |
| Cheque | 12.280.058 | 2.220 | 0,018% | 6,3% |
| Credit Card | 8.777.816 | 1.354 | 0,015% | 3,8% |
| Cash | 3.217.531 | 666 | 0,021% | 1,9% |
| Bitcoin | 689.038 | 244 | 0,035% | 0,7% |
| Reinvestment | 1.945.611 | 0 | 0 | 0 |
| Wire | 1.119.774 | 0 | 0 | 0 |

Năm bộ còn lại cùng hình dạng: ACH chiếm 73–88% số vụ. Reinvestment **không có
vụ nào** trên cả sáu bộ. Wire không có vụ nào trên bốn bộ Small/Medium, nhưng
có **5 vụ ở HI-Large và 3 vụ ở LI-Large**. Vì vậy hai kênh này bị loại khỏi vũ
trụ chấm điểm (lỗi dữ liệu 2). Nếu sau này chạy bộ Large, quy tắc loại Wire phải
xem lại.

**Ba đặc điểm dữ liệu cần biết:**

| Đặc điểm | HI-Small | LI-Small | HI-Medium | LI-Medium | Xử lý |
|---|---|---|---|---|---|
| Giao dịch đổi tiền (tiền trả ≠ tiền nhận) | 72.170, **0 vụ** | 98.876, **0 vụ** | 485.144, **0 vụ** | 467.879, **0 vụ** | Loại `is_fx` khỏi đặc trưng vì rò rỉ ngược |
| Giao dịch sau khoảng thời gian chính | 1.108 (655 vụ, 59%) | 223 (134 vụ) | 6.987 (4.084 vụ, 58%) | 1.370 (771 vụ) | Tách ra tập `drift` bằng bộ lọc ngày lành |
| Tự chuyển cho chính mình | 591.212 (11 vụ) | 804.477 (3 vụ) | 2.561.860 (72 vụ) | 2.499.351 (10 vụ) | Giữ nguyên |

**Tám kiểu hình rửa tiền** (từ file `*_Patterns.txt`): đếm số đường dây, trong
ngoặc là số giao dịch.

| Kiểu hình | Hình dạng | HI-Small | LI-Small | HI-Medium | LI-Medium |
|---|---|---|---|---|---|
| FAN-OUT | 1 tài khoản tỏa tiền ra nhiều tài khoản | 48 (342) | 19 (149) | 345 (2.128) | 69 (489) |
| FAN-IN | Nhiều tài khoản dồn tiền về 1 tài khoản | 40 (318) | 12 (73) | 355 (2.315) | 47 (329) |
| GATHER-SCATTER | Gom vào rồi tỏa ra tại cùng 1 tài khoản | 51 (716) | 12 (150) | 322 (4.289) | 43 (541) |
| SCATTER-GATHER | Tỏa ra nhiều trung gian rồi gom về 1 tài khoản khác | 44 (626) | 13 (182) | 331 (3.988) | 63 (886) |
| CYCLE | Tiền đi một vòng rồi quay về tài khoản gốc | 54 (287) | 12 (83) | 367 (2.235) | 46 (283) |
| RANDOM | Đi ngẫu nhiên qua các tài khoản, không quay về | 41 (191) | 15 (77) | 331 (1.667) | 49 (278) |
| BIPARTITE | Nhóm tài khoản nguồn chuyển sang nhóm tài khoản đích | 49 (263) | 16 (129) | 369 (2.135) | 72 (488) |
| STACK | Hai lớp bipartite chồng nhau | 43 (466) | 18 (180) | 336 (3.986) | 67 (615) |
| **Tổng theo kiểu hình** | | **370 (3.209)** | **117 (1.023)** | **2.756 (22.743)** | **456 (3.909)** |
| **% vụ rửa tiền thuộc kiểu hình** | | **62%** | **29%** | **65%** | **24%** |

Bộ Large: HI-Large có 16.467 đường dây (137.936 giao dịch, 61% số vụ), LI-Large
có 2.218 đường dây (19.461 giao dịch, 19% số vụ). Con số 19.461 khớp đúng bài
báo gốc. Bộ Large kéo dài từ 01/08/2022 tới 12/01/2023. Phần sau 05/11 có
40.719 dòng (HI) và 6.284 dòng (LI), khoảng 60% là rửa tiền: cùng hiện tượng
ngày muộn như bộ Small và Medium.

Hệ quả quan trọng nhất của bảng này: **ở bộ LI, ~75% vụ rửa tiền không thuộc kiểu
hình nào.** Đó là các khoản "hợp nhất" ngụy trang thành trả lương hoặc mua hàng,
nên không có hình dạng đồ thị để bắt. Điều này giải thích vì sao bài báo gốc chỉ
đạt F1 ~28% trên LI, so với ~65% trên HI.

### 4.2 AMLWorld — sau tiền xử lý (các tập dùng để huấn luyện và đo)

Chia theo thời gian 70/15/15. "Vũ trụ" là các dòng được chấm điểm (đã bỏ Wire
và Reinvestment). Mọi vụ rửa tiền đều nằm trong vũ trụ.

| Bộ | Tập | Khoảng thời gian (2022) | Giao dịch | Trong vũ trụ | Rửa tiền | Tỷ lệ nền trong vũ trụ |
|---|---|---|---|---|---|---|
| HI-Small | train | 01/09 – 07/09 14:52 | 3.554.065 | 2.956.734 | 2.856 | 0,097% |
| | val | 07/09 – 09/09 03:14 | 761.585 | 733.086 | 760 | 0,104% |
| | test | 09/09 – 10/09 | 761.587 | 734.506 | 906 | 0,123% |
| | drift | 11/09 – 18/09 | 1.108 | 1.108 | 655 | 59% |
| LI-Small | train | 01/09 – 07/09 14:48 | 4.846.678 | 4.041.621 | 2.231 | 0,055% |
| | val | 07/09 – 09/09 03:13 | 1.038.573 | 1.000.707 | 583 | 0,058% |
| | test | 09/09 – 10/09 | 1.038.575 | 1.002.473 | 617 | 0,062% |
| | drift | 11/09 – 17/09 | 223 | 223 | 134 | 60% |
| HI-Medium | train | 01/09 – 12/09 14:19 | 22.323.875 | 19.620.681 | 21.077 | 0,107% |
| | val | 12/09 – 15/09 01:24 | 4.783.687 | 4.603.545 | 5.489 | 0,119% |
| | test | 15/09 – 16/09 | 4.783.689 | 4.601.640 | 4.580 | 0,100% |
| | drift | 17/09 – 28/09 | 6.987 | 6.987 | 4.084 | 58% |
| LI-Medium | train | 01/09 – 12/09 14:14 | 21.875.079 | 19.220.458 | 10.648 | 0,055% |
| | val | 12/09 – 15/09 01:21 | 4.687.516 | 4.508.830 | 2.465 | 0,055% |
| | test | 15/09 – 16/09 | 4.687.518 | 4.507.492 | 2.157 | 0,048% |
| | drift | 17/09 – 27/09 | 1.370 | 1.370 | 771 | 56% |

Số vụ không thuộc kiểu hình nào trong tập train: HI-Small 1.327/2.856 (46%),
HI-Medium 8.681/21.077 (41%), LI-Small 1.679/2.231 (75%), LI-Medium
8.621/10.648 (81%). **Tập test chưa chạm lần nào.**

### 4.3 PaySim

| Loại giao dịch | Giao dịch | Gian lận | Tỷ lệ |
|---|---|---|---|
| CASH_OUT | 2.237.500 | 4.116 | 0,18% |
| TRANSFER | 532.909 | 4.097 | 0,77% |
| PAYMENT | 2.151.495 | 0 | 0 |
| CASH_IN | 1.399.284 | 0 | 0 |
| DEBIT | 41.432 | 0 | 0 |
| **Tổng** | **6.362.620** | **8.213** | **0,129%** |

- Bình thường: 6.354.407 giao dịch. Gian lận chỉ xuất hiện ở 2/5 loại giao dịch.
- Cờ `isFlaggedFraud` (luật "chuyển trên 200.000") chỉ bật **16 lần** trên 8.213
  vụ. Đây là minh họa sẵn có cho việc luật ngưỡng cố định bỏ sót.
- 99,85% người gửi chỉ có 1 giao dịch.
- Lỗi đã biết: giờ 4–8 có 100% là gian lận; 54% nhãn dương dồn trong 6,7% số
  dòng.
- Bản tiền xử lý đầy đủ: train 4.156.259 (2.609 gian lận), val 890.626 (528),
  test 890.628 (620), phần đuôi bị lọc 425.107 (4.456).

### 4.4 IEEE-CIS (bản 300.000 dòng)

| Sản phẩm (ProductCD) | Giao dịch | Gian lận | Tỷ lệ |
|---|---|---|---|
| W | 206.740 | 4.134 | 2,0% |
| C | 35.528 | 3.792 | 10,7% |
| R | 26.605 | 830 | 3,1% |
| H | 25.479 | 1.023 | 4,0% |
| S | 5.648 | 320 | 5,7% |
| **Tổng** | **300.000** | **10.099** | **3,37%** |

- Bình thường: 289.901 giao dịch; ~86 ngày.
- Chia tập: train 210.000 (6.340 gian lận), val 45.000 (2.002), test 45.000
  (1.757).
- Không có người nhận, nên nhóm đặc trưng đồ thị không dùng được.

### 4.5 Dữ liệu tự sinh

| Bộ | Giao dịch | Bình thường | Bất thường | Thành phần bất thường |
|---|---|---|---|---|
| Synthetic VN | 5.273 | 5.226 | 47 (0,89%) | chuỗi mule 18, chiếm quyền 17, lừa đầu tư/tình cảm 12 |
| Bộ tăng cường | 3.675 | 450 | 3.225 (dòng thuộc kịch bản) | chia nhỏ giao dịch 1.800, chuỗi mule 675, chiếm quyền 600, lừa đầu tư/tình cảm 600 |

- Cả hai bộ có thiết bị và IP **tự sinh**, đã gắn cờ `is_synthetic_field`.
- Bộ tăng cường dùng đơn vị tiền PaySim và qua kiểm định đối kháng: AUC 0,56 so
  với gian lận thật (ngưỡng đạt 0,4–0,6). Nghĩa là gần như không phân biệt được
  với gian lận thật về số tiền.

### 4.6 Từng cột gốc của AMLWorld được dùng thế nào

| Cột gốc | Đưa vào | Dùng làm gì |
|---|---|---|
| Timestamp | `timestamp` | Trục thời gian: chia tập, tính đặc trưng nhân quả |
| From Bank, To Bank | — | **Chưa dùng.** Có thể khai thác (giao dịch liên ngân hàng) |
| Account (gửi) | `account_id` | Khóa lịch sử hành vi, đỉnh đồ thị |
| Account (nhận) | `counterparty_id` | Đỉnh thứ hai của cạnh đồ thị |
| Amount Paid | `amount` | Số tiền, **theo nguyên tệ, chưa quy đổi USD** |
| Payment Currency | `currency` | Chưa thành đặc trưng; cần cho quy đổi USD |
| Amount Received, Receiving Currency | `is_fx` | Chỉ để dựng bảng tỷ giá và báo cáo lỗi rò rỉ |
| Payment Format | `channel` | Kênh thanh toán, đặc trưng mạnh nhất |
| Is Laundering | `label` | Nhãn |
| File Patterns | `laundering_pattern` | **Chỉ để đánh giá** recall theo kiểu hình, không cho mô hình thấy |
| accounts.csv | — | Đã thử ghép chủ thể: không cải thiện, đã loại |

### 4.7 Hai mươi đặc trưng mô hình đang dùng (AMLWorld)

Mọi đặc trưng đều **nhân quả**: chỉ dùng giao dịch xảy ra **trước** giao dịch
đang chấm, giống hệt điều kiện của hệ thống thật.

| # | Đặc trưng | Nhóm | Đo cái gì | Phục vụ phát hiện |
|---|---|---|---|---|
| 1 | `amount_norm` | Gốc | Số tiền đã chuẩn hóa | Giao dịch lớn bất thường |
| 2 | `channel_encoded` | Gốc | Hình thức thanh toán | ACH chứa ~86% vụ (mạnh nhưng là đặc thù máy sinh) |
| 3 | `log_n_prior` | Lịch sử tài khoản | Số giao dịch đã có trước đó | Tài khoản mới mở hoặc ít hoạt động (mule mới) |
| 4 | `log_gap` | Lịch sử tài khoản | Thời gian từ giao dịch trước | Dồn dập bất thường (velocity) |
| 5 | `z_amount` | Lịch sử tài khoản | Số tiền lệch bao nhiêu độ lệch chuẩn so với thói quen **của chính tài khoản** | Ngưỡng cá nhân hóa, đúng lời hứa trong hồ sơ |
| 6 | `log_amount_ratio` | Lịch sử tài khoản | Tỷ lệ so với mức trung bình của chính tài khoản | Như trên, ổn định hơn khi ít lịch sử |
| 7 | `new_counterparty` | Đối tác | Lần đầu chuyển cho người này | Quan hệ mới: FAN-OUT, SCATTER |
| 8 | `log_n_distinct_cp` | Đối tác | Đã chuyển cho bao nhiêu người khác nhau | Tỏa tiền: FAN-OUT |
| 9 | `log_n_prior_as_cp` | Đối tác | Người nhận đã nhận bao nhiêu lần | Tài khoản thu gom |
| 10 | `log_n_distinct_senders` | Đối tác | Người nhận đã nhận từ bao nhiêu người khác nhau | Dồn tiền: FAN-IN, GATHER |
| 11 | `is_round_amount` | Hình dạng số tiền | Số tiền chẵn trăm | Chia nhỏ để lách ngưỡng (structuring) |
| 12–13 | `hour_sin`, `hour_cos` | Đồng hồ | Giờ trong ngày (dạng tuần hoàn) | Giao dịch giờ bất thường |
| 14 | `dow` | Đồng hồ | Ngày trong tuần | Nhịp tuần |
| 15 | `log_amt_out_prior` | Dòng tiền | Tổng tiền tài khoản đã chuyển đi | Quy mô dòng tiền ra |
| 16 | `log_amt_in_prior` | Dòng tiền | Tổng tiền tài khoản đã nhận | Quy mô dòng tiền vào (nhìn được chiều nhận) |
| 17 | `log_n_in_prior` | Dòng tiền | Số lần đã nhận | FAN-IN nhìn từ chính tài khoản |
| 18 | `passthrough_ratio` | Trung chuyển | min(thu, chi) / max(thu, chi); 1 = nhận bao nhiêu chuyển bấy nhiêu | **Money mule**, layering, STACK |
| 19 | `flow_imbalance` | Trung chuyển | (thu − chi) / (thu + chi) | Tài khoản gom hay tài khoản xả |
| 20 | `log_cp_amt_in_prior` | Dòng tiền đối tác | Tổng tiền người nhận đã nhận trước đó | Người nhận là điểm gom lớn |

**Cố ý không đưa vào mô hình:**

| Cột | Lý do |
|---|---|
| `is_fx` (đổi tiền) | 0 vụ trên mọi bộ → mô hình học ngược chiều thực tế |
| Số dư tài khoản (PaySim) | Tác giả PaySim cấm dùng; rò rỉ nhãn |
| Thiết bị, IP | Tự sinh, không có trong dữ liệu gốc |
| `laundering_pattern` | Chính là nhãn |
| Đồng hồ trên PaySim | Giờ 4–8 có 100% gian lận |

**Ứng viên đã kiểm định nhưng chưa vào mã nguồn:** `log_amount_usd` (số tiền quy
đổi USD) cho +0,0324 AUPRC, KTC 95% [+0,0211; +0,0447]. Cần dựng lại bảng tỷ giá
chỉ từ tập train trước khi trích dẫn.

---

## 5. Lộ trình

### 5.1 Đã làm

| Thời gian | Việc | Mốc đạt được |
|---|---|---|
| 07/2026 | Tài liệu kiến trúc DRRE + Quantum, kế hoạch 3 vai trò, hồ sơ vòng 1 | Nộp vòng 1 |
| 21–27/07 | Dựng repo, PaySim + Synthetic VN, bản demo đầu | Pipeline đầu tiên chạy được |
| 18–22/09 | Thêm IEEE-CIS, trang giám sát và ưu tiên cảnh báo trên demo | Demo có dashboard |
| 07–10/10 | Manifest thí nghiệm, vector hóa (nhanh 29–42×), bộ tăng cường, báo cáo 10/10 | **Phát hiện kết quả PaySim không tái lập (dải 10× giữa seed)**; commit cuối cùng 11/10 |
| 17/10 | 14 đặc trưng nhân quả; đo lại IEEE và PaySim | DRRE vẫn thua GBDT trên dữ liệu thật |
| 18–19/10 | Chuyển sang IBM AMLWorld, tiền xử lý HI-Small | Đổi dữ liệu chính sang AML |
| 23–24/10 | 6 đặc trưng dòng tiền (20 đặc trưng); DRRE trên HI-Small (11,5 giờ) | GBDT 0,47–0,50 so với DRRE 0,24; recall đường dây 82–91% |
| 28/10 | LI-Small, kiểm thử chuyển giao HI→LI, đo tập drift | Độ nâng chuyển giao 299× |
| 31/10 | Báo cáo kết quả; chuyển sang máy 32 GB; tiền xử lý HI/LI-Medium; đo HI-Medium | GBDT 0,5125 trên HI-Medium (lift ~430×), giữ mức của Small |

### 5.2 Đang làm

| Việc | Trạng thái |
|---|---|
| Huấn luyện DRRE trên HI-Medium (5 epoch, mỗi phiên 1–2 epoch) | Xong 2/5: val AUPRC 0,3207 → **0,3717**, vẫn đang lên |
| Thống kê lại toàn bộ dữ liệu (tài liệu này) | Xong, cả 6 bộ AMLWorld |

### 5.3 Sẽ làm — đề xuất cho 7 ngày tới

Giả định chung kết **10/11/2026** theo kế hoạch triển khai. **Cần xác nhận lại
lịch chính thức của Ban tổ chức** và kết quả vòng bán kết (25/10) trước khi
chốt.

| Ngày | Việc | Mốc phải đạt | Ai quyết |
|---|---|---|---|
| 03–04/11 | Chốt định vị AML; trả lời 4 câu hỏi của báo cáo 31/10 | Một câu phát biểu bài toán thống nhất | Anh Dũng |
| 04/11 | Commit toàn bộ 46 file đang treo | Mã nguồn an toàn, truy nguyên được | Tuấn |
| 04–05/11 | DRRE HI-Medium epoch 3–5 (cắm sạc, 2 phiên ~4 giờ + ~2 giờ) | Có số DRRE cuối cùng trên Medium | — |
| 04/11 | Bảng tỷ giá chỉ từ train + `log_amount_usd` vào mã nguồn | Số +0,0324 được phép trích dẫn | — |
| 05/11 | Tính F1 lớp thiểu số cho GBDT trên HI/LI-Small, HI/LI-Medium | **Đặt kết quả cạnh số của bài báo IBM** | — |
| 05/11 | Bộ luật AML đơn giản (ngưỡng số tiền, chia nhỏ, tỏa/gom trong 24 giờ) | **Trả lời được Q3: hơn luật bao nhiêu** | — |
| 06/11 | Chuyển giao HI→LI trên Medium | Củng cố Q4 | — |
| 06/11 | **Chạy tập test một lần duy nhất** với cấu hình đã khóa | Bộ số liệu chính thức | Anh Dũng duyệt cấu hình trước |
| 07–08/11 | Slide, kịch bản demo, video dự phòng; sửa lại cách nêu lỗi dữ liệu 4 | Bộ trình bày hoàn chỉnh | Cả nhóm |
| 08–09/11 | Diễn tập 2 lần có bấm giờ; bộ câu hỏi phản biện | Mỗi người tự trả lời được phần mình | Cả nhóm |
| 10/11 | Chung kết | — | — |

**Không nên làm trong 7 ngày này:** sửa kiến trúc DRRE (8 giờ mỗi lần thử), mở
rộng lượng tử (giới hạn hàm mũ), chạy bộ Large (tiền xử lý ~180 triệu dòng cần
viết lại theo luồng).

---

## Phụ lục — nguồn

- Mô tả bộ dữ liệu IBM AMLWorld trên Kaggle:
  `kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml`
- Altman và cộng sự, *Realistic Synthetic Financial Transactions for
  Anti-Money Laundering Models*, NeurIPS 2023 Datasets & Benchmarks, arXiv
  2306.16424.
- Mô tả PaySim trên Kaggle: `kaggle.com/datasets/ealaxi/paysim1`.
- Cuộc thi IEEE-CIS Fraud Detection: `kaggle.com/competitions/ieee-fraud-detection`.
- Số liệu kết quả: `reports/eval/20261031-100336-evaluate_amlworld_val.json`
  (HI-Medium) và báo cáo 31/10/2026.
