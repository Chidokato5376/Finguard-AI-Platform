# FinGuard AI — Kịch bản trình bày chung kết (10/11/2026)

Nhóm CMET (AQ2026-111) · bản nháp 03/11/2026

Tài liệu này là khung trình bày theo đúng 5 phần, kèm nội dung nói, con số và
câu hỏi phản biện dự kiến cho từng phần.

- **[ĐIỀN SAU]**: con số sẽ có sau các thí nghiệm của tuần này. Không trình bày
  ô trống; nếu chưa có số thì bỏ dòng đó.
- **⚠**: điểm phải nói đúng cách, nếu không hội đồng dễ bắt lỗi.

Số liệu dữ liệu chi tiết nằm ở
[phan-tich-du-lieu-va-lo-trinh-2026-11-03.md](phan-tich-du-lieu-va-lo-trinh-2026-11-03.md).

---

## Khung thời gian trình bày

Giả định 15 phút trình bày. Nếu Ban tổ chức cho thời lượng khác thì co giãn
theo tỷ lệ.

| Phần | Thời lượng | Số slide | Thông điệp cần đọng lại |
|---|---|---|---|
| 1. Bài toán | 2,5 phút | 3 | Rửa tiền là bài toán **mạng lưới**, không phải từng giao dịch lẻ |
| 2. Dữ liệu | 3 phút | 4 | Chọn AMLWorld vì là bộ duy nhất có đủ hai đầu giao dịch, nhãn đầy đủ và mốc so sánh công bố |
| 3. Ý tưởng và mô hình | 4 phút | 4 | Ba nguồn tín hiệu: thói quen, thời điểm, quan hệ mạng lưới; lượng tử dùng cho bài toán phân bổ |
| 4. Thang đo và kết quả | 4 phút | 4 | 2.000 cảnh báo đầu: hơn 9/10 là vụ thật; có so sánh với 11 phương pháp |
| 5. Kết luận | 1,5 phút | 2 | Biết rõ cái gì chạy được, cái gì không, và đi tiếp thế nào |

---

## Phần 1 — Bài toán (2,5 phút)

### Slide 1.1 — Bối cảnh: rửa tiền ở Việt Nam là có thật và có tổ chức

**Vụ Mr Pips (Phó Đức Nam)** — TAND TP Hà Nội mở phiên sơ thẩm ngày 26/10/2026:

| Chi tiết | Con số | Ý nghĩa cho bài toán |
|---|---|---|
| Số bị cáo | 188 | Không phải hành vi cá nhân, mà là **mạng lưới có phân vai** |
| Bị hại | hơn 900 người, gần 1.560 tỷ đồng | Quy mô lớn |
| Tài khoản liên kết sàn giả | **66 tài khoản** ngân hàng và trung gian thanh toán | Tiền đi qua **nhiều nút**, không qua một tài khoản |
| Công ty "ma" | lập để mở tài khoản, thuê số điện thoại | Lớp che giấu (layering) |
| Rửa tiền | hơn 560 tỷ đồng mua 32 bất động sản, vàng, USD, SGD | Đủ ba giai đoạn rửa tiền |
| Trung gian thanh toán | Nguyễn Hòa Bình (Shark Bình) bị truy tố tội rửa tiền, cáo trạng nêu hơn 318 tỷ đồng | Rửa tiền đi qua **cả hạ tầng thanh toán hợp pháp** |

Có thể nhắc thêm một câu về **Vạn Thịnh Phát giai đoạn 2**: cáo buộc rửa tiền
hơn 445.000 tỷ đồng. Vụ này cho thấy quy mô có thể lớn tới mức nào.

⚠ **Vụ Mr Pips đang xét xử, chưa có bản án.** Phải nói "bị truy tố", "theo cáo
trạng", không nói "đã bị kết án". Kiểm tra lại tin mới nhất trước ngày 09/11, vì
phiên tòa dự kiến kéo dài khoảng 15 ngày từ 26/10 nên có thể tuyên án sát ngày
thi. Hồ sơ vòng 1 ghi "738 vụ việc, hơn 1.300 tỷ đồng" là số cũ, nên thay bằng
số trong cáo trạng.

### Slide 1.2 — Vì sao cách làm hiện tại không bắt được

- Luật cố định chấm **từng giao dịch riêng lẻ**, ví dụ "trên X triệu thì cảnh
  báo".
- Kẻ rửa tiền **chia nhỏ** số tiền, **trải qua nhiều tài khoản** và **xen giữa
  giao dịch bình thường**, nên từng giao dịch đều trông hợp lệ.
- Bằng chứng từ chính dữ liệu của nhóm: xếp 2.000 giao dịch lớn nhất lên đầu
  chỉ bắt được **16/5.489** vụ (HI-Medium, tập val).

### Slide 1.3 — Bài toán nhóm đặt ra

> **Phát hiện các dòng tiền bất thường đi qua mạng lưới tài khoản,** bằng cách kết
> hợp ba nguồn tín hiệu cho mỗi giao dịch:
>
> 1. **Thói quen giao dịch** của chính tài khoản: giao dịch này lệch bao nhiêu so
>    với lịch sử của nó.
> 2. **Thời điểm và nhịp giao dịch**: giờ, ngày, khoảng cách tới giao dịch trước,
>    độ dồn dập.
> 3. **Quan hệ mạng lưới**: tài khoản này nhận từ ai, chuyển cho ai, có phải nút
>    trung chuyển (nhận bao nhiêu chuyển đi bấy nhiêu) hay không.

**Yếu tố thứ ba bạn hỏi chính là quan hệ mạng lưới.** DRRE có đúng ba nhánh,
khớp với ba nguồn tín hiệu trên:

| Nguồn tín hiệu | Nhánh DRRE | Tên trong đặc tả |
|---|---|---|
| Bản thân giao dịch: số tiền, kênh, thời điểm | Nhánh giao dịch | `x_t` |
| Thói quen | Behavior Memory (GRU) + Context-aware Attention | `h̃_t` |
| **Quan hệ mạng lưới** | **Graph-based Relationship Encoder (GraphSAGE)** | `g_t` |

Một cổng **Adaptive Risk Fusion** tự học xem với mỗi giao dịch nên tin vào nguồn
nào nhiều hơn. Ví dụ: tài khoản mới chưa có thói quen thì dồn trọng số sang mạng
lưới.

**Đầu vào:** chuỗi giao dịch có thời gian, người gửi, người nhận, số tiền, kênh.

**Đầu ra:**
- Điểm rủi ro 0–100 cho từng giao dịch.
- Danh sách cảnh báo xếp hạng.
- Lý do cảnh báo.
- Đề xuất đội điều tra xử lý cảnh báo nào trong ca.

**Câu hỏi kinh doanh:** *đội điều tra chỉ xem được N cảnh báo/ngày, vậy chọn N
cảnh báo nào để bắt được nhiều vụ rửa tiền nhất?*

⚠ Hồ sơ vòng 1 nói về lừa đảo (deepfake, chiếm quyền Mobile Banking). Nên nói một
câu nối: *"Tiền lừa đảo như vụ Mr Pips phải đi qua các tài khoản mule và công ty
ma để được rửa; FinGuard tập trung vào khâu đó: phát hiện dòng tiền qua mạng
lưới."*

**Phản biện dự kiến:**

- *"Sao không làm phát hiện lừa đảo như hồ sơ vòng 1?"*
  Trả lời: lừa đảo và rửa tiền nối nhau ở khâu mule. Dữ liệu công khai có nhãn
  đầy đủ cho cả mạng lưới chỉ có ở AML. Dữ liệu ngân hàng Việt Nam không công
  khai được (Nghị định 13/2023).

---

## Phần 2 — Dữ liệu (3 phút)

### Slide 2.1 — Bộ dữ liệu: IBM AMLWorld

**Nguồn:** IBM công bố trên Kaggle; bài báo tại NeurIPS 2023 (Altman và cộng sự).
Dữ liệu sinh từ một thế giới ảo gồm cá nhân, doanh nghiệp và ngân hàng; một
phần nhỏ phạm tội rồi rửa tiền qua đủ 3 giai đoạn: sắp đặt, phân tán, hợp nhất.

**Các cột trong file giao dịch `*_Trans.csv` (11 cột):**

| Cột | Ví dụ | Ý nghĩa | Nhóm dùng vào việc gì |
|---|---|---|---|
| Timestamp | 2022/09/01 00:20 | Ngày giờ giao dịch (tới phút) | Trục thời gian, chia tập, đặc trưng nhịp |
| From Bank | 010 | Mã ngân hàng gửi | Chưa dùng |
| Account | 8000EBD30 | Tài khoản gửi | Nút nguồn của đồ thị, khóa lịch sử thói quen |
| To Bank | 001 | Mã ngân hàng nhận | Chưa dùng |
| Account | 8000F5340 | Tài khoản nhận | Nút đích của đồ thị |
| Amount Received | 3697.34 | Số tiền bên nhận nhận được | Dựng bảng tỷ giá |
| Receiving Currency | US Dollar | Loại tiền bên nhận | Như trên |
| Amount Paid | 3697.34 | Số tiền bên gửi trả | **Số tiền** dùng cho mô hình |
| Payment Currency | US Dollar | Loại tiền bên gửi (15 loại) | Quy đổi USD |
| Payment Format | ACH / Cheque / Credit Card / Cash / Bitcoin / Wire / Reinvestment | Hình thức thanh toán | Đặc trưng kênh |
| Is Laundering | 0 / 1 | **Nhãn**: 1 = rửa tiền | Mục tiêu học |

**Hai file đi kèm:**

| File | Nội dung |
|---|---|
| `*_Patterns.txt` | Danh sách đường dây rửa tiền theo 8 kiểu hình (FAN-OUT, FAN-IN, GATHER-SCATTER, SCATTER-GATHER, CYCLE, RANDOM, BIPARTITE, STACK), mỗi đường dây kèm các giao dịch của nó. **Chỉ dùng để đánh giá, mô hình không được thấy** |
| `*_accounts.csv` | Bank Name, Bank ID, Account Number, Entity ID, Entity Name: tài khoản nào thuộc chủ thể nào. Đã thử ghép, không cải thiện |

### Slide 2.2 — Quy mô và khác biệt giữa bộ High và Low

| Bộ | Giao dịch | Bình thường | Rửa tiền | Tỷ lệ | Tài khoản |
|---|---|---|---|---|---|
| HI-Small | 5.078.345 | 5.073.168 | 5.177 | 1/981 | 515.080 |
| LI-Small | 6.924.049 | 6.920.484 | 3.565 | 1/1.942 | 705.903 |
| **HI-Medium** | **31.898.238** | 31.863.008 | **35.230** | 1/905 | 2.076.999 |
| LI-Medium | 31.251.483 | 31.235.442 | 16.041 | 1/1.948 | 2.032.061 |
| HI-Large | 179.702.229 | 179.476.683 | 225.546 | 1/797 | 2.099.541 |
| LI-Large | 176.066.557 | 175.965.953 | 100.604 | 1/1.750 | 2.054.390 |

Tổng cộng 430,9 triệu giao dịch. Nhóm đã dùng 4 bộ đầu (75,2 triệu giao dịch).

**High (HI) khác Low (LI) ở ba điểm:**

| | HI — High Illicit | LI — Low Illicit |
|---|---|---|
| Tỷ lệ rửa tiền | ~1/900 | ~1/1.900 (hiếm hơn khoảng 2 lần) |
| % vụ thuộc 8 kiểu hình mạng lưới | **61–65%** | **19–29%** |
| Phần còn lại | Ít | Phần lớn là khoản "hợp nhất" ngụy trang thành trả lương, mua hàng, nên không có hình dạng mạng lưới |
| Độ khó | Dễ hơn | Khó hơn, gần thực tế hơn |

Hai nhóm dùng **cùng cấu trúc cột, cùng máy sinh**, nhưng là **các thế giới độc
lập**: bộ nhỏ không phải tập con của bộ lớn. Vì vậy học trên HI rồi thử trên LI
là phép thử khả năng chuyển giao thật.

### Slide 2.3 — Vì sao chọn AMLWorld để huấn luyện

| Lý do | Giải thích |
|---|---|
| Có **cả hai đầu** giao dịch | Dựng được đồ thị tài khoản. IEEE-CIS không có người nhận; PaySim 99,85% người gửi chỉ xuất hiện 1 lần |
| **Nhãn đầy đủ** | Dữ liệu thật không bao giờ có nhãn đầy đủ vì phần lớn vụ rửa tiền không bị phát hiện. Máy sinh lần theo đồng tiền bẩn nên gắn nhãn được mọi chặng |
| Nhìn được **nhiều ngân hàng** | Đúng bản chất rửa tiền xuyên ngân hàng |
| Có **nhãn kiểu hình** | Đo được mô hình bắt được hình dạng mạng lưới nào |
| Có **mốc so sánh đã công bố** | Bài báo NeurIPS 2023 có kết quả của GNN, LightGBM, XGBoost |
| Đủ lớn để kiểm chứng | HI-Medium có 35.230 vụ, gấp 12 lần HI-Small |
| Không vướng dữ liệu cá nhân | Dữ liệu tổng hợp, chia sẻ được theo giấy phép CDLA |

Nhóm **đã thử và loại** PaySim và IEEE-CIS vì các lỗi đo được:
- PaySim: giờ 4–8 có 100% gian lận; gian lận chỉ nằm ở 2/5 loại giao dịch.
- IEEE-CIS: không có người nhận nên không dựng được mạng lưới.

### Slide 2.4 — Tiền xử lý và kiểm soát rò rỉ, quá khớp

**11 bước tiền xử lý** (`src/data/preprocessing.py`):

1. Đọc CSV và đưa về **schema chung**: người gửi, người nhận, thời điểm, số tiền,
   kênh, nhãn.
2. Gắn nhãn kiểu hình từ file Patterns, **chỉ để chấm điểm**.
3. Chuẩn hóa số tiền bằng log(1 + x). Phép này không học tham số nên không rò rỉ.
4. **Sắp theo thời gian một cách tường minh**, không tin thứ tự trong file. File
   HI-Medium có 5.146 dòng ngày muộn nằm lẫn ở đầu file.
5. Tính **14 đặc trưng nhân quả**: thói quen, đối tác, hình dạng số tiền, đồng hồ.
6. Tính **6 đặc trưng dòng tiền hai chiều**: tiền vào, tiền ra, tỷ lệ trung
   chuyển.
7. Đánh dấu **vũ trụ chấm điểm**: không chấm Wire và Reinvestment vì hai kênh này
   có 0 vụ trên bộ Small/Medium. Chỉ **che** khỏi phép chấm, không xóa, vì lịch
   sử và cạnh đồ thị vẫn cần các dòng này.
8. **Lọc ngày hỏng**: ngày có dưới 100.000 giao dịch được tách sang tập riêng
   `drift`, vì đó là các ngày máy sinh giao dịch thường đã tắt, còn ~58% là rửa
   tiền. IBM cũng đã ghi nhận hiện tượng này trên Kaggle.
9. **Chia theo thời gian** 70% / 15% / 15%.
10. Mã hóa kênh thanh toán, **chỉ học trên tập train**.
11. Ghi parquet kèm **dấu vân tay sha256** và manifest để mọi con số truy ngược
    được.

**Chia tập HI-Medium:**

| Tập | Thời gian (2022) | Giao dịch | Rửa tiền | Tỷ lệ nền |
|---|---|---|---|---|
| Train | 01/09 – 12/09 | 22.323.875 | 21.077 | 0,107% |
| Val | 12/09 – 15/09 | 4.783.687 | 5.489 | 0,119% |
| Test | 15/09 – 16/09 | 4.783.689 | 4.580 | 0,100% |
| Drift (ngày hỏng) | 17/09 – 28/09 | 6.987 | 4.084 | 58% |

**Kiểm soát rò rỉ dữ liệu (data leakage):**

| Nguy cơ | Biện pháp | Kiểm chứng |
|---|---|---|
| Dùng tương lai để đoán quá khứ | Chia theo thời gian, **không bao giờ chia ngẫu nhiên** | — |
| Đặc trưng nhìn thấy giao dịch sau | Mọi đặc trưng chỉ dùng giao dịch **trước** (cộng dồn trừ giá trị hiện tại) | Test tự động thêm dòng tương lai và đòi giá trị cũ không đổi |
| Mã hóa học trên cả val/test | Fit **chỉ trên train** | — |
| Cột chứa đáp án | Không dùng file Patterns, số dư hay `is_fx` (đổi tiền có 0 vụ, mô hình sẽ học ngược thực tế) | Đo được trên cả 6 bộ |
| Lặp đi lặp lại trên tập test | **Tập test chưa chạm lần nào**; chỉ chạy một lần cuối với cấu hình đã khóa | Manifest ghi lần chạy |

**Kiểm soát quá khớp (overfitting):**

- Chọn checkpoint DRRE bằng **trung bình trượt** của AUPRC trên val, không lấy
  epoch đỉnh ngẫu nhiên.
- Mọi so sánh kèm **khoảng tin cậy 95% bằng bootstrap ghép cặp**.
- Quy tắc báo cáo: kết quả từ ≥3 seed thì nêu **trung vị kèm dải**.
- Kiểm tra ngoài phân phối: **học HI, thử LI**.
- Tập Small và Medium độc lập, nên giữ được mức trên Medium là bằng chứng không
  khớp riêng một bộ.

⚠ Hai điểm chưa sạch, nên tự nói trước khi bị hỏi:

1. Quy tắc bỏ Wire và Reinvestment được suy ra từ đếm nhãn trên **toàn bộ file**.
   Nó phản ánh một đặc điểm của máy sinh, nhưng về nguyên tắc vẫn nhìn vào nhãn.
2. Rất nhiều quyết định thiết kế đã dựa vào tập val, nên số trên val có thể hơi
   lạc quan. Đây là lý do số chính thức sẽ lấy từ tập test, chạy đúng một lần.

**Phản biện dự kiến:**

- *"Dữ liệu tổng hợp thì kết quả có ý nghĩa gì?"*
  Trả lời: chứng minh **phương pháp** trên bộ dữ liệu chuẩn có mốc so sánh công
  bố, không hứa hẹn con số tại ngân hàng thật. Bước tiếp theo là thử trên dữ liệu
  ngân hàng có kiểm soát (phần 5).
- *"Sao chia 70/15/15 mà IBM khuyến nghị 60/20/20?"*
  Trả lời: phải nói thật là nhóm chọn trước khi dùng AMLWorld. Để so trực tiếp với
  bài báo thì cần chạy lại GBDT theo 60/20/20 **[ĐIỀN SAU nếu kịp]**.

---

## Phần 3 — Ý tưởng hiện thực hóa (4 phút)

### Slide 3.1 — Kiến trúc tổng thể

```
Giao dịch → [Tiền xử lý: 20 đặc trưng nhân quả] → [Mô hình chấm điểm] → điểm rủi ro 0–100
                                                         │                    │
                                                 [Giải thích XAI]     [QAPE: chọn cảnh báo
                                                         │             cho đội điều tra trong ca]
                                                         └──────► Bảng điều khiển demo
```

### Slide 3.2 — Học máy: kỹ thuật gì, mô hình nào, vì sao

**Kỹ thuật 1 — Đặc trưng nhân quả theo ba nguồn tín hiệu.** Đây là phần mang lại
nhiều giá trị nhất.

| Nguồn | Đặc trưng | Vì sao cần |
|---|---|---|
| Thói quen | `log_n_prior`, `z_amount`, `log_amount_ratio` | Ngưỡng **riêng cho từng khách hàng**: 50 triệu là bình thường với doanh nghiệp nhưng bất thường với sinh viên |
| Thời điểm | `log_gap`, `hour_sin/cos`, `dow` | Mule thường giao dịch dồn dập, sai nhịp |
| Mạng lưới | `log_n_distinct_cp`, `log_n_distinct_senders`, `passthrough_ratio`, `flow_imbalance`… | Bắt **hình dạng**: tỏa ra, gom về, trung chuyển |

**Kỹ thuật 2 — Mô hình học sâu DRRE** (PyTorch, PyTorch Geometric):

| Thành phần | Kỹ thuật | Vì sao |
|---|---|---|
| Behavior Memory | GRU + bộ nhớ K = 4 khe | Nhớ thói quen của từng tài khoản qua thời gian |
| Context-aware Attention | Attention trên các khe nhớ | So giao dịch mới với các mẫu hành vi cũ; trọng số dùng luôn để giải thích |
| Graph Encoder | GraphSAGE 2 lớp | Lan thông tin giữa các tài khoản láng giềng; rửa tiền là cấu trúc đồ thị |
| Adaptive Fusion | Cổng softmax 3 nhánh | Tự chọn nguồn tin cậy cho từng giao dịch |
| Hàm mất mát | L_pred + 0,5·L_contrast + 0,3·L_BCE | Hai thành phần đầu tự giám sát nên dùng được dữ liệu không nhãn. L_BCE dùng nhãn, trọng số lớp dương ≈ 930 để bù mất cân bằng |
| Chống sụp biểu diễn | Mạng mục tiêu EMA (kiểu BYOL) | Tránh nghiệm tầm thường khi tự dự đoán chính mình |

**Kỹ thuật 3 — Mô hình cây tăng cường (GBDT)** (scikit-learn
HistGradientBoosting, 200 vòng, cân bằng lớp): nhận cùng 20 đặc trưng.

**Kỹ thuật 4 — Xử lý mất cân bằng 1/900:** cân bằng trọng số lớp. Riêng bước này
đưa AUPRC của GBDT từ **0,19 lên 0,44** (×2,25).

### Slide 3.3 — Tiến trình huấn luyện

| Giai đoạn | Việc | Kết quả |
|---|---|---|
| 07–10/2026 | PaySim, IEEE-CIS | Phát hiện lỗi dữ liệu; kết quả PaySim lệch 10 lần giữa các seed → chuyển dữ liệu |
| 19–24/10 | AMLWorld HI-Small; tăng đặc trưng 2 → 14 → 20 | GBDT 0,44 → 0,47 → 0,50; DRRE 0,24 (11,5 giờ, 10 epoch) |
| 28/10 | Học HI-Small, thử LI-Small | Chuyển giao giữ được độ nâng 299× |
| 31/10 – nay | **HI-Medium 31,9 triệu giao dịch**, máy 32 GB | GBDT 0,5125; DRRE đang huấn luyện |

**DRRE trên HI-Medium:**
- Mỗi epoch quét 682 cửa sổ, mỗi cửa sổ 32.768 giao dịch theo đúng thứ tự thời
  gian.
- Bộ nhớ hành vi của 2,08 triệu tài khoản được mang qua các cửa sổ.
- Mỗi epoch khoảng 1,7 giờ trên CPU. Huấn luyện chia phiên và tiếp tục được giữa
  chừng; kết quả giống hệt chạy liền, đã có test kiểm chứng.

| Epoch | Val AUPRC |
|---|---|
| 1 | 0,3207 |
| 2 | 0,3717 |
| 3–5 | **[ĐIỀN SAU]** |

### Slide 3.4 — Lượng tử: dùng cho việc gì, vì sao, và kết quả trung thực

**Bài toán lượng tử: QAPE — chọn cảnh báo cho đội điều tra.**

- Có n cảnh báo, mỗi cảnh báo có điểm rủi ro r_i và chi phí điều tra c_i.
- Đội có năng lực C mỗi ca.
- Cảnh báo có điểm ≥ 90 **bắt buộc** phải xử lý.
- Chọn tập con để tổng rủi ro được xử lý là lớn nhất.

Đây là **bài toán cái túi (knapsack)**, thuộc lớp NP-khó.

**Vì sao lượng tử:**
- Knapsack đưa được về dạng **QUBO** (tối ưu nhị phân bậc hai), cũng là dạng
  Ising mà **QAOA** giải tự nhiên.
- Đây là lớp bài toán mà tính toán lượng tử được kỳ vọng có lợi thế khi quy mô
  lớn.

**Công cụ:** Qiskit 1.4, qiskit-optimization (dựng QuadraticProgram và chuyển sang
QUBO), qiskit-algorithms (QAOA, tối ưu tham số bằng COBYLA), mô phỏng statevector
trên CPU. Đối chứng gồm Greedy, ILP (nghiệm tối ưu), Simulated Annealing và quy hoạch
động (DP).

**Kết quả:**

| Bộ giải | Chất lượng so với tối ưu | Thời gian | Quy mô |
|---|---|---|---|
| QAOA (16 qubit) | 93,26% (N = 10) | 13,5 giây | tối đa N = 10 |
| Simulated Annealing, **cùng QUBO** | 96,37% | — | — |
| ILP | 100% | 0,079 giây | — |
| Quy hoạch động | 99,87% | 0,13 giây | N = 1.000 |

**Phát hiện đáng nói:** Simulated Annealing cổ điển chạy trên **cùng QUBO** cũng
không đạt tối ưu. Vậy phần mất mát nằm ở **bước chuyển knapsack sang QUBO** (bit
bù và hệ số phạt làm méo vùng khả thi), không nằm ở phần cứng lượng tử. Nghĩa là
*ngay cả một máy lượng tử hoàn hảo cũng không thắng được với cách mã hóa này*.

**Thiết kế trung thực:** cờ `quantum.enabled = false`. Lượng tử chỉ được bật khi
benchmark chứng minh có lợi, và hiện nó chưa có lợi.

⚠ **Không nói "lượng tử giúp tăng hiệu quả".** Nói: *"Nhóm hình thức hóa bài toán
vận hành thành QUBO, benchmark QAOA với 4 bộ giải cổ điển, và chỉ ra giới hạn nằm
ở bước mã hóa."*

⚠ Quantum Kernel và VQC (nhánh lượng tử trong mô hình chấm điểm) **đã cài đặt
nhưng chưa có số đo**, nên không trình bày như kết quả.

**Phản biện dự kiến:**

- *"Vậy lượng tử có cần thiết không?"*
  Trả lời: ở quy mô hiện tại thì không, và nhóm đo được vì sao. Giá trị là khung
  QUBO đã sẵn sàng và kết luận "cần cách mã hóa ràng buộc tốt hơn trước khi chờ
  phần cứng".
- *"Sao chỉ 16 qubit?"*
  Trả lời: mô phỏng phải lưu 2ⁿ biên độ. Thêm 1 qubit là gấp đôi công việc, nên
  N = 1.000 cảnh báo cần khoảng 1.010 qubit.

---

## Phần 4 — Thang đo và kết quả (4 phút)

### Slide 4.1 — Vì sao không dùng accuracy, và dùng thang đo nào

Đoán *"mọi giao dịch đều sạch"* đã đạt **99,9% accuracy** mà không bắt được vụ
nào. Vì vậy accuracy vô nghĩa. ROC-AUC cũng chỉ để tham khảo: trên PaySim, hai lần
chạy có ROC-AUC gần bằng nhau trong khi AUPRC lệch nhau 10 lần.

Ký hiệu: P là số vụ rửa tiền thật trong tập; TP_K là số vụ thật nằm trong K cảnh
báo đầu; π = P / tổng số giao dịch là tỷ lệ nền.

| Thang đo | Công thức | Trả lời câu hỏi nào |
|---|---|---|
| **Precision@K** | TP_K / K | Trong K cảnh báo, bao nhiêu là thật (đỡ tốn công điều tra sai) |
| **Recall@K** | TP_K / P | Bắt được bao nhiêu phần trăm tổng số vụ |
| **F1@K** | 2·P·R / (P + R) | Cân bằng hai thang trên |
| **AUPRC** (Average Precision) | Σₙ (Rₙ − Rₙ₋₁)·Pₙ, cộng theo mọi ngưỡng | Chất lượng xếp hạng ở **đầu danh sách**; thang đo chính |
| **Bắt@K** | TP_K, với K = 500 / 1.000 / 2.000 | Ngôn ngữ vận hành: "đội xem 2.000 cảnh báo thì bắt được bao nhiêu vụ" |
| **Độ nâng (lift)** | Precision@K / π hoặc AUPRC / π | So với chọn ngẫu nhiên; so được giữa bộ HI và LI có tỷ lệ nền khác nhau |
| **Recall theo kiểu hình** | TP_K của kiểu hình k / số vụ của kiểu hình k | Bắt tốt hình dạng mạng lưới nào |
| **Recall đường dây** | Số đường dây có ≥1 giao dịch lọt top K / tổng số đường dây | Có lộ ra được cả mạng lưới không |
| **KTC 95% bootstrap** | Lấy mẫu lại có hoàn lại 300–1.000 lần, lấy phân vị 2,5% và 97,5% của hiệu số | Chênh lệch giữa hai mô hình có thật hay chỉ là nhiễu |
| **Tỷ lệ xấp xỉ (QAPE)** | Giá trị nghiệm của bộ giải / giá trị nghiệm tối ưu ILP | Lượng tử gần tối ưu tới đâu |

⚠ Định nghĩa recall đường dây ở trên cần xác nhận lại: script tính con số 91,1%
**không còn trong repo** (làm trên máy cũ). Phải dựng lại script và tính lại trên
HI-Medium trước khi trích **[ĐIỀN SAU]**.

### Slide 4.2 — Kết quả chính: HI-Medium, tập val

4.603.545 giao dịch trong vũ trụ chấm điểm, 5.489 vụ thật, tỷ lệ nền 0,119%.

| Phương pháp | AUPRC | Độ nâng | Precision@2000 | Recall@2000 | F1@2000 | Bắt@500 | Bắt@2000 |
|---|---|---|---|---|---|---|---|
| Ngẫu nhiên | 0,0012 | 1× | 0,1% | 0,04% | 0,001 | 0 | 2 |
| Số tiền lớn nhất (giống luật ngưỡng) | 0,0021 | 2× | 0,8% | 0,3% | 0,004 | 7 | 16 |
| Bảng tra theo kênh | 0,0075 | 6× | 0,9% | 0,3% | 0,005 | 3 | 18 |
| Isolation Forest (không dùng nhãn) | 0,0096 | 8× | 2,8% | 1,0% | 0,015 | 26 | 56 |
| Logistic, 2 đặc trưng | 0,0119 | 10× | 2,0% | 0,7% | 0,011 | 17 | 40 |
| DRRE (học trên HI-Small) | 0,2349 | 197× | 49,8% | 18,1% | 0,266 | 256 | 996 |
| DRRE (học trên HI-Medium, epoch 2) | 0,3717 | 312× | [ĐIỀN SAU] | [ĐIỀN SAU] | | | |
| Cây quyết định | 0,3865 | 324× | 89,5% | 32,6% | 0,478 | 461 | 1.789 |
| Random Forest | 0,4183 | 351× | 76,5% | 27,9% | 0,409 | 493 | 1.530 |
| **GBDT cân bằng, 20 đặc trưng** | **0,5125** | **430×** | **92,2%** | **33,6%** | **0,492** | **481** | **1.843** |

Hiệu số AUPRC giữa DRRE (học trên HI-Small) và GBDT là −0,278, KTC 95%
[−0,289; −0,266].

**Cách nói:** *"Nếu đội điều tra xem 2.000 cảnh báo, hệ thống đưa ra 1.843 vụ
thật, tức hơn 9 trên 10 cảnh báo là đúng. Xếp theo số tiền như luật ngưỡng chỉ
được 16 vụ."*

⚠ Phải nói kèm hai câu:

1. AMLWorld là dữ liệu tổng hợp. Kênh ACH chứa ~87% vụ rửa tiền, nên con số 92%
   chứng minh **phương pháp**, không phải hiệu quả tại ngân hàng thật.
2. Recall@2000 là 34%. Muốn phủ hết phải tăng ngân sách cảnh báo, và đó chính là
   vai trò của QAPE.

### Slide 4.3 — Bắt được hình dạng mạng lưới nào

Recall trong top 2.000 cảnh báo, HI-Medium val:

| Kiểu hình | Số vụ | GBDT | Cây | DRRE (HI-Small) |
|---|---|---|---|---|
| BIPARTITE | 381 | 59% | 56% | 33% |
| STACK | 540 | 55% | 52% | 29% |
| FAN-IN | 362 | 55% | **61%** | 36% |
| GATHER-SCATTER | 714 | 54% | 56% | 34% |
| SCATTER-GATHER | 617 | 54% | 49% | 25% |
| FAN-OUT | 294 | 51% | 47% | 21% |
| CYCLE | 345 | 43% | 38% | 21% |
| RANDOM | 253 | 42% | 41% | 23% |
| **Không thuộc kiểu hình nào** | 1.983 | **0%** | 0% | 0% |

**Đây là điểm mạnh và cũng là giới hạn rõ nhất.**
- Mô hình bắt được **khoảng một nửa mọi hình dạng mạng lưới** trong 2.000 cảnh
  báo.
- Mô hình **gần như không bắt được** các khoản rửa tiền ngụy trang thành giao dịch
  thường. Đây là khoản "hợp nhất" (integration), không có hình dạng mạng lưới.
  Điều này khớp đúng với bài toán đặt ra là "dòng tiền bất thường **trong mạng
  lưới**".

### Slide 4.4 — So với các phương pháp đã có

**(a) So với 11 phương pháp nội bộ:** bảng ở slide 4.2. GBDT trên 20 đặc trưng
thắng mọi phương pháp, kể cả DRRE.

**(b) Tiến triển theo đặc trưng** (HI-Small, val):

| Bộ đặc trưng | AUPRC |
|---|---|
| 14 đặc trưng | 0,4365 |
| 20 đặc trưng (+ dòng tiền) | 0,4684 |
| 21 đặc trưng (+ số tiền quy USD) | 0,5002 |

**(c) Giữ được khi tăng quy mô:** GBDT đạt độ nâng ~450× trên HI-Small và ~430×
trên HI-Medium, dù hai bộ là hai thế giới độc lập và bộ sau lớn gấp 6 lần.

**(d) Chuyển giao:**

| Kịch bản | Độ nâng |
|---|---|
| Học HI-Small, thử LI-Small | 299× |
| Học LI-Small, thử LI-Small | 99× |

Bản trên Medium: **[ĐIỀN SAU]**.

**(e) So với bài báo IBM** (F1 lớp rửa tiền, %):

| Mô hình | HI-Small | HI-Medium | LI-Small | LI-Medium |
|---|---|---|---|---|
| GNN (GIN) | 28,7 | 42,3 | 7,9 | 3,9 |
| GNN (PNA) | 56,8 | 59,7 | 16,5 | 27,7 |
| XGBoost + đặc trưng đồ thị | 63,2 | 65,7 | 27,3 | 28,2 |
| **FinGuard GBDT** | **[ĐIỀN SAU]** | **[ĐIỀN SAU]** | **[ĐIỀN SAU]** | **[ĐIỀN SAU]** |

Chưa so được trực tiếp, vì bài báo dùng F1 ở ngưỡng tối ưu và chia 60/20/20, còn
nhóm dùng AUPRC và chia 70/15/15. Kết luận chung vẫn trùng nhau: **ngay trong bài
báo, mô hình cây cộng đặc trưng đồ thị thắng mạng nơ-ron đồ thị.** Kết quả "GBDT
thắng DRRE" của nhóm nhất quán với tài liệu đã công bố.

**(f) Tốc độ:** khoảng 10 giây cho 1 triệu giao dịch trên toàn chuỗi (GBDT chấm
664.951 dòng/giây). DRRE chậm hơn khoảng 50 lần.

⚠ **Toàn bộ là số trên tập val.** Bộ số chính thức trên tập test sẽ chạy một lần
duy nhất ngày 06/11 **[ĐIỀN SAU]**. Nên thay bảng 4.2 bằng số test nếu kịp.

---

## Phần 5 — Kết luận và hướng phát triển (1,5 phút)

### Slide 5.1 — Kết luận

1. Rửa tiền là bài toán **mạng lưới**. Đặc trưng nhân quả về thói quen, thời điểm
   và quan hệ mạng lưới đưa AUPRC lên **430 lần mức ngẫu nhiên**, và giữ được khi
   dữ liệu tăng 6 lần.
2. Với 2.000 cảnh báo/ngày: **hơn 9/10 cảnh báo là vụ thật**, bắt được khoảng một
   nửa mọi hình dạng mạng lưới.
3. Mô hình cây trên đặc trưng tốt thắng mạng nơ-ron phức tạp (DRRE). Điều này khớp
   với kết luận của IBM.
4. Lượng tử: QAOA chưa thắng bộ giải cổ điển, và nhóm đã chỉ ra nguyên nhân nằm ở
   bước mã hóa QUBO.
5. Quy trình đo nghiêm ngặt:
   - không rò rỉ;
   - tập test giữ đến cuối;
   - khoảng tin cậy cho mọi so sánh;
   - tự phát hiện và xử lý các lỗi trong dữ liệu công khai.

⚠ Ý 3 là quyết định định vị chưa chốt (câu hỏi 1 của báo cáo 31/10). Hai cách
nói:

- **(A) Trung thực tối đa, khuyến nghị:** *"FinGuard dùng GBDT làm bộ chấm điểm
  chính; DRRE dùng làm bộ sinh biểu diễn và giải thích."* Embedding DRRE cộng
  thêm +0,0222 AUPRC vào GBDT, KTC 95% [+0,0026; +0,0427].
- **(B) Giữ DRRE làm mô hình chính:** chỉ nên chọn nếu DRRE trên Medium sau 5
  epoch tiến sát GBDT **[ĐIỀN SAU]**.

### Slide 5.2 — Hướng phát triển

| Hướng | Vì sao | Điều kiện |
|---|---|---|
| Thí điểm trên dữ liệu ngân hàng thật (ẩn danh) | Kiểm chứng ngoài dữ liệu tổng hợp | Hợp tác ngân hàng, tuân thủ Nghị định 13/2023 |
| Bắt rửa tiền **không có hình dạng mạng lưới** (khoản hợp nhất) | Đây là điểm mù hiện tại: gần 0% | Thêm thông tin chủ thể, ngành nghề, lương |
| Cảnh báo theo **đường dây** thay vì từng giao dịch | Điều tra viên làm việc theo vụ án | Gom cụm thành phần liên thông của các cảnh báo |
| Sửa kiến trúc DRRE | Ba nguyên nhân cấu trúc đã xác định: các nhánh dùng chung đầu vào, hợp nhất chỉ cộng không nhân, mất mát phụ bão hòa | Có GPU |
| Lượng tử: mã hóa ràng buộc không cần bit bù (QAOA có ràng buộc, mixer XY) | Tấn công đúng chỗ mất mát đã đo | Phần cứng thật khi đủ qubit |
| Mở rộng lên bộ Large (180 triệu giao dịch, 97 ngày) | Đo được phát hiện sớm theo thời gian | Viết lại tiền xử lý theo luồng |
| Kết hợp lừa đảo và rửa tiền | Theo tiền lừa đảo đến khâu rửa, đúng mô hình vụ Mr Pips | Dữ liệu thiết bị và kênh số |

---

## Phụ lục A — Việc phải xong trước 10/11 để điền các ô [ĐIỀN SAU]

| Hạn | Việc | Điền vào |
|---|---|---|
| 05/11 | DRRE HI-Medium epoch 3–5 | Slide 3.3, 4.2, quyết định A/B ở 5.1 |
| 05/11 | Dựng lại script recall đường dây, tính trên HI-Medium | Slide 4.1, 4.3 |
| 05/11 | F1 ở ngưỡng tối ưu cho GBDT trên 4 bộ | Slide 4.4(e) |
| 06/11 | Chuyển giao HI→LI trên Medium | Slide 4.4(d) |
| 06/11 | **Chạy tập test một lần** | Thay bảng 4.2 |
| 08/11 | Kiểm tra tin mới nhất về vụ Mr Pips | Slide 1.1 |
| 08–09/11 | Diễn tập 2 lần có bấm giờ; video demo dự phòng | — |

## Phụ lục B — Nguồn

- Phiên tòa vụ Mr Pips: [Nhân Dân — Mr Pips và 187 đồng phạm hầu tòa](https://nhandan.vn/mr-pips-va-187-dong-pham-hau-toa-post998094.html);
  [Báo Lào Cai — Shark Bình khai về hành vi rửa tiền hơn 318 tỷ đồng](https://baolaocai.vn/shark-binh-khai-gi-ve-hanh-vi-rua-tien-hon-318-ty-dong-cho-mr-pips-post913143.html).
- Vạn Thịnh Phát giai đoạn 2: [SGGP — Trương Mỹ Lan "rửa" hơn 445.000 tỷ đồng như thế nào](https://www.sggp.org.vn/xet-xu-vu-an-van-thinh-phat-giai-doan-2-truong-my-lan-rua-hon-445000-ty-dong-nhu-the-nao-post759763.html).
- IBM AMLWorld: Kaggle `ealtman2019/ibm-transactions-for-anti-money-laundering-aml`;
  Altman và cộng sự, NeurIPS 2023, arXiv 2306.16424.
- Số liệu HI-Medium: `reports/eval/20261031-100336-evaluate_amlworld_val.json`.
- Số liệu HI-Small, QAPE, chuyển giao: báo cáo 31/10/2026.
