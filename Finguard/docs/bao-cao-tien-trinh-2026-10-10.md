# Báo cáo tiến trình FinGuard AI — 2026-10-10

Nhóm CMET (AQ2026-111) · vòng bán kết · báo cáo nội bộ

Mọi số liệu trong tài liệu này đều truy được về một manifest trong `reports/runs/`
(lần huấn luyện) hoặc `reports/eval/` (lần đánh giá), kèm commit, seed, phiên bản
thư viện và dấu vân tay sha256 của tệp dữ liệu.

---

## 1. Đã làm được gì

| Hạng mục | Trạng thái |
|---|---|
| Tiền xử lý → train → val → test → báo cáo | chạy bằng một lệnh, có manifest truy nguyên |
| Quy mô lớn nhất đã huấn luyện | **3 triệu dòng PaySim**, 40–43 phút/lần |
| Tăng tốc so với bản đầu | **29–42 lần** (vector hóa theo bước thời gian) |
| Số lần huấn luyện có manifest | 5 |
| Số báo cáo đánh giá trên tập test | 6 |
| Kiểm thử tự động | **186 test pass** |
| Tầng tăng cường dữ liệu bằng kịch bản | đã sinh 1 bộ (3.675 dòng, 4 kịch bản), **chưa dùng để huấn luyện** |
| Bài toán phân bổ điều tra (QAPE) | đã benchmark QAOA / Greedy / ILP / SA / DP |

Hạ tầng đo lường gồm: manifest thí nghiệm, bootstrap ghép cặp cho khoảng tin cậy,
chọn checkpoint chống nhiễu bằng trung bình trượt, và đo chỉ số theo từng giai đoạn
thời gian (`--by-period`).

---

## 2. Bốn kết quả đạt mức có ý nghĩa thống kê

Khoảng tin cậy 95% bằng bootstrap ghép cặp (lấy cùng chỉ số cho cả hai bên).

| # | Kết quả | Hiệu số AUPRC | KTC 95% |
|---|---|---|---|
| 1 | Behavior Memory **làm giảm** chất lượng (IEEE-CIS) | −0,0068 | [−0,0103; −0,0040] |
| 2 | DRRE **kém** GBDT + lịch sử (IEEE-CIS) | −0,0200 | [−0,0290; −0,0114] |
| 3 | DRRE **kém** GBDT + lịch sử (PaySim) | −0,1313 | [−0,1740; −0,0844] |
| 4 | Huấn luyện 15 epoch hơn 3 epoch (PaySim, seed 42) | +0,1377 | [+0,1097; +0,1700] |

Ba trong bốn kết quả là bất lợi. Kết quả thứ 4 có lợi nhưng chỉ đúng với một seed
(xem mục 3).

**Lưu ý khi đọc**: "có ý nghĩa thống kê" chỉ nói khoảng tin cậy không chứa 0, không
nói độ lớn có đáng kể. Ví dụ hai checkpoint IEEE lệch nhau −0,0001 với KTC
[−0,0001; −0,0000] — có ý nghĩa thống kê nhưng vô nghĩa thực tiễn. Mọi phát biểu
phải đi kèm độ lớn.

---

## 3. Vấn đề lớn nhất: kết quả PaySim không tái lập được

Ba lần huấn luyện giống nhau hoàn toàn, chỉ khác số seed:

| Seed | Val AUPRC | **Test AUPRC** | ROC-AUC | Bắt được / 500 cảnh báo |
|---|---|---|---|---|
| 42 | 0,2464 | **0,1513** | 0,918 | **66** |
| 43 | 0,1660 | **0,0183** | 0,911 | **1** |
| 44 | 0,1271 | **0,0151** | 0,868 | **0** |

- Dải test rộng **10 lần**; trung vị 0,0183.
- Mốc "chỉ xếp theo số tiền" là 0,0156 → **2 trong 3 seed không hơn gì việc sắp
  xếp giao dịch theo số tiền**.
- Với 500 cảnh báo/ngày: 66 vụ, 1 vụ, 0 vụ.

Vì vậy **con số 0,1513 không được dùng làm số liệu chính**. Phát biểu đúng là
"test AUPRC 0,015–0,151 qua 3 seed, trung vị 0,018".

Phát hiện phụ quan trọng: **ROC-AUC ổn định (0,868–0,918) trong khi AUPRC lệch 10
lần**. Đây là dấu hiệu kinh điển của việc mô hình phân tách tốt phần lớn dữ liệu
nhưng chỉ một seed xếp đúng phần **đỉnh** danh sách. Với tỷ lệ gian lận 0,1%, chỉ
phần đỉnh có giá trị vì điều tra viên không xem tới dòng thứ 10.000. **ROC-AUC
0,918 không được dùng làm bằng chứng chất lượng**, dù nó là con số đẹp nhất.

Val cũng không dự báo được test: tỷ lệ val/test là 1,6 lần ở seed 42 nhưng 9,1 và
8,4 lần ở seed 43 và 44.

---

## 4. DRRE đầy đủ đang thua ở bốn mặt nào

**Mặt 1 — thua baseline cổ điển trên dữ liệu thật.** IEEE-CIS, 45.000 dòng test,
1.757 nhãn dương:

| Phương pháp | AUPRC | Bắt được / 500 |
|---|---|---|
| **DRRE đầy đủ** | **0,0976** | **51** |
| Logistic cùng đầu vào | 0,1010 | 64 |
| GBDT cùng đầu vào | 0,1079 | 77 |
| GBDT + lịch sử | 0,1176 | **105** |

Thua cả hồi quy logistic. Với cùng ngân sách 500 cảnh báo, GBDT + lịch sử bắt được
gấp đôi (105 so với 51).

**Mặt 2 — thua chính bản rút gọn của nó.** Bỏ nhánh Behavior Memory thì AUPRC
**tăng** 0,0068 (KTC [−0,0103; −0,0040] cho hiệu số có-trừ-không). Nhánh này chiếm
**13.504 / 56.201 = 24%** tham số của mô hình và đang làm hại.

**Mặt 3 — thua về tính tái lập.** Mục 3 ở trên: dải 10 lần giữa các seed.

**Mặt 4 — thua đúng ở nơi quan trọng nhất.** Mô hình giữ được ROC-AUC nhưng mất
AUPRC, tức là nó thua ở phần đỉnh danh sách cảnh báo — phần duy nhất được dùng
trong thực tế.

---

## 5. Nguyên nhân: do đâu

### 5.1 Mô hình bị bỏ đói đầu vào (nguyên nhân chắc chắn nhất)

DRRE hiện chỉ nhận **2 đặc trưng**: `amount_norm`, `channel_encoded`
(`src/drre/train.py::FEATURE_COLUMNS`). Baseline thắng nó dùng **6 đặc trưng**:
số tiền, kênh, giờ, số giao dịch trước đó, z-score số tiền, khoảng cách thời gian.

Nhóm đang yêu cầu một mạng nơ-ron 56.201 tham số đánh bại một cây quyết định được
cho **gấp ba lượng thông tin**. Khoảng cách 0,0976 so với 0,1176 có thể chỉ phản
ánh chênh lệch đầu vào, chưa phản ánh chênh lệch kiến trúc.

### 5.2 Behavior Memory không có gì để ghi nhớ

Ba điều kiện cùng lúc làm nhánh này vô dụng:

1. **Dữ liệu**: PaySim có 99,85% người gửi chỉ xuất hiện một lần; chỉ 2,4% giao
   dịch gian lận trong tập test có lịch sử người nhận. Chuỗi đưa vào GRU hầu hết
   dài bằng 1.
2. **Thiết kế**: trạng thái `(h, m)` được **khởi tạo lại ở đầu mỗi cửa sổ** và
   không mang qua giữa các cửa sổ hay các epoch — đây là điểm đơn giản hóa có chủ
   ý cho MVP, đã ghi trong docstring của `train.py`.
3. **Kích thước cửa sổ**: cửa sổ 8192 giao dịch trên 2,1 triệu dòng nghĩa là mỗi
   tài khoản chỉ được "nhớ" trong một lát thời gian rất ngắn.

Một bộ nhớ hành vi không có chuỗi hành vi để học thì chỉ còn là 13.504 tham số
nhiễu — khớp đúng với kết quả đo ở Mặt 2.

### 5.3 Phần có nhãn chỉ chiếm 17% hàm mất mát, và đầu ra chỉ có 129 tham số

Hàm mất mát: `L = L_pred + 0,5·L_contrast + 0,3·L_BCE`.

- `L_BCE` là **thành phần duy nhất dùng nhãn**, chiếm `0,3 / 1,8 ≈ 17%` trọng số.
- `RiskScoringEngine` — bộ phận biến embedding 128 chiều thành điểm rủi ro — chỉ
  có **129 tham số** và **chỉ nhận gradient từ `L_BCE`**.
- `pos_weight = 988,6`, và mỗi cửa sổ 8.192 dòng chỉ có khoảng **8 giao dịch gian
  lận**.

Nghĩa là toàn bộ việc học "điểm rủi ro cao nghĩa là gì" được quyết định bởi
khoảng 8 mẫu mỗi bước, mỗi mẫu được nhân trọng số gần 1.000, qua một tầng tuyến
tính 129 tham số. **Đây là giải thích cơ học trực tiếp cho dải 10 lần giữa các
seed**: đầu ra của mô hình được huấn luyện bởi rất ít mẫu rất "to tiếng", nên
hướng nó hội tụ phụ thuộc mạnh vào thứ tự mẫu do seed quyết định.

### 5.4 Không thể dùng val để cứu một lần chạy xấu

Tập val chỉ có 262 nhãn dương, nên đường val dao động tới 0,14 AUPRC giữa hai epoch
liền nhau, và tỷ lệ val/test khác nhau 1,6–9,1 lần theo seed. Việc chọn checkpoint
theo val đã giảm phương sai thật (dải seed 6,33 lần → 1,94 lần nếu so trên val),
nhưng không ngăn được việc hai seed vẫn cho kết quả test ở mức baseline số tiền.

---

## 6. Một dấu hiệu tích cực đáng theo

Đo theo giai đoạn thời gian trên tập test PaySim (`--by-period 4`):

| Giai đoạn | Gian lận / 112.500 dòng | Mật độ | DRRE (lift) | GBDT + lịch sử (lift) |
|---|---|---|---|---|
| 1 | 202 | 0,180% | 0,2769 (×154) | **0,3883 (×216)** |
| **2** | **54** | **0,048%** | **0,1273 (×265)** | 0,0783 (×163) |
| 3 | 44 | 0,039% | 0,0643 (×164) | **0,0809 (×207)** |
| 4 | 162 | 0,144% | 0,1311 (×91) | **0,3019 (×210)** |

Ở giai đoạn 2 — vùng **mật độ gian lận thấp nhất** — DRRE vượt GBDT về lift
(×265 so với ×163). Đó là vùng mà các dấu hiệu thống kê thô yếu nhất, cũng là vùng
mà kiến trúc DRRE được thiết kế cho. Nhưng giai đoạn này chỉ có 54 nhãn dương nên
đây là **giả thuyết đáng kiểm chứng, không phải kết luận**.

Trên IEEE-CIS, DRRE ổn định qua cả 4 giai đoạn (0,090–0,114), không có dấu hiệu
trôi theo thời gian — nhưng bị dẫn trước ở cả 4/4 giai đoạn.

---

## 7. Biện pháp dự kiến

Xếp theo tỷ lệ (cơ sở kỳ vọng) / (chi phí). Mỗi biện pháp gắn với nguyên nhân cụ thể.

| # | Biện pháp | Trị nguyên nhân | Chi phí máy |
|---|---|---|---|
| 1 | **Mở rộng đặc trưng 2 → 8** (`--features extended`): thêm giờ, số giao dịch trước đó, z-score số tiền, khoảng cách thời gian, để DRRE có cùng lượng thông tin như baseline | 5.1 | ~4,5 giờ (3 seed) |
| 2 | **Tăng trọng số phần có nhãn** và mở rộng `RiskScoringEngine` khỏi mức 129 tham số | 5.3 | ~2,2 giờ (3 seed) |
| 3 | **Dùng bộ dữ liệu tăng cường để huấn luyện** — nó có chuỗi money mule và chuỗi chiếm quyền tài khoản mà PaySim không có, tức là điều kiện dữ liệu duy nhất để Behavior Memory có thể có ích | 5.2 (phía dữ liệu) | ~1 giờ |
| 4 | **Behavior Memory mang trạng thái qua các cửa sổ** (`--stateful-behavior`) — chỉ có nghĩa khi làm cùng biện pháp 3, vì trên PaySim thuần thì không có chuỗi để mang | 5.2 (phía mô hình) | ~2,5 giờ |
| 5 | **Kiểm chứng giả thuyết vùng mật độ thấp** (mục 6) trên nhiều seed và nhiều cách chia giai đoạn | — | ~30 phút |

### Thay đổi quy tắc báo cáo — áp dụng ngay, không tốn chi phí máy

1. Mọi số liệu công bố phải là **trung vị của ≥3 seed, kèm dải**. Không bao giờ nêu
   một seed đơn lẻ.
2. **Không dùng ROC-AUC** làm số liệu chính trên dữ liệu mất cân bằng; luôn đi kèm
   AUPRC và số vụ bắt được theo ngân sách cảnh báo.
3. Mọi so sánh phải kèm khoảng tin cậy; "cao hơn" không có KTC thì không phải kết luận.
4. Số liệu chính dùng **IEEE-CIS** (dữ liệu thật). PaySim chỉ dùng để chứng minh
   quy mô và tốc độ, và luôn kèm cảnh báo lỗi sinh dữ liệu: gian lận chỉ bao giờ
   thuộc 2 trong 5 loại giao dịch, và giờ 0–8 chiếm 2,4% giao dịch nhưng 37,1%
   gian lận.

---

## 8. Đánh giá thẳng thắn

Phần **hạ tầng và phương pháp đo** của dự án đang chắc. Phần **mô hình** thì chưa
chứng minh được là hơn baseline cổ điển trên dữ liệu thật, và một thành phần được
quảng bá là điểm mới (Behavior Memory) đã có bằng chứng thống kê là đang làm hại.

Điều đáng nói: chính hạ tầng đo lường đó đã bắt được một kết quả không tái lập mà
một nhóm chỉ chạy một seed sẽ không bao giờ phát hiện. Nếu không chạy 3 seed, nhóm
sẽ bước vào bán kết với con số 0,1513 và không trả lời được câu hỏi "chạy lại có ra
không".

Kiến nghị: làm biện pháp 1 và 2 trước để biết DRRE có hơn baseline khi được cho đủ
đầu vào hay không, rồi trình bày toàn bộ hành trình — **kể cả các kết quả bất lợi**,
vì nếu hội đồng tự phát hiện ra chúng thì tốn kém hơn nhiều so với việc nhóm chủ
động nêu kèm giải thích kỹ thuật.

---

## Phụ lục — cách chạy lại mọi số liệu

```bash
# Huấn luyện 3 triệu dòng PaySim, 15 epoch, chọn checkpoint chống nhiễu
python -m src.drre.train --source paysim --epochs 15 --batch-size 8192 \
  --processed-dir data/processed_3m --checkpoint models/drre_paysim_3m_15ep.pt \
  --seed 42 --select smoothed

# Đánh giá trên tập test, so với baseline tốt nhất, kèm KTC và đo theo giai đoạn
python scripts/evaluate.py --source ieee_cis --split test \
  --checkpoint models/drre_ieee_v2.pt --compare-baseline gbdt_them_lich_su \
  --bootstrap 1000 --by-period 4
```

Tham chiếu manifest:

| Số liệu | Tệp |
|---|---|
| IEEE-CIS: DRRE so với baseline | `reports/eval/20261010-153001-evaluate_ieee_cis_test.json` |
| PaySim seed 42: so với baseline | `reports/eval/20261010-154235-evaluate_paysim_test.json` |
| PaySim seed 43 và 44 | `reports/eval/20261010-154433-evaluate_paysim_test.json` |
| Huấn luyện seed 42 / 43 / 44 | `reports/runs/20261009-130837-*.json`, `20261010-070800-*.json`, `20261010-075046-*.json` |
