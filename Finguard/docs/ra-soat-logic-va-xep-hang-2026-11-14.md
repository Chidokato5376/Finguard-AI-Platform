# Rà soát logic DRRE và bảng xếp hạng mô hình — 14/11/2026

Nhóm CMET (AQ2026-111). Tài liệu này trả lời hai câu hỏi: (1) bản DRRE mới đã sửa hết lỗi
logic chưa, còn lỗ hổng nào; (2) các mô hình xếp hạng thế nào theo từng phép đo, khi đo
cùng một quy tắc. Chưa bàn kết quả có khả quan hay không.

Bảng xếp hạng đầy đủ: `reports/leaderboard/processed_aml_v2_val/leaderboard.{md,json}`,
tạo bởi `scripts/leaderboard.py`.

---

## 1. Tám lỗi của lộ trình đợt 2

| # | Lỗi | Trạng thái | Sửa ở | Bằng chứng |
|---|---|---|---|---|
| 1 | Chấm theo tài khoản gửi, bài toán là chấm giao dịch (cạnh) | **Đã sửa** | R5 (2B5) | φ₃ nhận [g_người gửi ‖ g_người nhận]; cùng x_t, h̃_t, δ_t trong bộ hợp nhất. Bộ nhớ hành vi vẫn chỉ của người gửi — người nhận được mô tả qua hồ sơ nút (gửi/nhận) trong đồ thị |
| 2 | φ và đầu chấm điểm tuyến tính, hợp nhất chỉ cộng | **Đã sửa** | R3 (2A) | φ là MLP 2 lớp; hợp nhất ghép nối rồi MLP; đầu chấm điểm MLP |
| 3 | Ba nhánh chung đầu vào → cổng sụp | **Đã sửa** | R3 (2A) | Mỗi nhánh một nhóm cột (`BRANCH_OF_*`); test `test_nhanh_giao_dich_khong_thay_cot_cua_nhanh_khac` |
| 4 | δ_t tính ra nhưng không vào điểm | **Đã sửa** | R3 (2A) | δ_t và \|q_t − h̃_t\| vào nhánh thói quen, δ_t vào bộ hợp nhất |
| 5 | Đồ thị một chiều, không đặc trưng cạnh, nút chỉ-nhận = 0, **nhìn trước trong cửa sổ** | **Đã sửa** | R4 (2B1), R5 (2B2–2B5) | Đồ thị chỉ gồm cạnh TRƯỚC đầu khối (24 giờ); hai chiều, hai bộ trọng số; cạnh có số tiền, kênh, độ cũ; nút có hồ sơ gửi và nhận. Test: thêm giao dịch tương lai không đổi điểm cũ (đồ thị cũ thì đổi); đồ thị con trùng khít đồ thị đầy đủ |
| 6 | 83% mất mát cho nhiệm vụ phụ đã bão hòa; mỗi cập nhật thấy ~31 vụ; thứ tự cửa sổ cố định | **CHƯA SỬA** | 2C (R6) | l_pred ≈ 0,003; embedding chỉ dùng 6–8/128 chiều (R3 6,3; R4 6,7) |
| 7 | Đầu vào không chuẩn hóa; kênh là số 0–6 | **Đã sửa** | R3 (2A) | Chuẩn hóa học trên train, lưu trong checkpoint; kênh qua embedding 8 chiều |
| 8 | evaluate.py chấm DRRE không mang trạng thái | **Đã sửa** | Giai đoạn 0 | Số evaluate khớp số train.py (R3: 0,3487 so với 0,3490) |

**Kết luận: 7/8 lỗi đã sửa; lỗi 6 (cách huấn luyện) là việc của 2C.**

## 2. Lỗ hổng phát hiện thêm khi rà soát (ngoài 8 lỗi)

| Lỗ hổng | Hậu quả nếu để nguyên | Trạng thái |
|---|---|---|
| Baseline trong `evaluate.py` chỉ nhận 20 cột gốc, DRRE nhận 28 cột (có cột kịch bản) | So DRRE với GBDT là so hai lượng thông tin, thiên về DRRE (GBDT 20 cột 0,460 < 28 cột 0,574) | **Đã sửa 14/11**: baseline nhận đúng các cột kịch bản của checkpoint |
| `drre_contribution.py` (P3) dùng danh sách nhóm cột cứng của bộ 20 cột | Cột kịch bản không thuộc nhóm nào khi xáo trộn → tỷ trọng đóng góp sai | **Đã sửa 14/11**: cột kịch bản gom theo mô-đun |
| Không có bảng so sánh thống nhất; mỗi báo cáo so một cặp, khác ngày khác cách đo | Không xếp hạng được, dễ trích nhầm số cũ | **Đã có** `scripts/leaderboard.py` (mục 3) |
| Checkpoint và đặc trưng (R1) được chọn trên val, rồi báo cáo trên val | Số val lạc quan cho mô hình được chọn | **Quy ước**: val chỉ để so tương đối giữa các bậc; số chốt là **test, chạy đúng một lần** (R10). Tập giữ lại 15% cuối train (giao thức mục 2) **chưa cài** — cần trước Cổng 1 nếu chọn ngưỡng theo phân khúc |
| Một seed | Hiệu số nhỏ có thể là nhiễu seed | Cổng 1 chạy 3 seed (42, 43, 44) |
| Checkpoint HI-Medium (đợt 1, v1) dùng đồ thị nhìn trước | Phép đo P2 trên HI-Medium (+0,039) có thể bị thổi phồng | Không dùng làm số chốt; đo lại bằng bản không nhìn trước |

## 3. Bảng xếp hạng hiện tại — val HI-Small, cùng quy tắc

733.086 giao dịch trong vũ trụ chấm điểm, 760 vụ rửa tiền. Mỗi ô: giá trị (hạng).

| # | Mô hình | Cột | AUPRC | ROC-AUC | Bắt@500 | Bắt@1000 | Bắt@2000 | F1 tốt nhất | Hạng TB | Δ AUPRC vs hạng 1 [KTC 95%] |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | GBDT cân bằng | 28 | **0,574** (1) | 0,985 (1) | 367 (1) | 438 (1) | 503 (1) | 0,589 (1) | 1,00 | — |
| 2 | GBDT cân bằng (mốc đợt 1) | 20 | 0,460 (2) | 0,983 (4) | 297 (2) | 369 (2) | 446 (2) | 0,487 (2) | 2,33 | −0,113 [−0,135; −0,094] |
| 3 | Random Forest | 28 | 0,400 (3) | 0,983 (5) | 268 (3) | 357 (3) | 436 (3) | 0,435 (3) | 3,33 | −0,174 [−0,201; −0,147] |
| 4 | **DRRE R5 (2A + 2B đầy đủ, không nhìn trước)** | 28 | 0,355 (4) | 0,984 (2) | 228 (5) | 312 (4) | 411 (4) | 0,366 (5) | 4,00 | −0,219 [−0,245; −0,188] |
| 5 | DRRE R3 (2A, đồ thị nhìn trước) | 28 | 0,349 (5) | 0,983 (3) | 241 (4) | 279 (5) | 330 (7) | 0,392 (4) | 4,67 | −0,224 [−0,253; −0,195] |
| 6 | DRRE R4 (2A + đồ thị quá khứ một chiều) | 28 | 0,259 (6) | 0,981 (6) | 188 (6) | 273 (6) | 377 (5) | 0,319 (6) | 5,83 | −0,315 [−0,347; −0,283] |
| 7 | DRRE R2 (v1) | 28 | 0,174 (7) | 0,975 (7) | 147 (7) | 226 (7) | 344 (6) | 0,268 (7) | 6,83 | −0,399 [−0,431; −0,365] |
| 8 | Logistic | 28 | 0,047 (8) | 0,975 (8) | 43 (8) | 87 (8) | 155 (8) | 0,117 (8) | 8,00 | −0,526 |
| 9 | Ngẫu nhiên | — | 0,001 (9) | 0,512 (9) | 0 (9) | 1 (9) | 3 (9) | 0,002 (9) | 9,00 | −0,572 |

Recall theo kiểu hình trong 2.000 cảnh báo đầu: GBDT 28 cột đứng đầu 8/9 kiểu hình (hạng TB
1,11). **R5 đứng hạng 1 ở FAN-OUT (100%, GBDT 98%)**, hạng 2 ở CYCLE (82%) và RANDOM (82%), hạng
3 ở STACK (74%) và BIPARTITE (80%); hạng TB kiểu hình 3,56 — ngang Random Forest, cao nhất các
bản DRRE.

Hiệu số giữa các bậc (evaluate.py, bootstrap ghép cặp): R3 − R2 = +0,175 [0,144; 0,203];
R4 − R3 = −0,090 [−0,119; −0,064] (giá của việc bỏ nhìn trước); **R5 − R4 = +0,096
[0,074; 0,115]**. Embedding R5: số chiều hiệu dụng 7,7 (eff rank 10,3/128) — vẫn hẹp.

**Kiểm tra chéo:** số trong bảng khớp các báo cáo trước đến chữ số thứ tư (R3 0,3490 = evaluate
13/11; GBDT 28 cột 0,5735 = R1 "+core" 200 cây; GBDT 20 cột 0,4603 = R1 base20) — cùng dòng,
cùng nhãn, cùng thứ tự.

## 4. Logic có phục vụ đúng mục đích dự án không

| Mục đích | Đáp ứng bởi |
|---|---|
| Chấm điểm rủi ro TỪNG GIAO DỊCH theo thời gian thực | Đọc theo cạnh (R5); mọi đặc trưng và đồ thị chỉ dùng quá khứ; trạng thái mang qua cửa sổ như hệ thống thật |
| Không rò rỉ tương lai | Đặc trưng kịch bản chỉ dùng quá khứ (test nhân quả); đồ thị quá khứ (test thêm giao dịch tương lai); chia tập theo thời gian; is_fx bị loại |
| So sánh công bằng với mô hình cổ điển | Cùng dòng, cùng vũ trụ chấm điểm, cùng 28 cột, cùng cách chấm; KTC bootstrap ghép cặp |
| Thước đo vận hành của AML | Số vụ bắt được ở 500/1.000/2.000 cảnh báo; ma trận nhầm lẫn ở ngưỡng chọn trên val; theo kênh; theo kiểu hình |
| Chứng minh DRRE đóng góp gì | P1 (bảng này), P2/P3 (`drre_contribution.py`, đã sửa nhóm cột), P4 theo kiểu hình — chạy ở Cổng 1 với 3 seed |

## 5. Việc còn lại trước khi có kết luận

1. ~~Chạy nốt R5~~ — xong 14/11 (val 0,247 / 0,307 / 0,354 / 0,372; checkpoint = epoch 3).
2. 2C (lỗi 6) → R6.
3. Cổng 1: R6 so với GBDT 28 cột, 3 seed, P1–P4; cài tập giữ lại 15% cuối train nếu cần chọn
   ngưỡng mà không dùng val.
4. Test một lần (R10).

## 6. Kết quả Giai đoạn 2C (lỗi 6) — 14–15/11

| Bậc | Thay đổi so với R5 | AUPRC tập giữ lại (15% cuối train) | AUPRC val | Bắt@2000 val | Embedding: số chiều hiệu dụng |
|---|---|---|---|---|---|
| R5 | — | 0,348 | **0,355** | **411** | 7,7 (eff rank 10,3) |
| R6a | Đầu chấm điểm học lại trên embedding đóng băng (mọi vụ cùng lúc) | 0,353 | 0,333 | 375 | 7,7 |
| R6b | Thêm số hạng chống sụp embedding (λ = 0,1) khi huấn luyện | 0,342 | 0,349 | 406 | **10,5 (eff rank 14,1)** |
| R6 | R6b + đầu chấm điểm học lại | **0,361** | 0,343 | 380 | 10,5 |

- Đầu chấm điểm được chọn hàm mất mát và số epoch trên tập giữ lại (không nhìn val): BCE
  trọng số đầy đủ thắng ở cả R6a và R6 (√ và focal thấp hơn).
- Số hạng chống sụp làm embedding rộng hơn ~37% nhưng vẫn chỉ ~11/128 chiều; đường val theo
  epoch của R6b trùng R5 (0,249/0,312/0,348/0,366 so với 0,247/0,307/0,354/0,372).
- **Kết luận: 2C không cải thiện DRRE.** Mọi hiệu số với R5 (≤ 0,022) nhỏ hơn nhiều độ rộng
  KTC đo được ở các phép so khác (~±0,03) và trái dấu giữa tập giữ lại và val — tức là nhiễu.
  Cách huấn luyện không phải điểm nghẽn; khoảng cách với GBDT 28 cột (−0,22) nằm ở lượng thông
  tin/kiến trúc, không ở cách học đầu chấm điểm hay độ rộng embedding.
- Cấu hình DRRE mang vào Cổng 1: **R5** (đơn giản nhất trong nhóm tốt nhất). Mã 2C giữ lại
  (`--lambda-vc`, `--accumulate`, `scripts/refit_head.py`) để dùng khi cần.
