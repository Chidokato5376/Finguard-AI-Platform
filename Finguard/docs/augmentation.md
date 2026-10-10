# Tầng huấn luyện tăng cường dữ liệu bằng kịch bản tổng hợp

**Trạng thái triển khai.** Tầng này được thiết kế như một mở rộng của quy trình
huấn luyện hiện có. Bảy thành phần của tầng (hiệu chỉnh, sinh đặc tả, kiểm tra
đặc tả, simulator, injector, dataset có phiên bản, khai thác đối kháng) **đã
được triển khai và có kiểm thử tự động**; phần **chưa hoàn thành là kết quả
thực nghiệm** — giao thức so sánh ba cấu hình đã có mã và chỉ số nhưng chưa
chạy đủ nhiều seed, nên **chưa có kết luận nào về đóng góp của LLM**.

| Thành phần | File | Trạng thái |
|---|---|---|
| Calibration | `src/augmentation/calibration.py` | Đã triển khai, có kiểm thử |
| Spec generator | `src/augmentation/llm_adapter.py` | Đã triển khai, có kiểm thử (transport offline, không cần mạng) |
| Spec validator | `src/augmentation/spec_schema.py` | Đã triển khai, có kiểm thử |
| Simulator | `src/augmentation/simulator.py` | Đã triển khai, có kiểm thử |
| Injector + kiểm soát chất lượng | `src/augmentation/quality.py` | Đã triển khai, có kiểm thử |
| Synthetic dataset có phiên bản | `src/augmentation/dataset.py` | Đã triển khai, có kiểm thử |
| Adversarial mining | `src/augmentation/mining.py` | Đã triển khai, **mặc định TẮT** |
| Giao thức đánh giá 3 cấu hình | `src/augmentation/protocol.py` | Có mã + chỉ số; **chưa chạy đủ seed** |

Toàn bộ 42 kiểm thử của tầng nằm trong `tests/test_augmentation.py`.

## 1. Bối cảnh và mục tiêu

Dữ liệu gian lận có nhãn có hai đặc điểm bất lợi: lớp dương rất hiếm, và phủ
không đầy đủ các kiểu hình đã xuất hiện. Tầng này giảm hai hạn chế đó mà không
làm suy yếu tính tin cậy của kết luận. **Dữ liệu tổng hợp là nguồn bổ sung cho
huấn luyện, không phải căn cứ đánh giá**; mọi kết luận về hiệu năng chỉ rút ra
từ dữ liệu thực giữ lại, chia theo thời gian. Hàm
`protocol.assert_eval_set_is_real()` chặn việc vi phạm nguyên tắc này, và được
gọi tự động trong `protocol.evaluate_scores()`.

## 2. Bốn nguyên tắc thiết kế

1. **Neo thực nghiệm.** Tham số kịch bản được hiệu chỉnh từ dữ liệu thực có
   nhãn và từ tri thức miền có dẫn nguồn. `Calibration` ghi lại **nguồn của
   từng biên** (`du_lieu_thuc` / `tri_thuc_mien` / `tham_so_van_hanh`), và
   manifest giữ nguyên thông tin đó. LLM không là nguồn sự thật — nó chỉ đề
   xuất biến thể trong biên đã hiệu chỉnh.
2. **LLM sinh đặc tả, mã tất định sinh dữ liệu.** LLM trả JSON đặc tả; mọi dòng
   giao dịch do simulator tất định phát ra, nên nhãn đúng theo cấu tạo và kết
   quả tái lập được từ `(spec_id, seed)`.
3. **Cô lập bất định.** Sự không tất định của LLM chỉ tồn tại ở bước sinh đặc
   tả. Từ khi dataset có phiên bản, huấn luyện không gọi API nữa.
4. **Cách ly vận hành.** `src/drre/` và `src/dashboard/` không import gì từ
   `llm_adapter`. Nền tảng vận hành chỉ nạp trọng số.

## 3. Hợp đồng đặc tả

`spec_schema.py` định nghĩa từ vựng đóng và biên tuyệt đối. Một đặc tả bị loại
vì một trong ba lý do, và tỷ lệ bị loại được ghi vào manifest như chỉ báo độ ổn
định của bộ sinh:

| Mã lý do | Nghĩa |
|---|---|
| `closed_vocabulary` | `archetype` / `channel` / `day_part` ngoài danh sách cho phép |
| `out_of_bounds` | tham số số học ngoài biên hiệu chỉnh hoặc biên tuyệt đối |
| `schema` | thiếu khóa bắt buộc, hoặc có khóa không thuộc mẫu |
| `parse` | model trả về không phải JSON hợp lệ |

Bốn kiểu hình hiện có: `money_mule_chain`, `account_takeover`,
`investment_romance_scam`, `smurfing_structuring`.

## 4. Quyền riêng tư của đường gọi LLM

Dữ liệu giao dịch thật **không** được gửi cho nhà cung cấp LLM. Prompt chỉ chứa
tên cột của schema, tên kiểu hình, từ vựng đóng và biên tham số tổng hợp.
`llm_adapter.assert_prompt_free_of_raw_data()` đối chiếu prompt với chính dữ
liệu thật và **chặn** nếu phát hiện định danh hay số tiền thật; hàm này được gọi
trước mỗi lần gửi. Khóa API chỉ lấy từ biến môi trường do người dùng tự đặt;
manifest ghi **tên biến môi trường**, không bao giờ ghi giá trị khóa.

Transport nào không thực sự gọi model sẽ khai `is_model_backed = False` và
`SpecGenerator` tự gán `origin = "rule"`. Nhờ vậy không thể vô tình dán nhãn
"llm" cho đặc tả không do LLM sinh — điều kiện tiên quyết để so sánh được cấu
hình (ii) với (iii).

## 5. Ba lớp kiểm soát chất lượng

**Lớp 1 — trước khi sinh:** spec validator (mục 3).

**Lớp 2 — sau khi sinh:** kiểm định phân biệt (adversarial validation) trên các
cột **không mang ngữ nghĩa gian lận**, cộng với khoảng cách phân phối (KS và
Jensen–Shannon). Phép đo chính là *dòng gian lận tổng hợp* so với *dòng gian lận
thật* — so với toàn bộ dữ liệu nền thì AUC cao là đương nhiên và vô nghĩa.

Việc chọn cột nào là "trung tính" quyết định phép kiểm có ý nghĩa hay không, và
hai cột đã bị loại khỏi cổng sau khi đo:

- **Giờ trong ngày**: `day_part` là lựa chọn *có ngữ nghĩa* của đặc tả (giao
  dịch rạng sáng là dấu hiệu rủi ro thật). Đo trên synthetic_vn: AUC 0,656 khi
  tính giờ, 0,517 khi bỏ giờ.
- **Phần định trị của log(số tiền)**: là hàm của độ lớn số tiền, nên nó phát
  hiện lại chính việc tăng cường có chủ ý. Đo trên PaySim: 0,645 trong khi phân
  phối phần xu khớp rất tốt (KS 0,046).

Cả hai vẫn được **báo cáo riêng** trong `distances` của manifest, nên không có
gì bị che.

Trạng thái của lớp 2 có **ba** giá trị, không phải hai: `dat`, `khong_dat`, và
`chua_ket_luan` khi mỗi lớp có dưới 50 mẫu. Với nền synthetic_vn (28 dòng gian
lận thật), AUC dao động 0,51–0,63 chỉ vì đổi seed, nên kết quả ở đó là *chưa
kết luận* — phải nói rõ như vậy khi báo cáo, không được trình bày như đã đạt.

**Lớp 3 — chống rò rỉ:** `fraud_scenario_type`, `scenario_code`, `spec_id` là
cột truy vết. `quality.assert_no_trace_leakage()` chặn việc dùng chúng làm đặc
trưng đầu vào. Lưu ý phân biệt: dùng chúng làm **nhãn** cho đầu phân loại tình
huống là hợp lệ và được khuyến khích; dùng làm **đầu vào** là rò rỉ. Việc chia
tập theo nhóm kịch bản do `quality.scenario_group_split()` thực hiện.

## 6. Hai dấu vết nhân tạo đã phát hiện và xử lý

Cả hai do chính lớp kiểm soát 2 tìm ra khi chạy thật, không phải suy đoán:

1. **Độ lượng tử hóa số tiền.** PaySim có 100% số tiền khớp đúng 2 chữ số thập
   phân, còn simulator phát float đầy đủ sau các phép nhân suy giảm → bộ phân
   biệt tách được với **AUC 1,000** chỉ bằng phần thập phân. Xử lý:
   `simulator.infer_amount_decimals()` suy độ lượng tử hóa của nền và simulator
   làm tròn theo.
2. **Từ vựng cột `channel` chưa thống nhất giữa các nguồn.**
   `src/data/schema.py` định nghĩa `Channel` = domestic/international/e_wallet/
   unknown, nhưng loader PaySim ghi thẳng loại giao dịch PaySim (`cash_out`,
   `transfer`, `payment`, `cash_in`, `debit`) vào cột này. Mọi dòng tổng hợp
   mang giá trị không tồn tại trong nền → tách được ngay. Xử lý tạm thời:
   `simulator.channel_override_needed()` phát hiện tình trạng này và kênh của
   dòng tổng hợp được bốc theo phân phối thực nghiệm của gian lận thật; việc
   ghi đè được ghi vào `manifest.notes`. **Đây là cách chữa triệu chứng** — gốc
   vấn đề là cột `channel` cần được chuẩn hóa ở tầng loader, và đó là một quyết
   định riêng vì nó ảnh hưởng tới `channel_encoded` của các checkpoint đã
   huấn luyện.

Kết quả hiện tại trên nền PaySim (1.041 dòng gian lận thật — đủ mẫu để kết
luận): **AUC kiểm định phân biệt 0,560 → ĐẠT**. Khoảng cách phân phối số tiền
KS = 0,239, được báo cáo kèm cảnh báo: bộ sinh theo luật cố tình quét đều ba mức
số tiền để phủ vùng thưa, nên phân phối huấn luyện khác phân phối thực tế — phải
nêu rõ khi báo cáo.

## 7. Giao thức đánh giá và cổng tuyên bố

Ba cấu hình so sánh trên **cùng một lát dữ liệu thực**:

| Cấu hình | Dữ liệu huấn luyện |
|---|---|
| (i) `real_only` | chỉ dữ liệu thực |
| (ii) `rule_augmented` | thực + dữ liệu từ simulator theo luật |
| (iii) `llm_augmented` | thực + dữ liệu từ đặc tả do LLM đề xuất |

Cấu hình (ii) là **đường cơ sở bắt buộc**: nếu thiếu, phần cải thiện của (iii)
so với (i) có thể chỉ do bản thân việc tăng cường dữ liệu.

Chỉ số: PR-AUC, recall tại ngân sách cảnh báo cố định (khớp ràng buộc năng lực
của QAPE), độ hiệu chỉnh (ECE, Brier), recall theo từng kiểu hình, và tỷ lệ
cảnh báo giả **theo dõi riêng** vì dữ liệu tổng hợp có thể làm bộ phát hiện quá
nhạy. Khả năng khái quát hóa đo bằng cách giữ lại một số kiểu hình, không dùng
khi huấn luyện.

**Cổng tuyên bố** (`protocol.CLAIM_GATE`): chỉ tuyên bố "đặc tả do LLM đề xuất
có đóng góp" khi (iii) vượt (ii) trên dữ liệu thực giữ lại, với khoảng tin cậy
95% của hiệu số không chứa 0, trên ít nhất 3 seed. Hàm
`protocol.decide_llm_claim()` trả về phán quyết kèm câu kết luận để đưa thẳng
vào báo cáo.

## 8. Phạm vi và hạn chế

- Giá trị của đặc tả do LLM đề xuất so với mô phỏng theo luật là **giả thuyết
  thực nghiệm, chưa phải kết luận**.
- Vòng khai thác mẫu đối kháng làm dữ liệu huấn luyện phụ thuộc vào mô hình, nên
  **mặc định tắt** và phải tắt trong mọi cấu hình đối chứng. Khi bật,
  `mining.apply_focus()` chỉ được **thu hẹp** biên, không bao giờ mở rộng.
- Dữ liệu tổng hợp chỉ biểu diễn các kiểu hình đã được mô tả trong tri thức
  miền; nó không thay thế việc thu thập trường hợp thực tế mới.
- Biên `chain_length` và `amount_decay` không suy được từ PaySim/IEEE-CIS (hai
  bộ này không có chuỗi mule để đo) nên lấy từ tri thức miền — đây là **tham số
  giả định**, phải nêu rõ khi báo cáo.
- Trên nền có ít gian lận thật (vd. synthetic_vn: 28 dòng), kiểm định phân biệt
  **không kết luận được**.

## 9. Cách chạy

Đường cơ sở theo luật, cấu hình (ii), không cần internet:

```bash
python -m src.augmentation.build --background synthetic_vn --origin rule
```

Đặc tả do LLM đề xuất, cấu hình (iii) — tự đặt khóa trên máy mình trước
(`$env:DEEPSEEK_API_KEY='...'` trong PowerShell), không ghi khóa vào repo:

```bash
python -m src.augmentation.build --background paysim --origin llm --provider deepseek --model deepseek-chat --n-specs 12
```

Chạy đường LLM khi không có mạng (đặc tả sẽ được gán `origin="rule"`, **không**
dùng được cho cấu hình (iii)):

```bash
python -m src.augmentation.build --background paysim --origin llm --offline
```

Mỗi lần chạy tạo một thư mục bất biến `data/synthetic/<version>/` gồm
`rows.parquet`, `specs.json` và `manifest.json`. Manifest ghi seed, phiên bản
prompt, nhà cung cấp, tên biến môi trường chứa khóa, biên hiệu chỉnh kèm nguồn
từng biên, tỷ lệ đặc tả bị loại theo lý do, báo cáo kiểm soát chất lượng, commit
và phiên bản thư viện. `.gitignore` chỉ cho commit `manifest.json` và
`specs.json`, không commit dữ liệu đã sinh.
