# Data

## Nguồn dữ liệu

| Nguồn | Kích thước | Vai trò | Giấy phép |
|---|---|---|---|
| [PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1) | ~6.3M giao dịch | Behavior Memory + Graph Encoder (chuỗi giao dịch theo tài khoản, quan hệ gửi–nhận) | Public (Kaggle, CC BY-SA) |
| [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection) | ~590K giao dịch | Bổ sung đặc trưng giao dịch cho Behavior Memory, huấn luyện đầu dò bán giám sát ℒ_BCE, tập kiểm định chéo | Public (Kaggle, competition rules) |
| Synthetic VN (tự sinh) | Tùy cấu hình | Kiểm thử concept drift, kịch bản đặc thù VN (deepfake context, chiếm quyền Mobile Banking) | Nội bộ — xem `src/data/synthetic_generator.py` |

## ⚠ Cần xác minh trước khi ingest

PaySim gốc trên Kaggle **không có sẵn** `device_id` / `ip_country` trong schema chuẩn — chỉ có `step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig, nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud`. Nếu pipeline cần các trường thiết bị/IP (dùng trong kịch bản demo §7 của tài liệu đặc tả), chúng phải được sinh bổ sung một cách minh bạch và gắn cờ rõ ràng là dữ liệu tổng hợp — không được trình bày như dữ liệu gốc trước ban giám khảo.

## Unified Schema

Ba nguồn dữ liệu được chuẩn hóa về một schema chung trước khi vào pipeline huấn luyện. Định nghĩa chính thức: `src/data/schema.py` (dataclass `TransactionRecord`).

| Trường | Kiểu | Mô tả |
|---|---|---|
| `account_id` | string | Định danh tài khoản gửi — khóa tích lũy Behavior Memory |
| `counterparty_id` | string | Định danh tài khoản nhận — đỉnh thứ hai trong đồ thị giao dịch |
| `timestamp` / `step` | datetime / int | Trục thời gian cho GRU và decay hành vi |
| `amount` | float | Số tiền giao dịch |
| `channel` | categorical | nội địa / quốc tế / ví điện tử... |
| `device_id` | string | Tín hiệu thiết bị — xem cảnh báo xác minh nguồn ở trên |
| `ip_country` | categorical | Tín hiệu định vị — xem cảnh báo xác minh nguồn ở trên |
| `label` | binary, optional | Nhãn gian lận — chỉ dùng cho fine-tune bán giám sát (ℒ_BCE), không bắt buộc cho ℒ_pred / ℒ_contrast |

## Giới hạn dữ liệu đã biết

- **Class imbalance nghiêm trọng** (tỷ lệ gian lận thường < 1%): xử lý bằng contrastive learning + self-supervised prediction loss làm tín hiệu chính; nhãn chỉ dùng để fine-tune.
- **PaySim là dữ liệu mô phỏng**, không phản ánh 100% hành vi người dùng Việt Nam — bù đắp một phần bằng nhánh synthetic hiệu chỉnh theo thống kê công khai NHNN.
- **IEEE-CIS không có trường counterparty rõ ràng** — chỉ dùng cho Behavior Memory và huấn luyện phụ, không dùng làm nguồn chính cho Graph Encoder.
- **Rủi ro temporal leakage**: bắt buộc time-based split (`config/config.yaml: data.split.method`), không chia ngẫu nhiên.

## Quy trình cập nhật dữ liệu

```bash
# Tải PaySim / IEEE-CIS thủ công vào data/raw/ (yêu cầu Kaggle API + xác thực cuộc thi)
python -m src.data.loaders --dataset paysim --output data/raw/paysim.csv
python -m src.data.preprocessing --config config/config.yaml
```
