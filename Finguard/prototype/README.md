# FinGuard AML — prototype web (Vòng 2, AI-Quantum Challenge 2026)

Web chạy thật trên dữ liệu AMLWorld HI-Medium và các mô hình đội đã huấn luyện ở đợt 2.

| Trang | Chức năng |
|---|---|
| Tổng quan | Bài toán, luồng xử lý, các kết quả chính |
| Hàng đợi cảnh báo | 233.762 giao dịch thật (3 giờ của val HI-Medium), điểm 0–100, 3 mức xử lý, đối chiếu nhãn thật |
| Vì sao đáng ngờ | Đóng góp từng đặc trưng vào điểm GBDT, 3 mô-đun kịch bản, đồ thị đường đi của tiền |
| Phân bổ ca trực (QAPE) | Chọn cảnh báo cho ca trực: đến-trước-làm-trước, tham lam, ILP chính xác, QUBO (ủ mô phỏng, QAOA) |
| Chạy thử mô hình | Chấm lại trực tiếp bằng GBDT 28 + checkpoint DRRE R7; tải tệp của mình lên; tính AUPRC so với nhãn |
| Kết quả đánh giá | Bảng xếp hạng HI-Small/HI-Medium, thang cải tiến R2→R7, P2 không rò rỉ, theo kiểu rửa tiền |
| Dữ liệu & checkpoint | Tải mọi checkpoint R2–R7, GBDT, dữ liệu mẫu, results.json; lệnh tái lập |

## Chạy trên máy

Từ thư mục `Finguard/` (cần môi trường của dự án, hoặc chỉ `pip install -r prototype/requirements.txt`):

```bash
streamlit run prototype/app.py
```

## Gói dữ liệu `prototype/bundle/` (~40 MB)

Dựng lại từ kết quả huấn luyện bằng `python -X utf8 scripts/build_prototype_bundle.py`
(cần `data/processed_hi_medium/` và `reports/r7/leaderboard/.../scores/`). Gói chỉ dùng tập train và
val; **tập test không được đọc**.

- `data/hi_medium_val_window.parquet`: 13/09/2022 09:00–12:00 của val HI-Medium, 28 đặc trưng, nhãn,
  kiểu hình và điểm từng mô hình đã chấm với đủ lịch sử.
- `models/`: `gbdt_28_hi_medium.joblib` (scikit-learn 1.4.2) và checkpoint DRRE `drre_r2…r7_*.pt`.
- `reference.npz`: phân vị điểm trên toàn val (đưa lô mới về cùng thang 0–100), phân vị đặc trưng.
- `results.json`: bảng xếp hạng, P2, đường học, ngưỡng cảnh báo, hiệu chỉnh xác suất.

## Đưa lên Streamlit Community Cloud (miễn phí)

1. Đẩy repo (gồm thư mục `Finguard/prototype/` và `Finguard/src/`) lên GitHub.
2. Vào https://share.streamlit.io → **Create app** → chọn repo, nhánh, và
   **Main file path** = `Finguard/prototype/app.py`.
3. **Advanced settings** → Python **3.10**. Streamlit tự dùng `Finguard/prototype/requirements.txt`
   (cùng thư mục với tệp chính).
4. Deploy (lần đầu ~5–10 phút vì cài torch). Ghi đường dẫn `https://<ten>.streamlit.app` vào báo cáo và
   tài liệu hướng dẫn.

Dữ liệu AMLWorld là dữ liệu mô phỏng công khai của IBM (Kaggle:
`ealtman2019/ibm-transactions-for-anti-money-laundering-aml`), không chứa thông tin khách hàng thật.
