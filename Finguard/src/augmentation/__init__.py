"""Tầng huấn luyện tăng cường dữ liệu bằng kịch bản tổng hợp (Mục 6).

=== TRẠNG THÁI TRIỂN KHAI — ĐỌC TRƯỚC KHI TRÍCH DẪN ===

Tầng này được thiết kế như một MỞ RỘNG của quy trình huấn luyện hiện có.
Trạng thái từng thành phần tính đến lần cập nhật gần nhất của package:

  ĐÃ TRIỂN KHAI VÀ CÓ KIỂM THỬ
    - Calibration        (calibration.py)   — hiệu chỉnh biên tham số từ dữ liệu thực có nhãn
    - Spec generator     (llm_adapter.py)   — adapter độc lập nhà cung cấp, có transport offline
    - Spec validator     (spec_schema.py)   — từ vựng đóng, biên giá trị, khóa ngoài mẫu
    - Simulator          (simulator.py)     — tất định theo seed, gán nhãn theo cấu tạo
    - Injector + QC      (quality.py)       — chèn vào nền, kiểm định phân biệt, khoảng cách phân phối
    - Synthetic dataset  (dataset.py)       — có phiên bản, manifest ghi seed/prompt/nhà cung cấp
    - Adversarial mining (mining.py)        — MẶC ĐỊNH TẮT, chỉ bật khi nghiên cứu riêng

  CHƯA CHẠY ĐỦ (là kiến trúc mục tiêu, không phải kết quả)
    - Giao thức đánh giá 3 cấu hình (protocol.py) đã có mã và chỉ số, nhưng
      KẾT QUẢ so sánh (i)/(ii)/(iii) trên nhiều seed CHƯA được chạy xong.
      Không được tuyên bố LLM có đóng góp cho tới khi cấu hình (iii) vượt
      cấu hình (ii) có ý nghĩa thống kê — xem protocol.py::CLAIM_GATE.

=== BỐN NGUYÊN TẮC BẤT BIẾN (Mục 6.2) ===

1. Neo thực nghiệm — LLM KHÔNG là nguồn sự thật. Mọi tham số số học phải
   nằm trong biên do calibration.py suy ra từ dữ liệu thực hoặc tri thức
   miền có dẫn nguồn.
2. LLM sinh đặc tả, mã tất định sinh dữ liệu — LLM chỉ trả JSON đặc tả;
   mọi dòng giao dịch do simulator.py sinh ra, nên nhãn đúng theo cấu tạo.
3. Cô lập bất định — sau khi dataset đã có phiên bản, huấn luyện KHÔNG
   gọi API. Không có import nào từ llm_adapter trong src/drre/ hay
   src/dashboard/.
4. Cách ly vận hành — nền tảng vận hành chỉ nạp trọng số, không phụ thuộc
   nhà cung cấp LLM nào.

=== CHỐNG RÒ RỈ ===

`fraud_scenario_type` và `scenario_code` là cột TRUY VẾT, KHÔNG được dùng
làm đặc trưng đầu vào. Lưu ý phân biệt: dùng chúng làm NHÃN cho đầu phân
loại tình huống là hợp lệ; dùng làm ĐẦU VÀO là rò rỉ. Hàm
quality.assert_no_trace_leakage() kiểm tra điều này tự động.
"""

from src.augmentation.spec_schema import (
    ARCHETYPES,
    ScenarioSpec,
    SpecRejection,
    TRACE_ONLY_COLUMNS,
    validate_spec,
)

__all__ = [
    "ARCHETYPES",
    "ScenarioSpec",
    "SpecRejection",
    "TRACE_ONLY_COLUMNS",
    "validate_spec",
]
