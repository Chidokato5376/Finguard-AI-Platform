"""Kho tri thức kiểu hình rửa tiền + truy xuất theo vector (Module 7).

Ba thành phần, dùng được độc lập:
  - `typology_kb`  : 8 kiểu hình rửa tiền dưới dạng VĂN BẢN có cấu trúc, kèm
                     "chữ ký" trên các đặc trưng mà pipeline thật sự tính.
  - `retriever`    : chỉ mục vector (TF-IDF, hoặc embedding nếu môi trường có)
                     + hai đường truy xuất: theo tín hiệu số và theo câu hỏi.
  - `narrative`    : ghép kết quả truy xuất thành lời giải thích cho chuyên
                     viên rủi ro; có đường gọi LLM tùy chọn, mặc định không gọi.

Xem docs/knowledge-rag.md để biết vì sao thiết kế như vậy và giới hạn đã đo.
"""

from src.knowledge.narrative import AlertNarrative, build_narrative
from src.knowledge.retriever import (
    RetrievalHit,
    TypologyRetriever,
    available_encoders,
)
from src.knowledge.typology_kb import (
    TYPOLOGIES,
    SIGNATURE_FEATURES,
    Typology,
    knowledge_documents,
    typology_by_name,
)

__all__ = [
    "AlertNarrative",
    "RetrievalHit",
    "SIGNATURE_FEATURES",
    "TYPOLOGIES",
    "Typology",
    "TypologyRetriever",
    "available_encoders",
    "build_narrative",
    "knowledge_documents",
    "typology_by_name",
]
