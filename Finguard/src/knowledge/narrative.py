"""Sinh lời giải thích cho một cảnh báo — bước "G" của RAG.

=== LUỒNG ĐẦY ĐỦ ===

    đặc trưng của giao dịch (số)
        -> chuẩn hóa z-score theo tập train          [retriever.z_scores]
        -> câu truy vấn tiếng Việt                   [retriever.signals_to_query]
        -> truy xuất kho tri thức bằng vector        [retriever.search / match_by_signals]
        -> ghép thành lời giải thích                 [file này]

=== HAI CAM KẾT ===

1. KHÔNG BỊA. Mọi câu trong lời giải thích đều đến từ một trong hai nguồn:
   tín hiệu ĐÃ ĐO được trên chính giao dịch đó, hoặc văn bản ĐÃ CÓ trong
   `typology_kb`. Không có bước nào sinh ra khẳng định mới.

2. KHÔNG GỬI DỮ LIỆU THẬT RA NGOÀI. Đường LLM (tùy chọn, mặc định tắt) chỉ
   nhận các CỤM TỪ mô tả tín hiệu và văn bản kiểu hình — không số tiền, không
   định danh tài khoản, không dấu thời gian. Cùng cam kết với
   `src/augmentation/llm_adapter.py`.

Lời giải thích luôn nói rõ độ tin cậy của phép khớp kiểu hình. Khi tín hiệu
yếu, câu trả lời đúng là "không khớp kiểu hình nào rõ rệt" chứ không phải
chọn đại kiểu hình có điểm cao nhất.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping

from src.knowledge.retriever import (
    SIGNAL_THRESHOLD,
    RetrievalHit,
    TypologyRetriever,
    _SIGNAL_PHRASES,
    signals_to_query,
)

#: Dưới ngưỡng cosine này thì coi như KHÔNG khớp kiểu hình nào. Đặt ở 0,35
#: vì chữ ký là vector thưa (2-4 chiều khác 0 trên 5 chiều): hai vector ngẫu
#: nhiên trong không gian đó thường cho cosine quanh 0,2-0,3, nên ngưỡng phải
#: nằm trên mức đó mới có nghĩa.
MATCH_THRESHOLD = 0.35


@dataclass(frozen=True)
class AlertNarrative:
    """Lời giải thích cho một cảnh báo, đã tách thành phần để dashboard tự dựng."""

    headline: str
    signals: list[str] = field(default_factory=list)
    typology: str | None = None
    typology_vi: str | None = None
    confidence: float = 0.0
    mechanism: str = ""
    next_steps: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    generator: str = "template"

    def as_text(self) -> str:
        """Lời giải thích dưới dạng một đoạn văn bản liền mạch."""
        lines = [self.headline]
        if self.signals:
            lines.append("Tín hiệu quan sát được: " + "; ".join(self.signals) + ".")
        if self.mechanism:
            lines.append(self.mechanism)
        if self.next_steps:
            lines.append("Cần kiểm tra thêm: " + " ".join(
                f"({i}) {step}" for i, step in enumerate(self.next_steps, start=1)))
        return "\n".join(lines)


def _observed_signals(z_scores: Mapping[str, float]) -> list[str]:
    """Các tín hiệu vượt ngưỡng, diễn giải bằng tiếng Việt, kèm độ lệch chuẩn."""
    out: list[str] = []
    for feature, (high, low) in _SIGNAL_PHRASES.items():
        value = float(z_scores.get(feature, 0.0))
        if value >= SIGNAL_THRESHOLD:
            out.append(f"{high} ({value:+.1f} độ lệch chuẩn)")
        elif value <= -SIGNAL_THRESHOLD:
            out.append(f"{low} ({value:+.1f} độ lệch chuẩn)")
    return out


def build_narrative(
    row: Mapping[str, float],
    retriever: TypologyRetriever,
    risk_score: float | None = None,
    llm: Callable[[str], str] | None = None,
) -> AlertNarrative:
    """Dựng lời giải thích cho một giao dịch.

    Args:
        row: đặc trưng của giao dịch (đã qua `add_causal_features`).
        retriever: đã gọi `fit_reference(train_df)`.
        risk_score: điểm rủi ro của mô hình, nếu có — chỉ dùng để mở đầu câu.
        llm: hàm nhận prompt, trả về văn bản. Mặc định None = dùng mẫu câu,
            không gọi mạng. Prompt KHÔNG chứa dữ liệu giao dịch thật.

    Returns:
        `AlertNarrative`. Khi không khớp kiểu hình nào trên `MATCH_THRESHOLD`,
        trường `typology` là None và `headline` nói rõ điều đó.
    """
    z_scores = retriever.z_scores(row)
    signals = _observed_signals(z_scores)

    # Cổng HIỂN THỊ: không có đặc trưng mạng lưới nào ở đuôi cực trị thì
    # không nói gì về kiểu hình. Không có cổng này, panel kiểu hình nổ với
    # 74,7% giao dịch bình thường (đo trên val AMLWorld) và thành nhiễu.
    # Cổng nằm ở ĐÂY chứ không nằm trong match_by_signals: xếp hạng luôn trả
    # lời được, chỉ việc hiển thị mới cần dè dặt.
    if not retriever.is_unusual(row):
        return AlertNarrative(
            headline=(
                (f"Điểm rủi ro {risk_score:.0f}/100. " if risk_score is not None else "")
                + "Không có tín hiệu mạng lưới nào lệch đủ mạnh để quy về một kiểu "
                  "hình rửa tiền."
            ),
            signals=signals,
            generator="template",
        )

    by_signal = retriever.match_by_signals(row, k=3)
    query = signals_to_query(z_scores)
    by_text: list[RetrievalHit] = retriever.search(query, k=3) if query else []

    best = by_signal[0] if by_signal else None
    prefix = f"Điểm rủi ro {risk_score:.0f}/100. " if risk_score is not None else ""

    if best is None or best.score < MATCH_THRESHOLD:
        return AlertNarrative(
            headline=(
                prefix + "Không khớp rõ kiểu hình rửa tiền nào trong kho tri thức"
                + (f" (khớp cao nhất {best.name} chỉ đạt {best.score:.2f}, dưới ngưỡng "
                   f"{MATCH_THRESHOLD})." if best else ", vì giao dịch không có tín hiệu "
                   "mạng lưới nào lệch đáng kể.")
            ),
            signals=signals,
            sources=[hit.name for hit in by_text],
            generator="template",
        )

    typology = best.typology
    headline = (
        prefix + f"Giao dịch khớp kiểu hình {typology.name} ({typology.vi_name}), "
        f"độ khớp {best.score:.2f}."
    )
    if not typology.distinguishable:
        headline += (
            f" Lưu ý: {typology.name} không phân biệt được chắc chắn bằng đặc trưng "
            "của một giao dịch đơn lẻ — cần nhìn cả đồ thị nhiều bước để khẳng định."
        )

    narrative = AlertNarrative(
        headline=headline,
        signals=signals,
        typology=typology.name,
        typology_vi=typology.vi_name,
        confidence=best.score,
        mechanism=typology.mechanism,
        next_steps=list(typology.red_flags),
        sources=[hit.name for hit in (by_text or by_signal)],
        generator="template",
    )

    if llm is None:
        return narrative

    # Đường LLM: prompt chỉ gồm CỤM TỪ tín hiệu và văn bản kiểu hình đã có
    # sẵn trong kho — không định danh, không số tiền, không dấu thời gian.
    prompt = (
        "Bạn là chuyên viên phòng chống rửa tiền. Viết 3-4 câu tiếng Việt giải "
        "thích cho đồng nghiệp, CHỈ dựa trên thông tin dưới đây, không thêm "
        "khẳng định mới.\n\n"
        f"Kiểu hình nghi ngờ: {typology.name} ({typology.vi_name})\n"
        f"Cơ chế: {typology.mechanism}\n"
        f"Tín hiệu quan sát được: {'; '.join(signals) if signals else 'không có'}\n"
        f"Dấu hiệu cần kiểm: {' '.join(typology.red_flags)}\n"
    )
    try:
        text = llm(prompt).strip()
    except Exception:  # pragma: no cover - phụ thuộc mạng
        return narrative
    if not text:
        return narrative
    return AlertNarrative(
        headline=text,
        signals=narrative.signals,
        typology=narrative.typology,
        typology_vi=narrative.typology_vi,
        confidence=narrative.confidence,
        mechanism=narrative.mechanism,
        next_steps=narrative.next_steps,
        sources=narrative.sources,
        generator="llm",
    )
