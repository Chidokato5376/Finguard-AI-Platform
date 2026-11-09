"""Truy xuất kiểu hình rửa tiền — hai đường, dùng được độc lập.

=== HAI ĐƯỜNG TRUY XUẤT, VÀ VÌ SAO CẦN CẢ HAI ===

1. `match_by_signals` — từ SỐ. Lấy vector đặc trưng đã chuẩn hóa z-score của
   một giao dịch, so cosine với "chữ ký" của từng kiểu hình. Đây là đường
   chạy tự động cho mọi cảnh báo: không cần ai gõ câu hỏi nào.

2. `search` — từ CHỮ. Lập chỉ mục vector trên văn bản mô tả 8 kiểu hình, rồi
   truy xuất theo một câu hỏi tiếng Việt. Đây là đường cho chuyên viên tra
   cứu ("gom tiền từ nhiều người rồi chuyển đi ngay là kiểu gì").

Hai đường bổ cho nhau: đường số không trả lời được câu hỏi mở, đường chữ
không tự chạy được. `signals_to_query` nối chúng lại — biến tín hiệu số của
một giao dịch thành câu truy vấn tiếng Việt để chạy qua đường chữ.

=== VỀ BỘ MÃ HÓA VECTOR ===

Mặc định là TF-IDF ở mức ký tự (`char_wb`, 3-5 gram). Chọn như vậy có lý do:
kho tri thức chỉ có 8 văn bản nên mô hình ngôn ngữ lớn không mang lại gì
thêm; TF-IDF ký tự không cần tách từ tiếng Việt, không cần tải model, chạy
được offline trên máy yếu, và kết quả tái lập 100% giữa các lần chạy.

Nếu môi trường có `sentence-transformers` thì `encoder="embedding"` dùng nó.
KHÔNG cài mặc định: nó kéo theo ~90 MB trọng số phải tải về, trong khi lợi
ích trên 8 văn bản chưa được đo. Muốn dùng thì `available_encoders()` cho
biết có sẵn hay chưa.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from src.knowledge.typology_kb import (
    SIGNATURE_FEATURES,
    TYPOLOGIES,
    Typology,
    knowledge_documents,
)

logger = logging.getLogger(__name__)

#: Ngưỡng |z| để coi một tín hiệu là "đáng nói" khi DIỄN ĐẠT thành câu chữ.
#: 1,0 độ lệch chuẩn — đủ để lọc nhiễu, đủ thấp để không bỏ sót.
SIGNAL_THRESHOLD = 1.0

#: Phần đuôi mỗi phía của phân phối tham chiếu được coi là "bất thường" khi
#: QUYẾT ĐỊNH có khớp kiểu hình hay không.
#:
#: VÌ SAO KHÔNG DÙNG z-score CHO VIỆC NÀY: các đặc trưng đếm ở đây lệch rất
#: mạnh. Đo trên val AMLWorld, `log_n_prior_as_cp` có |z| >= 1 ở 68% số dòng
#: — tức ngưỡng z biến "bất thường" thành "đa số". Hệ quả đo được: 74,7%
#: giao dịch BÌNH THƯỜNG được gán một kiểu hình, so với 82,9% giao dịch rửa
#: tiền. Panel kiểu hình trên dashboard khi đó nổ với 3/4 giao dịch và không
#: còn mang thông tin gì.
#:
#: Phân vị thì miễn nhiễm với độ lệch của phân phối: đặt 0,01 nghĩa là mỗi
#: đặc trưng chỉ "kêu" ở 2% số dòng cực trị của chính nó, bất kể hình dạng
#: phân phối ra sao.
TAIL_QUANTILE = 0.01

#: Diễn giải tiếng Việt cho từng đặc trưng chữ ký, theo hướng cao/thấp.
#: Dùng cho `signals_to_query` và cho phần giải thích trên dashboard.
_SIGNAL_PHRASES: dict[str, tuple[str, str]] = {
    "log_n_distinct_cp": (
        "gửi tiền cho rất nhiều đối tác khác nhau",
        "chỉ giao dịch với một vài đối tác quen",
    ),
    "log_n_distinct_senders": (
        "nhận tiền từ rất nhiều người gửi khác nhau",
        "rất ít người gửi tiền tới tài khoản này",
    ),
    "log_n_prior_as_cp": (
        "tài khoản nhận đã nhận tiền nhiều lần trước đó",
        "tài khoản nhận gần như chưa từng nhận tiền",
    ),
    "new_counterparty": (
        "cặp gửi - nhận này giao dịch với nhau lần đầu",
        "cặp gửi - nhận này đã giao dịch quen thuộc",
    ),
    "log_n_prior": (
        "tài khoản đã có lịch sử giao dịch dày",
        "tài khoản gần như mới, hầu như chưa giao dịch",
    ),
}


@dataclass(frozen=True)
class RetrievalHit:
    """Một kết quả truy xuất."""

    name: str
    score: float
    typology: Typology


def available_encoders() -> dict[str, bool]:
    """Bộ mã hóa nào dùng được trong môi trường hiện tại."""
    try:  # pragma: no cover - phụ thuộc môi trường
        import sentence_transformers  # noqa: F401

        has_embedding = True
    except Exception:
        has_embedding = False
    return {"tfidf": True, "embedding": has_embedding}


def signals_to_query(z_scores: Mapping[str, float]) -> str:
    """Biến tín hiệu số của một giao dịch thành câu truy vấn tiếng Việt.

    Chỉ lấy các đặc trưng lệch quá `SIGNAL_THRESHOLD` độ lệch chuẩn — giao
    dịch bình thường sẽ cho câu rỗng, và đó là câu trả lời đúng (không có gì
    để truy xuất).
    """
    parts: list[str] = []
    for feature, (high, low) in _SIGNAL_PHRASES.items():
        value = float(z_scores.get(feature, 0.0))
        if value >= SIGNAL_THRESHOLD:
            parts.append(high)
        elif value <= -SIGNAL_THRESHOLD:
            parts.append(low)
    return ", ".join(parts)


class TypologyRetriever:
    """Chỉ mục truy xuất trên kho tri thức kiểu hình.

    Args:
        encoder: "tfidf" (mặc định), "embedding", hoặc "auto" — auto dùng
            embedding nếu môi trường có, ngược lại rơi về tfidf.

    Ví dụ:
        >>> r = TypologyRetriever()
        >>> hits = r.search("nhiều người gửi tiền về một tài khoản rồi chuyển đi ngay")
        >>> hits[0].name
        'FAN-IN'
    """

    def __init__(self, encoder: str = "tfidf") -> None:
        if encoder == "auto":
            encoder = "embedding" if available_encoders()["embedding"] else "tfidf"
        if encoder not in ("tfidf", "embedding"):
            raise ValueError(f"encoder phải là 'tfidf' | 'embedding' | 'auto', nhận {encoder!r}")
        self.encoder = encoder
        self._names = [name for name, _ in knowledge_documents()]
        documents = [text for _, text in knowledge_documents()]

        if encoder == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer

            # char_wb 3-5 gram: không cần tách từ tiếng Việt, và bắt được cả
            # biến thể hình thái ("chuyển"/"chuyển khoản") mà tách từ bỏ lỡ.
            self._vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
            self._matrix = self._vectorizer.fit_transform(documents)
        else:  # pragma: no cover - cần gói tùy chọn
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
            self._matrix = np.asarray(self._model.encode(documents, normalize_embeddings=True))

        self._reference: dict[str, tuple[float, float]] | None = None
        self._bounds: dict[str, tuple[float, float]] | None = None
        self._signature_matrix = np.array(
            [[t.signature.get(f, 0.0) for f in SIGNATURE_FEATURES] for t in TYPOLOGIES],
            dtype=float,
        )
        # Bộ phân loại kiểu hình HỌC từ train (tùy chọn, xem fit_supervised).
        self._classifier = None
        self._classifier_features: list[str] = []

    # -- đường CHỮ ------------------------------------------------------

    def search(self, query: str, k: int = 3) -> list[RetrievalHit]:
        """Truy xuất kiểu hình theo một câu hỏi tiếng Việt."""
        if not query.strip():
            return []
        if self.encoder == "tfidf":
            from sklearn.metrics.pairwise import cosine_similarity

            similarity = cosine_similarity(
                self._vectorizer.transform([query]), self._matrix).ravel()
        else:  # pragma: no cover - cần gói tùy chọn
            vector = np.asarray(self._model.encode([query], normalize_embeddings=True))
            similarity = (self._matrix @ vector.T).ravel()
        order = np.argsort(-similarity)[:k]
        return [
            RetrievalHit(self._names[i], float(similarity[i]), TYPOLOGIES[i])
            for i in order
        ]

    # -- đường SỐ -------------------------------------------------------

    def fit_reference(
        self, frame, features: Sequence[str] = SIGNATURE_FEATURES, tail: float = TAIL_QUANTILE,
    ) -> "TypologyRetriever":
        """Ghi nhớ trung bình/độ lệch chuẩn của tập THAM CHIẾU để chuẩn hóa z-score.

        Phải fit trên tập TRAIN, đúng như mọi bước chuẩn hóa khác của dự án:
        fit trên tập đang chấm là để phân phối của tập đó lọt vào bước chuẩn
        hóa (xem src/data/preprocessing.py).
        """
        reference: dict[str, tuple[float, float]] = {}
        bounds: dict[str, tuple[float, float]] = {}
        for feature in features:
            if feature not in frame.columns:
                continue
            values = frame[feature].astype(float).to_numpy()
            std = float(np.nanstd(values))
            reference[feature] = (float(np.nanmean(values)), std if std > 1e-9 else 1.0)
            bounds[feature] = (
                float(np.nanquantile(values, tail)),
                float(np.nanquantile(values, 1.0 - tail)),
            )
        if not reference:
            raise ValueError(
                f"Khung dữ liệu không có đặc trưng chữ ký nào trong {list(features)}. "
                "Chạy src.data.features.add_causal_features trước."
            )
        self._reference = reference
        self._bounds = bounds
        logger.info(
            "Đã fit tham chiếu trên %d đặc trưng chữ ký (đuôi %.0f%% mỗi phía)",
            len(reference), 100 * tail,
        )
        return self

    def is_unusual(self, row: Mapping[str, float]) -> bool:
        """Giao dịch có ÍT NHẤT MỘT đặc trưng chữ ký nằm ngoài đuôi phân vị chưa.

        Đây là cổng quyết định CÓ ĐÁNG HIỂN THỊ kiểu hình hay không — một câu
        hỏi khác hẳn với "kiểu hình nào giống nhất". `match_by_signals` luôn
        trả về thứ hạng (nó là phép xếp hạng, luôn trả lời được);
        `build_narrative` mới gọi hàm này để quyết định có nói gì không.

        Tách hai việc này là có lý do đã đo: nhét cổng vào thẳng trong
        `match_by_signals` làm độ chính xác top-3 của đường chữ ký rơi từ
        46,4% xuống 10,8%, vì 4/5 giao dịch rửa tiền không có đặc trưng nào ở
        đuôi cực trị và bị tính là trượt — trong khi thứ hạng của chúng vẫn
        đúng. Cổng chỉ nên chặn việc HIỂN THỊ.

        Đo trên val AMLWorld với `TAIL_QUANTILE = 0,01`: cổng mở cho 18,9%
        giao dịch bình thường và 21,3% giao dịch rửa tiền. Hai con số này gần
        nhau, nên cổng KHÔNG phải một bộ phát hiện — nó chỉ để panel kiểu
        hình không nổ với 3/4 số giao dịch (số đo cũ khi dùng ngưỡng z: 74,7%
        và 82,9%).
        """
        if self._bounds is None:
            raise RuntimeError("Phải gọi fit_reference(train_df) trước.")
        for feature, (low, high) in self._bounds.items():
            value = row.get(feature)
            if value is None:
                continue
            value = float(value)
            if np.isnan(value):
                continue
            if value < low or value > high:
                return True
        return False

    def z_scores(self, row: Mapping[str, float]) -> dict[str, float]:
        """Chuẩn hóa một dòng về z-score theo tập tham chiếu đã fit."""
        if self._reference is None:
            raise RuntimeError("Phải gọi fit_reference(train_df) trước khi chuẩn hóa.")
        out: dict[str, float] = {}
        for feature, (mean, std) in self._reference.items():
            value = row.get(feature)
            if value is None or (isinstance(value, float) and np.isnan(value)):
                continue
            out[feature] = (float(value) - mean) / std
        return out

    def fit_supervised(
        self, frame, pattern_column: str = "laundering_pattern",
        features: Sequence[str] | None = None,
    ) -> "TypologyRetriever":
        """Học bộ phân loại kiểu hình từ tập TRAIN, thay cho chữ ký viết tay.

        VÌ SAO CẦN: chữ ký viết tay chỉ mã hóa được hình dạng MỘT BƯỚC. Đo
        trên tập val AMLWorld (461 giao dịch có nhãn kiểu hình thật), nó đạt
        FAN-IN 72,7% và FAN-OUT 61,0% top-1 — nhưng STACK 1,0%, CYCLE 2,0%,
        GATHER-SCATTER và BIPARTITE 0,0%, tổng thể 15,2% top-1, THUA cả mốc
        "luôn đoán kiểu phổ biến nhất" (22,8%). Bốn kiểu hình thất bại đều
        cần ngữ cảnh đồ thị nhiều bước, thứ mà đặc trưng của một giao dịch
        đơn lẻ không mang.

        ĐO ĐƯỢC GÌ SAU KHI HỌC (cùng 461 giao dịch val, học trên 1.529 dòng
        train có nhãn kiểu hình):

            Kiểu hình        n   | chữ ký tay top-1/3 | học top-1/3
            STACK          105   |   1,0% /  29,5%    |  32,4% / 76,2%
            SCATTER-GATHER  93   |  11,8% /  83,9%    |  18,3% / 76,3%
            GATHER-SCATTER  70   |   0,0% /  25,7%    |   5,7% / 42,9%
            CYCLE           51   |   0,0% /   9,8%    |  29,4% / 54,9%
            FAN-IN          44   |  72,7% /  90,9%    |  54,5% / 70,5%
            FAN-OUT         41   |  61,0% /  82,9%    |  53,7% / 82,9%
            BIPARTITE       35   |   0,0% /   2,9%    |  22,9% / 65,7%
            RANDOM          22   |   0,0% /  31,8%    |   4,5% / 50,0%
            ----------------------------------------------------------
            TỔNG           461   |  15,0% /  46,4%    |  27,1% / 66,8%
            mốc đoán bừa 1/8     |  12,5% /  37,5%    |
            mốc luôn đoán STACK  |  22,8%             |

        Đọc bảng này cho đúng: đường học thắng tổng thể và vượt được cả mốc
        "luôn đoán kiểu phổ biến nhất", nhưng nó THUA chữ ký tay ở đúng hai
        kiểu hình một bước (FAN-IN, FAN-OUT). Vì vậy chữ ký viết tay KHÔNG bị
        bỏ: nó vẫn là đường chạy khi chưa có nhãn kiểu hình nào (bộ dữ liệu
        mới, hoặc ngân hàng thật chưa gán nhãn typology), nó đọc được bằng
        mắt, và nó mạnh hơn ở hai kiểu hình đó.

        Cả hai đường đều gần như bất lực với GATHER-SCATTER (5,7%) và RANDOM
        (4,5%) — đúng như `Typology.distinguishable=False` đã khai báo trước
        khi đo. Con số dùng được trong thực tế là TOP-3 66,8%: đưa cho chuyên
        viên ba kiểu hình khả dĩ thì hai phần ba số lần có kiểu đúng trong đó.

        Args:
            frame: tập train đã có cột `pattern_column` và các đặc trưng.
            pattern_column: cột nhãn kiểu hình (xem
                `src.data.loaders.attach_laundering_pattern`).
            features: đặc trưng dùng cho bộ phân loại. Mặc định lấy mọi đặc
                trưng nhân quả có trong khung — nhiều hơn SIGNATURE_FEATURES,
                vì ở đây không cần đọc bằng mắt mà cần chính xác.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        if pattern_column not in frame.columns:
            raise KeyError(
                f"Không có cột {pattern_column!r}. Chạy tiền xử lý với --patterns-path "
                "để sinh nhãn kiểu hình."
            )
        if features is None:
            from src.data.features import ALL_FEATURES

            features = [c for c in ALL_FEATURES if c in frame.columns]
        features = list(features)

        labels = frame[pattern_column].astype(object)
        usable = labels.notna() & (labels != "UNSPECIFIED")
        if int(usable.sum()) < 50:
            raise ValueError(
                f"Chỉ có {int(usable.sum())} dòng có nhãn kiểu hình — quá ít để học. "
                "Cần ít nhất 50."
            )
        x = frame.loc[usable, features].astype(float).to_numpy()
        y = labels[usable].to_numpy()

        model = make_pipeline(
            StandardScaler(),
            # Không truyền multi_class: với solver lbfgs, sklearn tự dùng
            # multinomial, và tham số đó đã bị khai tử ở các bản mới.
            LogisticRegression(max_iter=2000, class_weight="balanced"),
        )
        model.fit(x, y)
        self._classifier = model
        self._classifier_features = features
        logger.info(
            "Đã học bộ phân loại kiểu hình trên %d dòng, %d đặc trưng, %d lớp: %s",
            int(usable.sum()), len(features), len(model.classes_), list(model.classes_),
        )
        return self

    def match_by_signals(self, row: Mapping[str, float], k: int = 3) -> list[RetrievalHit]:
        """Khớp một giao dịch với kiểu hình.

        Dùng bộ phân loại đã học nếu có (`fit_supervised`), ngược lại dùng
        cosine giữa z-score và chữ ký viết tay. Điểm trả về là xác suất ở
        đường thứ nhất và cosine ở đường thứ hai — hai thang khác nhau, nên
        `MATCH_THRESHOLD` trong narrative.py được đặt cho thang cosine và
        đường học máy luôn vượt nó khi xác suất đủ cao.

        Kiểu hình có chữ ký rỗng (RANDOM) luôn nhận điểm 0 ở đường chữ ký —
        theo định nghĩa nó không có hình dạng để khớp, và trả về một con số
        giả ở đó sẽ làm bảng đánh giá trông tốt hơn thực tế.
        """
        if self._classifier is not None:
            vector = np.array(
                [[float(row.get(f, 0.0) or 0.0) for f in self._classifier_features]],
                dtype=float,
            )
            probabilities = self._classifier.predict_proba(vector).ravel()
            order = np.argsort(-probabilities)[:k]
            by_name = {t.name: t for t in TYPOLOGIES}
            return [
                RetrievalHit(
                    str(self._classifier.classes_[i]),
                    float(probabilities[i]),
                    by_name[str(self._classifier.classes_[i])],
                )
                for i in order
            ]

        z = self.z_scores(row)
        vector = np.array([z.get(f, 0.0) for f in SIGNATURE_FEATURES], dtype=float)
        norm = float(np.linalg.norm(vector))
        if norm < 1e-9:
            return []
        signature_norms = np.linalg.norm(self._signature_matrix, axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            similarity = np.where(
                signature_norms > 1e-9,
                (self._signature_matrix @ vector) / (signature_norms * norm),
                0.0,
            )
        order = np.argsort(-similarity)[:k]
        return [
            RetrievalHit(TYPOLOGIES[i].name, float(similarity[i]), TYPOLOGIES[i])
            for i in order
        ]
