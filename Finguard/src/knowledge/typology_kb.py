"""Kho tri thức 8 kiểu hình rửa tiền — văn bản + chữ ký trên đặc trưng.

=== VÌ SAO CÓ FILE NÀY ===

Mô hình chấm ra một con số. Con số đó không trả lời được câu hỏi mà chuyên
viên rủi ro thật sự hỏi: "giao dịch này giống kiểu rửa tiền nào, và tôi phải
tìm thêm bằng chứng gì". Kho này nối hai thứ đó lại: mỗi kiểu hình được mô tả
bằng VĂN BẢN (để truy xuất theo ngữ nghĩa) và bằng CHỮ KÝ SỐ (để khớp với
đúng những đặc trưng pipeline đang tính).

=== HAI GIỚI HẠN PHẢI NÓI TRƯỚC ===

1. Chữ ký được định nghĩa trên các đặc trưng CỦA MỘT GIAO DỊCH. Có những
   kiểu hình chỉ phân biệt được khi nhìn cả đồ thị nhiều bước, nên với bộ
   đặc trưng hiện tại:
     - GATHER-SCATTER và SCATTER-GATHER có chữ ký GẦN NHƯ TRÙNG NHAU (cả hai
       đều là "vừa nhiều người gửi tới, vừa gửi đi nhiều nơi"); thứ phân biệt
       chúng là THỨ TỰ, mà một dòng giao dịch đơn lẻ không mang thông tin đó.
     - RANDOM theo định nghĩa không có hình dạng, nên nó là kiểu hình mà chữ
       ký không giúp gì được.
   Những giới hạn này được ĐO trong tests/test_knowledge.py chứ không chỉ
   ghi trong tài liệu.

2. Mô tả văn bản là kiến thức nghiệp vụ phổ thông về các kiểu hình rửa tiền,
   được viết lại cho dự án này. Nó KHÔNG phải trích dẫn từ tài liệu pháp lý
   và không được dùng làm căn cứ tuân thủ.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: Các đặc trưng mà chữ ký kiểu hình được định nghĩa trên đó. Đây là tập con
#: của `src.data.features.ALL_FEATURES` — cố ý chỉ lấy nhóm ĐỐI TÁC và nhóm
#: LỊCH SỬ, vì chỉ hai nhóm đó mang thông tin về hình dạng mạng lưới. Số tiền
#: và đồng hồ không nói gì về hình dạng, nên không nằm trong chữ ký.
SIGNATURE_FEATURES: tuple[str, ...] = (
    "log_n_distinct_cp",       # tài khoản này đã gửi cho bao nhiêu đối tác khác nhau
    "log_n_distinct_senders",  # đã có bao nhiêu người gửi khác nhau tới đối tác
    "log_n_prior_as_cp",       # đối tác đã nhận tiền bao nhiêu lần trước đó
    "new_counterparty",        # cặp (gửi, nhận) này xuất hiện lần đầu
    "log_n_prior",             # tài khoản này đã có bao nhiêu giao dịch trước đó
)


@dataclass(frozen=True)
class Typology:
    """Một kiểu hình rửa tiền.

    Attributes:
        name: Tên như AMLWorld gắn nhãn (khớp `AMLWORLD_TYPOLOGIES`).
        vi_name: Tên tiếng Việt dùng trên dashboard.
        summary: Một câu — cái gì đang xảy ra.
        mechanism: Cơ chế, 2-4 câu, viết cho người không chuyên kỹ thuật.
        red_flags: Dấu hiệu chuyên viên nên tìm thêm. Mỗi mục là một việc
            KIỂM TRA ĐƯỢC, không phải một tính từ.
        signature: Hướng kỳ vọng của từng đặc trưng trong SIGNATURE_FEATURES,
            theo thang z-score: +1 = cao hơn hẳn mức thường, -1 = thấp hơn
            hẳn, 0 = không nói gì. Đặc trưng không có mặt coi như 0.
        distinguishable: False nếu kiểu hình này KHÔNG phân biệt được bằng
            đặc trưng một-giao-dịch (xem giới hạn 1 ở đầu file).
    """

    name: str
    vi_name: str
    summary: str
    mechanism: str
    red_flags: tuple[str, ...]
    signature: dict[str, float] = field(default_factory=dict)
    distinguishable: bool = True

    def as_document(self) -> str:
        """Kiểu hình dưới dạng một đoạn văn bản — đơn vị để lập chỉ mục vector."""
        flags = " ".join(self.red_flags)
        return f"{self.name}. {self.vi_name}. {self.summary} {self.mechanism} {flags}"


TYPOLOGIES: tuple[Typology, ...] = (
    Typology(
        name="FAN-OUT",
        vi_name="Phân tán ra nhiều tài khoản",
        summary="Một tài khoản chia tiền ra rất nhiều tài khoản nhận khác nhau "
                "trong thời gian ngắn.",
        mechanism="Tiền bẩn nằm ở một chỗ thì dễ bị thu giữ, nên bước đầu tiên "
                  "thường là bẻ nhỏ. Một tài khoản nguồn chuyển cho hàng chục "
                  "tài khoản nhận, mỗi khoản đủ nhỏ để không chạm ngưỡng báo "
                  "cáo. Các tài khoản nhận thường mới mở và trước đó không có "
                  "quan hệ gì với tài khoản nguồn.",
        red_flags=(
            "Số đối tác nhận khác nhau tăng vọt so với lịch sử của chính tài khoản đó.",
            "Phần lớn các cặp gửi-nhận đều là lần đầu giao dịch với nhau.",
            "Các khoản chia có giá trị gần bằng nhau và nằm ngay dưới ngưỡng báo cáo.",
            "Tài khoản nguồn vừa nhận một khoản lớn ngay trước đợt chia.",
        ),
        signature={
            "log_n_distinct_cp": 1.0,
            "new_counterparty": 1.0,
            "log_n_prior": 0.5,
            "log_n_distinct_senders": -0.5,
        },
    ),
    Typology(
        name="FAN-IN",
        vi_name="Gom về một tài khoản",
        summary="Rất nhiều tài khoản khác nhau cùng chuyển tiền về một tài khoản đích.",
        mechanism="Đây là chiều ngược của phân tán: sau khi tiền đã được bẻ nhỏ "
                  "và đi qua nhiều tay, nó phải được gom lại để rút hoặc để "
                  "chuyển tiếp. Tài khoản đích thường là tài khoản trung gian "
                  "(money mule) — nhận từ nhiều nguồn không liên quan đến nhau, "
                  "rồi gần như lập tức chuyển đi.",
        red_flags=(
            "Số người gửi khác nhau tới cùng một tài khoản cao bất thường.",
            "Những người gửi không có quan hệ kinh doanh nào với nhau.",
            "Tài khoản đích chuyển tiền đi gần hết ngay sau khi nhận.",
            "Tài khoản đích mới mở và chưa từng có lịch sử giao dịch đều đặn.",
        ),
        signature={
            "log_n_distinct_senders": 1.0,
            "log_n_prior_as_cp": 1.0,
            "log_n_distinct_cp": -0.5,
        },
    ),
    Typology(
        name="CYCLE",
        vi_name="Dòng tiền vòng tròn",
        summary="Tiền đi qua một chuỗi tài khoản rồi quay về đúng điểm xuất phát.",
        mechanism="Mục đích là tạo ra lịch sử giao dịch trông hợp lệ. Sau vài "
                  "bước, tiền quay lại tài khoản ban đầu nhưng khi đó đã có một "
                  "chuỗi chứng từ phía sau để giải trình nguồn gốc. Dấu hiệu "
                  "nhận ra là các cặp gửi-nhận LẶP LẠI chứ không mới, và tổng "
                  "tiền gần như không đổi qua vòng.",
        red_flags=(
            "Tài khoản nhận hôm nay đã từng là tài khoản gửi cho chính bên này trước đó.",
            "Chuỗi giao dịch khép kín trong vòng vài ngày.",
            "Số tiền gần như giữ nguyên qua từng bước, chỉ hao phí phí chuyển khoản.",
            "Không có hàng hóa hay dịch vụ nào tương ứng với dòng tiền.",
        ),
        signature={
            "log_n_prior_as_cp": 1.0,
            "new_counterparty": -1.0,
            "log_n_prior": 1.0,
        },
    ),
    Typology(
        name="GATHER-SCATTER",
        vi_name="Gom rồi phân tán",
        summary="Tiền được gom từ nhiều nguồn về một điểm, rồi lại chia ra nhiều đích.",
        mechanism="Kết hợp hai bước: gom (fan-in) rồi phân tán (fan-out) qua "
                  "cùng một tài khoản trung tâm. Tài khoản ở giữa đóng vai trò "
                  "trạm trung chuyển, và chính nó là mắt xích dễ phát hiện nhất "
                  "vì vừa có nhiều người gửi tới vừa gửi đi nhiều nơi.",
        red_flags=(
            "Một tài khoản vừa có nhiều người gửi tới, vừa gửi cho nhiều đối tác.",
            "Tiền vào và tiền ra gần bằng nhau trong một khoảng thời gian ngắn.",
            "Số dư cuối kỳ của tài khoản trung tâm rất nhỏ so với tổng tiền đi qua.",
        ),
        signature={
            "log_n_distinct_senders": 1.0,
            "log_n_distinct_cp": 1.0,
            "log_n_prior": 0.5,
        },
        distinguishable=False,
    ),
    Typology(
        name="SCATTER-GATHER",
        vi_name="Phân tán rồi gom",
        summary="Tiền được chia ra nhiều tài khoản trung gian rồi gom lại ở một đích khác.",
        mechanism="Ngược thứ tự với gom-rồi-phân-tán: chia trước, gom sau. Mục "
                  "đích là cắt đứt liên hệ trực tiếp giữa điểm đầu và điểm cuối "
                  "— không có giao dịch nào nối thẳng hai đầu, nên nhìn từng "
                  "giao dịch riêng lẻ thì không thấy gì bất thường.",
        red_flags=(
            "Nhiều tài khoản trung gian nhận rồi chuyển đi gần như toàn bộ số tiền.",
            "Các tài khoản trung gian chỉ hoạt động trong đúng khoảng thời gian này.",
            "Điểm đầu và điểm cuối không có giao dịch trực tiếp nào với nhau.",
        ),
        signature={
            "log_n_distinct_cp": 1.0,
            "log_n_distinct_senders": 1.0,
            "new_counterparty": 0.5,
        },
        distinguishable=False,
    ),
    Typology(
        name="BIPARTITE",
        vi_name="Hai nhóm giao dịch chéo",
        summary="Một nhóm tài khoản gửi và một nhóm tài khoản nhận, giao dịch chéo "
                "nhau dày đặc.",
        mechanism="Thay vì một trạm trung chuyển duy nhất, mạng lưới dùng hai "
                  "nhóm: mọi tài khoản nhóm A đều có thể gửi cho mọi tài khoản "
                  "nhóm B. Cách này làm loãng dấu vết — không tài khoản nào nổi "
                  "bật, nhưng mật độ kết nối giữa hai nhóm thì cao bất thường so "
                  "với giao dịch bình thường.",
        red_flags=(
            "Hai nhóm tài khoản giao dịch chéo với mật độ cao nhưng không nhóm nào "
            "giao dịch ra ngoài.",
            "Không tài khoản đơn lẻ nào có khối lượng đủ lớn để tự nó bị gắn cờ.",
            "Các tài khoản trong cùng một nhóm được mở gần nhau về thời gian.",
        ),
        signature={
            "log_n_distinct_cp": 1.0,
            "log_n_distinct_senders": 1.0,
            "new_counterparty": -0.5,
            "log_n_prior_as_cp": 0.5,
        },
    ),
    Typology(
        name="STACK",
        vi_name="Xếp tầng nhiều lớp",
        summary="Tiền đi qua một chuỗi dài nhiều lớp tài khoản nối tiếp nhau.",
        mechanism="Mỗi lớp thêm một bước giữa nguồn tiền bẩn và đích cuối. Càng "
                  "nhiều lớp thì việc truy ngược càng tốn thời gian, đặc biệt khi "
                  "các lớp nằm ở ngân hàng hoặc quốc gia khác nhau. Từng giao "
                  "dịch trong chuỗi đều trông bình thường; chỉ chuỗi mới bất thường.",
        red_flags=(
            "Chuỗi chuyển tiếp dài, mỗi tài khoản nhận rồi chuyển đi gần hết.",
            "Thời gian giữ tiền ở mỗi lớp rất ngắn.",
            "Chuỗi đi qua nhiều ngân hàng khác nhau mà không có lý do kinh doanh.",
        ),
        signature={
            "log_n_prior_as_cp": 1.0,
            "log_n_prior": 1.0,
            "log_n_distinct_cp": -0.5,
            "log_n_distinct_senders": -0.5,
        },
    ),
    Typology(
        name="RANDOM",
        vi_name="Không theo hình dạng rõ rệt",
        summary="Giao dịch rửa tiền không tạo thành hình dạng mạng lưới nhận ra được.",
        mechanism="AMLWorld gán nhãn này cho các giao dịch rửa tiền không khớp "
                  "kiểu hình nào. Về mặt phát hiện, đây là nhóm khó nhất: không "
                  "có hình dạng đồ thị để bắt, nên nhánh đồ thị không giúp được "
                  "gì và mô hình phải dựa hoàn toàn vào lệch hành vi của tài khoản.",
        red_flags=(
            "Số tiền lệch mạnh so với lịch sử chi tiêu của chính tài khoản đó.",
            "Khoảng cách thời gian giữa các giao dịch thay đổi đột ngột.",
        ),
        signature={},
        distinguishable=False,
    ),
)


def typology_by_name(name: str) -> Typology:
    """Tra một kiểu hình theo tên AMLWorld. Tên lạ thì báo lỗi kèm danh sách hợp lệ."""
    for item in TYPOLOGIES:
        if item.name == name:
            return item
    raise KeyError(
        f"Không có kiểu hình {name!r}. Hợp lệ: {[t.name for t in TYPOLOGIES]}"
    )


def knowledge_documents() -> list[tuple[str, str]]:
    """Kho tri thức dưới dạng [(tên, văn bản)] — đầu vào cho chỉ mục vector."""
    return [(item.name, item.as_document()) for item in TYPOLOGIES]
