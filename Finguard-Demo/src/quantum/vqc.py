"""Variational Quantum Circuit — phương án nâng cao (Mục 4.5).

phi_4(e_t^c) = <psi(e_t^c; theta)| O |psi(e_t^c; theta)>

Đưa một bản nén (PCA) của e_t^c qua mạch tham số hóa (ansatz huấn luyện
được), đo một observable, dùng kết quả làm đặc trưng bổ sung cho Adaptive
Risk Fusion. Huấn luyện đồng thời với phần cổ điển qua TorchConnector
(qiskit-machine-learning) — thư viện xử lý sẵn gradient qua estimator
(parameter-shift/adjoint tùy backend), không cần tự cài đặt thủ công.

=== VỊ TRÍ TRONG KIẾN TRÚC — khác biệt với quantum_kernel.py ===
Đây là "phương án nâng cao" của Mục 4.5, đứng SAU Quantum Kernel
(quantum_kernel.py, "phương án ưu tiên"):
  - Quantum Kernel: KHÔNG có tham số huấn luyện, chỉ dùng làm kernel cho
    SVM/kNN ở tầng phân loại — rẻ, dễ kiểm chứng, ít rủi ro.
  - VQC (file này): CÓ tham số huấn luyện (ansatz), có thể học biểu diễn
    tinh hơn nhưng ĐẮT hơn nhiều (mỗi bước gradient cần nhiều lần gọi
    Estimator — xem GHI CHÚ HIỆU NĂNG trong benchmark_vs_classical) và dễ
    gặp "barren plateau" (gradient biến mất khi mạch/số qubit tăng) —
    rủi ro kỹ thuật cao hơn hẳn, đúng lý do đặc tả gốc xếp đây là lựa
    chọn TÙY CHỌN, chỉ làm nếu còn thời gian sau Tuần 5 (README §7).

⚠ CHỈ BẬT (config.yaml: quantum.vqc.enabled) SAU KHI benchmark_vs_classical()
xác nhận cải thiện AUPRC thực sự so với một MLP cổ điển cùng vai trò —
đúng nguyên tắc bắt buộc đã áp dụng cho Quantum Kernel.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None

try:
    from qiskit.circuit import QuantumCircuit
    from qiskit.circuit.library import RealAmplitudes, ZFeatureMap
    from qiskit.primitives import StatevectorEstimator
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_machine_learning.connectors import TorchConnector
    from qiskit_machine_learning.neural_networks import EstimatorQNN
except ImportError:  # pragma: no cover
    QuantumCircuit = None
    RealAmplitudes = None
    ZFeatureMap = None
    StatevectorEstimator = None
    SparsePauliOp = None
    TorchConnector = None
    EstimatorQNN = None


def _build_vqc_circuit(n_qubits: int, ansatz_reps: int):
    """Dựng mạch: ZFeatureMap (mã hóa dữ liệu, không tham số huấn luyện)
    nối với RealAmplitudes (ansatz, tham số huấn luyện theta).

    Dùng ZFeatureMap (reps=1, mã hóa góc quay Rz theo từng chiều dữ liệu)
    thay vì ZZFeatureMap (như quantum_kernel.py dùng cho kernel) — ZZ
    entangle dữ liệu chéo nhau phù hợp cho kernel similarity, nhưng ở
    đây ansatz RealAmplitudes đã đảm nhiệm việc entangle có tham số học
    được; dùng ZFeatureMap giữ mạch mã hóa đơn giản, nông hơn, giảm rủi
    ro barren plateau khi ghép thêm ansatz phía sau.
    """
    feature_map = ZFeatureMap(feature_dimension=n_qubits, reps=1)
    ansatz = RealAmplitudes(num_qubits=n_qubits, reps=ansatz_reps)

    circuit = QuantumCircuit(n_qubits)
    circuit.compose(feature_map, inplace=True)
    circuit.compose(ansatz, inplace=True)
    return circuit, feature_map, ansatz


class VQCEmbeddingHead(nn.Module if nn is not None else object):
    """Mạch lượng tử tham số hóa — nhận embedding ĐÃ NÉN PCA (n_qubits
    chiều), trả về MỘT giá trị vô hướng phi_4 = <Z_0> trong [-1, 1].

    PCA fit RIÊNG bằng sklearn (ngoài đồ thị tính toán PyTorch), giống
    hệt pattern trong quantum_kernel.py — PCA không cần lan truyền
    gradient qua, chỉ nén chiều một lần trước khi vào mạch.
    """

    def __init__(self, n_qubits: int = 8, ansatz_reps: int = 2) -> None:
        if TorchConnector is None:
            raise ImportError(
                "Cần cả torch VÀ qiskit-machine-learning. Chạy: "
                "pip install torch qiskit-machine-learning"
            )
        super().__init__()
        self.n_qubits = n_qubits

        circuit, feature_map, ansatz = _build_vqc_circuit(n_qubits, ansatz_reps)
        observable = SparsePauliOp("Z" + "I" * (n_qubits - 1))  # đo Pauli-Z trên qubit 0

        qnn = EstimatorQNN(
            circuit=circuit,
            observables=observable,
            input_params=list(feature_map.parameters),
            weight_params=list(ansatz.parameters),
            estimator=StatevectorEstimator(),
            # BẮT BUỘC True: cần gradient chảy NGƯỢC vào embedding cổ điển
            # (input) để huấn luyện đồng thời với phần DRRE phía trước,
            # không chỉ cập nhật tham số ansatz (weight).
            input_gradients=True,
        )
        self.qnn_layer = TorchConnector(qnn)

    def forward(self, x_compressed: "torch.Tensor") -> "torch.Tensor":
        """
        Args:
            x_compressed: (batch, n_qubits) — embedding đã nén PCA đúng
                n_qubits chiều (xem QuantumEnhancedEmbedding.fit_pca).

        Returns:
            (batch, 1) — phi_4, trong [-1, 1].
        """
        return self.qnn_layer(x_compressed)


class QuantumEnhancedEmbedding:
    """Kết hợp PCA nén chiều (cổ điển) + VQCEmbeddingHead (lượng tử) —
    dùng làm phi_4(e_t^c) bổ sung cho Adaptive Risk Fusion (Mục 4.4) khi
    quantum.vqc.enabled = true trong config.yaml.
    """

    def __init__(self, n_qubits: int = 8, ansatz_reps: int = 2, pca_components: int | None = None) -> None:
        if torch is None:
            raise ImportError("torch chưa được cài. Chạy: pip install torch")
        self.n_qubits = n_qubits
        self.pca = PCA(n_components=pca_components or n_qubits)
        self.vqc_head = VQCEmbeddingHead(n_qubits=n_qubits, ansatz_reps=ansatz_reps)
        self._fitted = False

    def fit_pca(self, e: np.ndarray) -> None:
        """Fit PCA trên tập embedding cổ điển trước khi ánh xạ lượng tử."""
        self.pca.fit(e)
        self._fitted = True

    def transform(self, e: np.ndarray) -> "torch.Tensor":
        if not self._fitted:
            raise RuntimeError("Gọi fit_pca() trước khi transform().")
        compressed = self.pca.transform(e)
        return torch.tensor(compressed, dtype=torch.float32)

    def forward(self, e: np.ndarray) -> "torch.Tensor":
        """Tính phi_4 cho một batch embedding cổ điển (numpy)."""
        x = self.transform(e)
        return self.vqc_head(x)


def benchmark_vs_classical(
    e_train: np.ndarray,
    y_train: np.ndarray,
    e_val: np.ndarray,
    y_val: np.ndarray,
    n_qubits: int = 6,
    ansatz_reps: int = 1,
    epochs: int = 15,
    lr: float = 0.1,
) -> dict[str, float]:
    """So sánh AUPRC: đầu đọc VQC (lượng tử, tham số huấn luyện) vs MLP
    cổ điển cùng vai trò (đọc embedding đã nén PCA -> điểm nhị phân).

    ĐÂY LÀ BƯỚC BẮT BUỘC trước khi bật quantum.vqc.enabled trong
    config.yaml — xem cảnh báo đầu file.

    === GHI CHÚ HIỆU NĂNG (đo đạc thực tế trên phần cứng tham chiếu dùng
    để phát triển repo này — không suy đoán) ===
    Mỗi bước gradient của VQC cần Estimator tính kỳ vọng đo NHIỀU LẦN:
    1 lần cho forward, và với mỗi tham số trong weight_params (ansatz) +
    input_params (vì input_gradients=True), cần thêm các lần gọi Estimator
    cho gradient. Hai điểm đo cụ thể (KHÔNG đủ để suy ra quy luật scaling
    tổng quát — chỉ là hai mốc tham khảo cụ thể, xem cảnh báo dưới):
      - n_qubits=4, ansatz_reps=1, epochs=10, n_train=40: ~27s
      - n_qubits=8, ansatz_reps=2, epochs=5,  n_train=20: ~28s
    ⚠ KHÔNG suy ra "thời gian gần như không đổi bất kể quy mô" từ hai
    điểm trên — chúng thay đổi ĐỒNG THỜI nhiều biến (n_qubits, reps,
    epochs, n_train), trùng hợp cho kết quả gần nhau, không chứng minh
    một quy luật scaling cụ thể nào. Khuyến nghị thực dụng: LUÔN đo thử
    với epochs=1 trên tập nhỏ trước, nhân ước lượng lên theo epochs/
    n_train thật cần dùng, thay vì tin vào một công thức ngoại suy.
    
    """
    if torch is None:
        raise ImportError("torch chưa được cài. Chạy: pip install torch")

    from sklearn.metrics import average_precision_score

    pca = PCA(n_components=n_qubits)
    pca.fit(e_train)
    x_train = torch.tensor(pca.transform(e_train), dtype=torch.float32)
    x_val = torch.tensor(pca.transform(e_val), dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)

    loss_fn = nn.BCEWithLogitsLoss()

    # --- Baseline cổ điển: MLP nhỏ cùng vai trò đọc embedding đã nén ---
    classical_head = nn.Sequential(
        nn.Linear(n_qubits, n_qubits), nn.Tanh(), nn.Linear(n_qubits, 1)
    )
    opt_c = torch.optim.Adam(classical_head.parameters(), lr=lr)
    for _ in range(epochs):
        opt_c.zero_grad()
        logits = classical_head(x_train).squeeze(-1)
        loss_fn(logits, y_train_t).backward()
        opt_c.step()
    with torch.no_grad():
        val_probs_c = torch.sigmoid(classical_head(x_val).squeeze(-1)).numpy()
    auprc_classical = average_precision_score(y_val, val_probs_c)

    # --- VQC ---
    logger.info(
        "Huấn luyện VQC (%d qubit, %d epoch) — có thể mất vài chục giây "
        "tới vài phút, xem GHI CHÚ HIỆU NĂNG trong docstring.", n_qubits, epochs,
    )
    vqc_head = VQCEmbeddingHead(n_qubits=n_qubits, ansatz_reps=ansatz_reps)
    readout = nn.Linear(1, 1)  # phi_4 in [-1,1] -> logit
    opt_q = torch.optim.Adam(list(vqc_head.parameters()) + list(readout.parameters()), lr=lr)
    for _ in range(epochs):
        opt_q.zero_grad()
        phi4 = vqc_head(x_train)
        logits = readout(phi4).squeeze(-1)
        loss_fn(logits, y_train_t).backward()
        opt_q.step()
    with torch.no_grad():
        val_probs_q = torch.sigmoid(readout(vqc_head(x_val)).squeeze(-1)).numpy()
    auprc_vqc = average_precision_score(y_val, val_probs_q)

    return {
        "auprc_classical_mlp": float(auprc_classical),
        "auprc_vqc": float(auprc_vqc),
        "improvement": float(auprc_vqc - auprc_classical),
    }
