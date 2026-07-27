"""Quantum-enhanced Embedding — Quantum Kernel (Mục 4.5, phương án ưu tiên).

K_Q(e_i, e_j) = | <0| U_dagger(e_j) U(e_i) |0> |^2

Dùng FidelityQuantumKernel (qiskit-machine-learning), thay cho kernel RBF
trong một bộ phân biệt nhỏ (SVM/kNN) trên không gian embedding, để tinh
chỉnh biên giữa vùng rủi ro cao và rủi ro thấp. Không cần huấn luyện
tham số mạch.

GIỚI HẠN (bắt buộc nêu khi báo cáo): simulator Qiskit chỉ mô phỏng hiệu
quả tới ~10-16 qubit -> chỉ nhận bản nén PCA của embedding, không nhận
toàn bộ vector e_t^c gốc. Nhánh này là TÙY CHỌN — chỉ bật (quantum.enabled
trong config.yaml) sau khi benchmark_vs_classical() xác nhận cải thiện
AUPRC thực sự so với bỏ nhánh này trên cùng tập kiểm định.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC

try:
    from qiskit.circuit.library import ZZFeatureMap
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
except ImportError:  # pragma: no cover
    ZZFeatureMap = None
    FidelityQuantumKernel = None


class QuantumEnhancedKernel:
    """Nén embedding cổ điển bằng PCA rồi ánh xạ qua feature map lượng tử."""

    def __init__(self, n_qubits: int = 12, pca_components: int | None = None) -> None:
        if FidelityQuantumKernel is None:
            raise ImportError(
                "qiskit-machine-learning chưa được cài. Xem requirements.txt."
            )
        self.n_qubits = n_qubits
        self.pca = PCA(n_components=pca_components or n_qubits)
        feature_map = ZZFeatureMap(feature_dimension=self.pca.n_components, reps=2)
        self.kernel = FidelityQuantumKernel(feature_map=feature_map)
        self._fitted = False

    def fit_pca(self, e: np.ndarray) -> None:
        """Fit PCA trên tập embedding cổ điển trước khi ánh xạ lượng tử."""
        self.pca.fit(e)
        self._fitted = True

    def compute_kernel_matrix(self, e_a: np.ndarray, e_b: np.ndarray | None = None) -> np.ndarray:
        """Tính K_Q(e_i, e_j) cho các embedding đã nén PCA.

        Args:
            e_a: (n, embedding_dim) — embedding cổ điển từ DRRE.
            e_b: (m, embedding_dim) hoặc None (tự-kernel với e_a).
        """
        if not self._fitted:
            raise RuntimeError("Gọi fit_pca() trước khi compute_kernel_matrix().")
        a_compressed = self.pca.transform(e_a)
        b_compressed = self.pca.transform(e_b) if e_b is not None else None
        return self.kernel.evaluate(x_vec=a_compressed, y_vec=b_compressed)


def benchmark_vs_classical(
    e_train: np.ndarray,
    y_train: np.ndarray,
    e_val: np.ndarray,
    y_val: np.ndarray,
    n_qubits: int = 12,
) -> dict[str, float]:
    """So sánh AUPRC: SVM với Quantum Kernel vs SVM với RBF kernel cổ điển.

    ĐÂY LÀ BƯỚC BẮT BUỘC trước khi bật quantum.enabled trong config.yaml —
    xem cảnh báo ở đầu file và docs/architecture.md.
    """
    from sklearn.metrics import average_precision_score

    # Baseline cổ điển
    svm_classical = SVC(kernel="rbf", probability=True)
    svm_classical.fit(e_train, y_train)
    auprc_classical = average_precision_score(
        y_val, svm_classical.predict_proba(e_val)[:, 1]
    )

    # Nhánh quantum
    qk = QuantumEnhancedKernel(n_qubits=n_qubits)
    qk.fit_pca(e_train)
    k_train = qk.compute_kernel_matrix(e_train)
    k_val = qk.compute_kernel_matrix(e_val, e_train)

    svm_quantum = SVC(kernel="precomputed", probability=True)
    svm_quantum.fit(k_train, y_train)
    auprc_quantum = average_precision_score(y_val, svm_quantum.predict_proba(k_val)[:, 1])

    return {
        "auprc_classical_rbf": float(auprc_classical),
        "auprc_quantum_kernel": float(auprc_quantum),
        "improvement": float(auprc_quantum - auprc_classical),
    }
