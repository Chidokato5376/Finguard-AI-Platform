"""Test cho src/quantum/vqc.py — chạy MẠCH LƯỢNG TỬ THẬT qua
EstimatorQNN/TorchConnector, không mock.

QUAN TRỌNG VỀ THỜI GIAN: mỗi lần huấn luyện VQC (kể cả epochs=1) tốn vài
giây vì Estimator phải tính kỳ vọng đo nhiều lần cho gradient (xem GHI
CHÚ HIỆU NĂNG trong src/quantum/vqc.py). Test ở đây dùng quy mô NHỎ NHẤT
có thể (n_qubits=2-3, epochs=1-2, vài mẫu) — đủ để xác nhận pipeline
đúng, không dùng để đánh giá chất lượng VQC (không phải mục đích của
unit test).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.quantum.vqc import QuantumEnhancedEmbedding, VQCEmbeddingHead, benchmark_vs_classical


def test_vqc_head_forward_output_bounded() -> None:
    """phi_4 = <Z_0> phải luôn trong [-1, 1] — tính chất toán học của
    kỳ vọng đo Pauli-Z, không phụ thuộc dữ liệu đầu vào hay tham số.
    """
    torch.manual_seed(0)
    head = VQCEmbeddingHead(n_qubits=3, ansatz_reps=1)
    x = torch.randn(4, 3)
    out = head(x)
    assert out.shape == (4, 1)
    assert torch.all(out.abs() <= 1.0 + 1e-6)


def test_vqc_head_gradient_flows_to_ansatz_params() -> None:
    """Xác nhận backward() thực sự cập nhật được tham số ansatz — nếu
    gradient không chảy được (vd. do input_gradients cấu hình sai), lỗi
    này sẽ ÂM THẦM khiến VQC không bao giờ học được gì mà không crash.
    """
    torch.manual_seed(1)
    head = VQCEmbeddingHead(n_qubits=2, ansatz_reps=1)
    x = torch.randn(3, 2)

    out = head(x)
    out.sum().backward()

    grads = [p.grad for p in head.parameters()]
    assert len(grads) > 0
    assert any(g is not None and torch.any(g != 0) for g in grads)


def test_vqc_head_gradient_flows_to_input() -> None:
    """input_gradients=True phải cho phép gradient chảy NGƯỢC vào chính
    embedding đầu vào — cần thiết để huấn luyện đồng thời với DRRE phía
    trước (không chỉ huấn luyện ansatz một cách cô lập).
    """
    torch.manual_seed(2)
    head = VQCEmbeddingHead(n_qubits=2, ansatz_reps=1)
    x = torch.randn(3, 2, requires_grad=True)

    out = head(x)
    out.sum().backward()

    assert x.grad is not None
    assert torch.any(x.grad != 0)


def test_quantum_enhanced_embedding_requires_fit_pca_first() -> None:
    qe = QuantumEnhancedEmbedding(n_qubits=2)
    with pytest.raises(RuntimeError):
        qe.transform(np.random.randn(3, 8))


def test_quantum_enhanced_embedding_end_to_end() -> None:
    qe = QuantumEnhancedEmbedding(n_qubits=2, ansatz_reps=1, pca_components=2)
    e_train = np.random.randn(10, 8)
    qe.fit_pca(e_train)

    out = qe.forward(np.random.randn(3, 8))
    assert out.shape == (3, 1)
    assert torch.all(out.abs() <= 1.0 + 1e-6)


def test_benchmark_vs_classical_runs_end_to_end_minimal_scale() -> None:
    """Test chậm nhất trong bộ (vài giây) — chạy TOÀN BỘ pipeline benchmark
    (PCA + MLP cổ điển + VQC thật) ở quy mô tối thiểu, chỉ xác nhận không
    crash và trả về đúng cấu trúc kết quả, KHÔNG đánh giá VQC có tốt hơn
    MLP hay không (không phải mục đích của unit test — xem benchmark thật
    trong docs/architecture.md).
    """
    np.random.seed(0)
    e_train = np.random.randn(12, 6)
    y_train = (np.random.rand(12) < 0.4).astype(int)
    e_val = np.random.randn(6, 6)
    y_val = (np.random.rand(6) < 0.4).astype(int)

    result = benchmark_vs_classical(
        e_train, y_train, e_val, y_val, n_qubits=2, ansatz_reps=1, epochs=2
    )

    assert set(result.keys()) == {"auprc_classical_mlp", "auprc_vqc", "improvement"}
    assert 0.0 <= result["auprc_classical_mlp"] <= 1.0
    assert 0.0 <= result["auprc_vqc"] <= 1.0
    assert result["improvement"] == pytest.approx(
        result["auprc_vqc"] - result["auprc_classical_mlp"]
    )
