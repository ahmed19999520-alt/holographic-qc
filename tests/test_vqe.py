import numpy as np
import pytest
from holographic_qc.algorithms.vqe import VQE, VariationalAnsatz, PauliOperator


@pytest.fixture
def ising_h4():
    return VQE.ising_hamiltonian(4, J=1.0, h=1.0)


@pytest.fixture
def heisenberg_h4():
    return VQE.heisenberg_hamiltonian(4, J=1.0)


def test_ising_hamiltonian_hermitian(ising_h4):
    assert np.allclose(ising_h4, ising_h4.conj().T, atol=1e-10)


def test_ising_hamiltonian_real(ising_h4):
    assert np.allclose(ising_h4.imag, np.zeros_like(ising_h4.imag), atol=1e-10)


def test_ising_hamiltonian_shape():
    for n in [2, 3, 4]:
        H = VQE.ising_hamiltonian(n)
        assert H.shape == (2**n, 2**n)


def test_heisenberg_hermitian(heisenberg_h4):
    assert np.allclose(heisenberg_h4, heisenberg_h4.conj().T, atol=1e-10)


def test_ansatz_state_normalized():
    ansatz = VariationalAnsatz(n_qubits=3, depth=2)
    params = ansatz.initial_params(seed=0)
    state = ansatz.state_vector(params)
    norm = np.sum(np.abs(state)**2)
    assert norm == pytest.approx(1.0, abs=1e-10)


def test_ansatz_gradient_finite_difference():
    H = VQE.ising_hamiltonian(3, J=1.0, h=1.0)
    ansatz = VariationalAnsatz(n_qubits=3, depth=2)
    params = ansatz.initial_params(seed=42)
    grad = ansatz.gradient(params, H)
    assert len(grad) == len(params)
    for i in range(len(params)):
        eps = 1e-5
        p_plus = params.copy(); p_plus[i] += eps
        p_minus = params.copy(); p_minus[i] -= eps
        fd = (ansatz.energy(p_plus, H) - ansatz.energy(p_minus, H)) / (2 * eps)
        assert abs(grad[i] - fd) < 1e-4, f"Gradient mismatch at param {i}"


def test_vqe_energy_above_ground(ising_h4):
    vqe = VQE(ising_h4, 4, depth=1, max_iter=10)
    result = vqe.run(seed=0)
    exact = result["exact_ground_energy"]
    assert result["optimal_energy"] >= exact - 1e-6


def test_vqe_converges_ising(ising_h4):
    vqe = VQE(ising_h4, 4, depth=3, max_iter=300)
    result = vqe.run(seed=42)
    assert result["energy_error"] < 0.1
    assert result["converged"]


def test_vqe_deeper_is_better(ising_h4):
    results = []
    for depth in [1, 2, 3]:
        vqe = VQE(ising_h4, 4, depth=depth, max_iter=200)
        r = vqe.run(seed=7)
        results.append(r["energy_error"])
    assert results[2] <= results[0] + 0.05


def test_vqe_variance_small_at_optimum(ising_h4):
    vqe = VQE(ising_h4, 4, depth=3, max_iter=300)
    result = vqe.run(seed=42)
    var = vqe.variance(result["optimal_params"])
    assert var >= -1e-8


def test_vqe_heisenberg_energy(heisenberg_h4):
    exact = np.linalg.eigvalsh(heisenberg_h4)[0]
    vqe = VQE(heisenberg_h4, 4, depth=4, max_iter=400)
    result = vqe.run(seed=0)
    assert result["energy_error"] < 1.0


def test_vqe_energy_landscape_1d(ising_h4):
    vqe = VQE(ising_h4, 4, depth=2, max_iter=50)
    result = vqe.run(seed=0)
    thetas, energies = vqe.energy_landscape_1d(result["optimal_params"], 0, n_points=20)
    assert len(thetas) == 20
    assert len(energies) == 20
    assert np.all(np.isfinite(energies))


def test_pauli_operators_anticommute():
    sx = PauliOperator.SIGMA_X
    sy = PauliOperator.SIGMA_Y
    anticomm = sx @ sy + sy @ sx
    assert np.allclose(anticomm, np.zeros((2, 2)), atol=1e-10)


def test_pauli_operators_trace_zero():
    for op in [PauliOperator.SIGMA_X, PauliOperator.SIGMA_Y, PauliOperator.SIGMA_Z]:
        assert abs(np.trace(op)) < 1e-10


def test_pauli_kron_op_shape():
    for n in [2, 3, 4]:
        for site in range(n):
            op = PauliOperator.kron_op(PauliOperator.SIGMA_X, site, n)
            assert op.shape == (2**n, 2**n)