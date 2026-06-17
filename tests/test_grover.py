import math
import numpy as np
import pytest
from holographic_qc.algorithms.grover import GroverAlgorithm, GroverOracle, DiffusionOperator


def test_oracle_marks_target():
    oracle = GroverOracle(3, [5])
    state = np.ones(8, dtype=complex) / np.sqrt(8)
    result = oracle.apply(state)
    assert result[5] == pytest.approx(-state[5])
    for i in [0, 1, 2, 3, 4, 6, 7]:
        assert result[i] == pytest.approx(state[i])


def test_oracle_matrix_correct():
    oracle = GroverOracle(3, [2, 5])
    U = oracle.matrix()
    assert U[2, 2] == pytest.approx(-1.0)
    assert U[5, 5] == pytest.approx(-1.0)
    assert U[0, 0] == pytest.approx(1.0)


def test_diffusion_operator_unitary():
    for n in [2, 3, 4]:
        diff = DiffusionOperator(n)
        U = diff.matrix()
        product = U @ U.conj().T
        assert np.allclose(product, np.eye(2**n), atol=1e-10)


def test_grover_optimal_iterations():
    for n in [4, 6, 8]:
        for k in [1, 2, 4]:
            grover = GroverAlgorithm(n, list(range(k)))
            n_iter = grover.optimal_iterations()
            N = 2**n
            theta = np.arcsin(np.sqrt(k / N))
            expected = int(np.floor(np.pi / (4 * theta)))
            assert n_iter == expected


def test_grover_success_probability_optimal():
    grover = GroverAlgorithm(6, [5])
    n_iter = grover.optimal_iterations()
    p = grover.success_probability(n_iter)
    assert p > 0.9


def test_grover_run_finds_target():
    grover = GroverAlgorithm(6, [42])
    result = grover.run_with_measurement(n_shots=1000)
    assert result["success_rate"] > 0.85
    assert result["n_iterations"] > 0


def test_grover_multiple_targets():
    grover = GroverAlgorithm(8, [10, 50, 100, 200])
    result = grover.run_with_measurement(n_shots=2000)
    assert result["success_rate"] > 0.75


def test_grover_quadratic_speedup():
    grover = GroverAlgorithm(10, [512])
    speedup = grover.verify_quadratic_speedup()
    assert speedup["speedup_factor"] > 1.0
    assert speedup["quantum_iterations"] < speedup["classical_expected_calls"]


def test_grover_resource_estimate():
    grover = GroverAlgorithm(8, [100])
    resources = grover.circuit_resource_estimate()
    assert "n_qubits" in resources
    assert resources["oracle_calls"] == resources["optimal_iterations"]
    assert resources["speedup_factor"] > 1.0


def test_grover_state_normalization():
    grover = GroverAlgorithm(5, [3, 7])
    state, _ = grover.run()
    norm = np.sum(np.abs(state)**2)
    assert norm == pytest.approx(1.0, abs=1e-10)


def test_grover_no_targets():
    grover = GroverAlgorithm(4, [])
    n_iter = grover.optimal_iterations()
    assert n_iter == 0
    p = grover.success_probability(0)
    assert p == pytest.approx(0.0)