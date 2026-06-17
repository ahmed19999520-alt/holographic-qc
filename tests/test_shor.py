import math
import pytest
from holographic_qc.algorithms.shor import ShorAlgorithm, PeriodFinder, ModularExponentiator
from holographic_qc.algorithms.qft import QuantumFourierTransform


def test_qft_unitarity():
    for n in [2, 3, 4, 5]:
        qft = QuantumFourierTransform(n)
        assert qft.verify_unitarity(), f"QFT not unitary for n={n}"


def test_qft_inverse():
    for n in [2, 3, 4]:
        qft = QuantumFourierTransform(n)
        assert qft.verify_inverse(), f"QFT inverse failed for n={n}"


def test_qft_bit_reversal():
    import numpy as np
    qft = QuantumFourierTransform(3)
    state = np.zeros(8, dtype=complex)
    state[0] = 1.0
    result = qft.bit_reversal_permutation(state)
    assert result[0] == pytest.approx(1.0)


def test_modular_exp_basic():
    me = ModularExponentiator(7, 15)
    assert me.compute(0) == 1
    assert me.compute(1) == 7
    assert me.compute(4) == pow(7, 4, 15)


def test_period_finder_continued_fraction():
    pf = PeriodFinder(7, 15, n_precision=8)
    r = pf.continued_fraction_period(64)
    assert r is None or r > 0


def test_shor_factor_15():
    shor = ShorAlgorithm(15)
    result = shor.factor_classical_simulation()
    assert result is not None
    p, q = result
    assert p * q == 15
    assert p > 1 and q > 1


def test_shor_factor_21():
    shor = ShorAlgorithm(21)
    result = shor.factor_classical_simulation()
    assert result is not None
    p, q = result
    assert p * q == 21


def test_shor_factor_35():
    shor = ShorAlgorithm(35)
    result = shor.factor_classical_simulation()
    assert result is not None
    p, q = result
    assert p * q == 35


def test_shor_even_number():
    shor = ShorAlgorithm(14)
    result = shor.factor_classical_simulation()
    assert result == (2, 7) or result == (7, 2)


def test_shor_resource_estimate():
    shor = ShorAlgorithm(15)
    res = shor.circuit_resource_estimate()
    assert "n_logical_qubits" in res
    assert res["n_logical_qubits"] > 0
    assert res["circuit_depth"] > 0


def test_shor_verify_factorization():
    shor = ShorAlgorithm(15)
    assert shor.verify_factorization(3, 5)
    assert shor.verify_factorization(5, 3)
    assert not shor.verify_factorization(2, 7)
    assert not shor.verify_factorization(1, 15)


def test_shor_is_prime():
    shor = ShorAlgorithm(17)
    assert shor.is_prime_classical(17)
    assert not shor.is_prime_classical(15)
    assert not shor.is_prime_classical(1)


def test_shor_is_prime_power():
    shor = ShorAlgorithm(9)
    is_pp, base, exp = shor.is_prime_power(9)
    assert is_pp
    assert base == 3 and exp == 2