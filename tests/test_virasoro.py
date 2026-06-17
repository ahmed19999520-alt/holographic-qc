import numpy as np
import pytest
from holographic_qc.core.virasoro import VirasoroAlgebra, VirasoroConfig, WardIdentityVerifier


@pytest.fixture
def algebra_c1():
    config = VirasoroConfig(central_charge=1.0, max_mode=5)
    return VirasoroAlgebra(config)


@pytest.fixture
def algebra_c2():
    config = VirasoroConfig(central_charge=2.0, max_mode=5)
    return VirasoroAlgebra(config)


def test_commutator_scalar_basic(algebra_c1):
    lin, cen = algebra_c1.commutator_scalar(2, 3)
    assert lin == pytest.approx(-1.0)
    assert cen == pytest.approx(0.0)


def test_commutator_central_term(algebra_c1):
    lin, cen = algebra_c1.commutator_scalar(3, -3)
    assert lin == pytest.approx(6.0)
    assert cen == pytest.approx(1.0 / 12.0 * 3 * (9 - 1))


def test_commutator_antisymmetry(algebra_c1):
    lin12, cen12 = algebra_c1.commutator_scalar(2, -3)
    lin21, cen21 = algebra_c1.commutator_scalar(-3, 2)
    assert lin12 == pytest.approx(-lin21)


def test_jacobi_identity(algebra_c1):
    for l, m, n in [(1, 2, -3), (2, -1, -1), (3, -2, -1)]:
        assert algebra_c1.verify_jacobi_identity(l, m, n)


def test_ope_coefficient_coincident_points(algebra_c1):
    with pytest.raises(ValueError):
        algebra_c1.ope_tilde_coeff(1.0 + 0j, 1.0 + 0j)


def test_ope_coefficient_value(algebra_c1):
    result = algebra_c1.ope_tilde_coeff(2.0 + 0j, 0.0 + 0j)
    assert abs(result - (1.0 / 2.0) / 2.0**4) < 1e-10


def test_two_point_function(algebra_c1):
    result = algebra_c1.two_point_function(3.0 + 0j, 1.0 + 0j, 1.0)
    assert abs(result - 1.0 / 4.0) < 1e-10


def test_central_charge_from_commutator(algebra_c2):
    lin, cen = algebra_c2.commutator_scalar(2, -2)
    expected_cen = 2.0 / 12.0 * 2 * (4 - 1)
    assert cen == pytest.approx(expected_cen)


def test_kac_table_shape(algebra_c1):
    table = algebra_c1.kac_table(4, 3)
    assert table.shape == (3, 2)


def test_character_convergence(algebra_c1):
    q = 0.1 + 0j
    chi = algebra_c1.character(0.0, q, n_levels=30)
    assert np.isfinite(chi)
    assert chi.real > 0


def test_character_diverges_at_q1(algebra_c1):
    with pytest.raises(ValueError):
        algebra_c1.character(0.0, 1.0 + 0j)


def test_partition_states_level0(algebra_c1):
    states = algebra_c1.partition_states(0)
    assert len(states) == 1
    assert states[0] == ()


def test_partition_states_level2(algebra_c1):
    states = algebra_c1.partition_states(2)
    assert len(states) == 2


def test_gram_matrix_positive_definite_above_vacuum(algebra_c1):
    G = algebra_c1.gram_matrix(0.5, 1)
    eigvals = np.linalg.eigvalsh(G)
    assert np.all(eigvals >= -1e-10)


def test_lyapunov_bound(algebra_c1):
    kB = 1.380649e-23
    hbar = 1.054571817e-34
    T = 4.0
    bound = 2.0 * np.pi * kB * T / hbar
    lam = algebra_c1.lyapunov_from_central_charge(T)
    assert lam <= bound + 1e-10


def test_ward_identity_closure(algebra_c1):
    verifier = WardIdentityVerifier(algebra_c1)
    errors = verifier.verify_algebra_closure(mode_range=3)
    for key, err in errors.items():
        assert err < 1e-10, f"Algebra closure failed for {key}: error={err}"