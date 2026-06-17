import numpy as np
import pytest
from holographic_qc.materials.bi2se3 import Bi2Se3
from holographic_qc.materials.hgte import HgTeCdTe
from holographic_qc.materials.trapped_ions import TrappedIonChain


@pytest.fixture
def bi2se3():
    return Bi2Se3()


@pytest.fixture
def hgte():
    return HgTeCdTe()


@pytest.fixture
def ions():
    return TrappedIonChain(n_ions=8)


def test_bi2se3_xi_value(bi2se3):
    hbar = 1.054571817e-34
    eV = 1.602176634e-19
    xi_expected = hbar * bi2se3.fermi_velocity / (bi2se3.bulk_gap_eV * eV)
    assert bi2se3.xi == pytest.approx(xi_expected, rel=1e-6)


def test_bi2se3_central_charge(bi2se3):
    assert bi2se3.central_charge == pytest.approx(1.0, abs=1e-10)


def test_bi2se3_T2_decreases_with_temperature(bi2se3):
    T2_4K = bi2se3.t2_standard_ns(4.0)
    T2_10K = bi2se3.t2_standard_ns(10.0)
    assert T2_4K > T2_10K


def test_bi2se3_T2_quadratic_scaling(bi2se3):
    T2_1 = bi2se3.t2_standard_ns(1.0)
    T2_2 = bi2se3.t2_standard_ns(2.0)
    ratio = T2_1 / T2_2
    assert ratio == pytest.approx(4.0, rel=1e-4)


def test_bi2se3_holographic_T2_exceeds_standard(bi2se3):
    T2_std = bi2se3.t2_standard_ns(4.0)
    T2_holo = bi2se3.t2_holographic_ns(4.0, 1e-6)
    assert T2_holo > T2_std


def test_bi2se3_stm_ldos_positive(bi2se3):
    ldos = bi2se3.stm_ldos(100e-9, 0.001)
    assert ldos > 0.0


def test_bi2se3_stm_ldos_holographic_correction_sign(bi2se3):
    ldos_near = bi2se3.stm_ldos(5e-9, 0.001)
    ldos_far = bi2se3.stm_ldos(100e-9, 0.001)
    assert ldos_near > 0 and ldos_far > 0


def test_bi2se3_coherence_length_at_T(bi2se3):
    xi_T = bi2se3.coherence_length_at_T(4.0)
    assert xi_T > 0.0
    xi_T_cold = bi2se3.coherence_length_at_T(0.1)
    assert xi_T_cold > xi_T


def test_bi2se3_wiedemann_franz_violation(bi2se3):
    ratio = bi2se3.wiedemann_franz_ratio()
    assert ratio == pytest.approx(0.0, abs=1e-40)


def test_bi2se3_noise_spectral_density_positive(bi2se3):
    S = bi2se3.noise_spectral_density(1e12, 4.0)
    assert S > 0.0


def test_bi2se3_noise_thermal_scaling(bi2se3):
    S_hot = bi2se3.noise_spectral_density(1e12, 10.0)
    S_cold = bi2se3.noise_spectral_density(1e12, 1.0)
    assert S_hot > S_cold


def test_bi2se3_arpes_spectrum_shape(bi2se3):
    k = np.linspace(-1e9, 1e9, 100)
    E = np.linspace(-1.0, 1.0, 80)
    spectrum = bi2se3.arpes_spectrum(k, E)
    assert spectrum.shape == (80, 100)
    assert np.all(spectrum >= 0)


def test_hgte_xi_value(hgte):
    hbar = 1.054571817e-34
    eV = 1.602176634e-19
    xi_expected = hbar * hgte.fermi_velocity / (hgte.bulk_gap_eV * eV)
    assert hgte.xi == pytest.approx(xi_expected, rel=1e-6)


def test_hgte_central_charge(hgte):
    assert hgte.central_charge == pytest.approx(2.0, abs=1e-10)


def test_hgte_transport_sigma_positive(hgte):
    coeffs = hgte.transport_coefficients(4.0)
    assert coeffs["sigma_dc_S"] > 0.0


def test_hgte_transport_wf_violation(hgte):
    coeffs = hgte.transport_coefficients(4.0)
    assert coeffs["WF_violation_fraction"] == pytest.approx(1.5, rel=1e-5)


def test_hgte_gap_function_topological_phase(hgte):
    gap_topo = hgte.gap_as_function_of_well_width(7.5)
    gap_trivial = hgte.gap_as_function_of_well_width(4.0)
    assert gap_topo > 0
    assert gap_trivial < 0


def test_ions_central_charge(ions):
    assert ions.central_charge == pytest.approx(0.5, abs=1e-10)


def test_ions_at_criticality(ions):
    ions_critical = TrappedIonChain(n_ions=8, transverse_field_ratio=1.0)
    ions_off = TrappedIonChain(n_ions=8, transverse_field_ratio=2.0)
    assert ions_critical.at_criticality
    assert not ions_off.at_criticality


def test_ions_lyapunov_below_bound(ions):
    T_mK = 1.0
    kB = 1.380649e-23
    hbar = 1.054571817e-34
    bound = 2.0 * np.pi * kB * T_mK * 1e-3 / hbar
    lam = ions.lyapunov_exponent(T_mK)
    assert lam <= bound + 1e-10


def test_ions_scrambling_time_log_scaling(ions):
    t1 = ions.scrambling_time_ms(1.0)
    ions2 = TrappedIonChain(n_ions=80)
    t2 = ions2.scrambling_time_ms(1.0)
    ratio = t2 / t1
    expected = np.log(80) / np.log(8)
    assert ratio == pytest.approx(expected, rel=1e-4)


def test_ions_entanglement_entropy_scaling(ions):
    S1 = ions.entanglement_entropy_critical(4)
    S2 = ions.entanglement_entropy_critical(8)
    assert S2 > S1
    slope = (S2 - S1) / (np.log(8) - np.log(4))
    assert slope == pytest.approx(ions.central_charge / 3.0, rel=1e-5)


def test_ions_ising_hamiltonian_shape():
    ions_small = TrappedIonChain(n_ions=4)
    H = ions_small.ising_hamiltonian()
    assert H.shape == (16, 16)


def test_ions_ising_hamiltonian_hermitian():
    ions_small = TrappedIonChain(n_ions=4)
    H = ions_small.ising_hamiltonian()
    assert np.allclose(H, H.conj().T, atol=1e-10)


def test_ions_protocol_steps_non_empty(ions):
    steps = ions.otoc_protocol_steps()
    assert len(steps) > 0
    assert all(isinstance(s, str) for s in steps)