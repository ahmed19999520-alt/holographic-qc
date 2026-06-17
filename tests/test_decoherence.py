import numpy as np
import pytest
from holographic_qc.protection.decoherence import HolographicDecoherence, LindbladEvolution
from holographic_qc.protection.majorana import MajoranaQubit, MajoranaFermionSystem
from holographic_qc.protection.holographic_error_correction import (
    HolographicCode, SurfaceCode, PentagonHaPPYCode
)
from holographic_qc.core.ads_cft import AdsCft3
from holographic_qc.materials.bi2se3 import Bi2Se3


@pytest.fixture
def mat():
    return Bi2Se3()


@pytest.fixture
def ads(mat):
    return AdsCft3(central_charge=mat.central_charge, ads_radius=mat.xi, fermi_velocity=mat.fermi_velocity)


@pytest.fixture
def holo_dec(ads, mat):
    return HolographicDecoherence(ads_system=ads, material=mat)


def test_lindblad_trace_preservation():
    H = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    ev = LindbladEvolution(H, [sigma_z], [0.1])
    rho0 = np.array([[0.6, 0.4], [0.4, 0.4]], dtype=np.complex128)
    rho_t = ev.evolve(rho0, 1.0)
    assert np.trace(rho_t) == pytest.approx(1.0, abs=1e-6)


def test_lindblad_hermitian_output():
    H = np.diag([0.0, 1.0]).astype(np.complex128)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    ev = LindbladEvolution(H, [sigma_z], [0.1])
    rho0 = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.complex128)
    rho_t = ev.evolve(rho0, 0.5)
    assert np.allclose(rho_t, rho_t.conj().T, atol=1e-8)


def test_coherence_decay_monotone():
    H = np.zeros((2, 2), dtype=np.complex128)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    ev = LindbladEvolution(H, [sigma_z], [1.0])
    rho0 = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.complex128)
    times = np.linspace(0, 3.0, 50)
    coh = ev.coherence_decay(rho0, times)
    magnitudes = np.abs(coh)
    assert np.all(np.diff(magnitudes) <= 1e-10)


def test_phonon_rate_temperature_scaling(holo_dec, mat):
    r1 = holo_dec.standard_phonon_rate_2d(4.0)
    r2 = holo_dec.standard_phonon_rate_2d(8.0)
    ratio = r2 / r1
    assert ratio == pytest.approx(4.0, rel=1e-5)


def test_holographic_enhancement_unity_at_ratio1(holo_dec, mat):
    enh = holo_dec.coherence_time_ratio(mat.xi, mat.xi)
    assert enh == pytest.approx(1.0, abs=1e-10)


def test_holographic_enhancement_positive(holo_dec, mat):
    enh = holo_dec.coherence_time_ratio(1e-6, mat.xi)
    assert enh > 1.0


def test_holographic_enhancement_scales_with_c(mat):
    ads1 = AdsCft3(central_charge=1.0, ads_radius=mat.xi, fermi_velocity=mat.fermi_velocity)
    ads2 = AdsCft3(central_charge=2.0, ads_radius=mat.xi, fermi_velocity=mat.fermi_velocity)
    dec1 = HolographicDecoherence(ads1, mat)
    dec2 = HolographicDecoherence(ads2, mat)
    enh1 = dec1.coherence_time_ratio(1e-6, mat.xi)
    enh2 = dec2.coherence_time_ratio(1e-6, mat.xi)
    assert enh2 > enh1


def test_combined_T2_exceeds_standard(holo_dec, mat):
    T_std = holo_dec.coherence_time_standard(4.0)
    T_holo = holo_dec.coherence_time_holographic(4.0, 1e-6)
    assert T_holo > T_std


def test_majorana_qubit_parity_operator():
    qubit = MajoranaQubit(100e-9, 1.1e-9)
    P = qubit.parity_operator()
    assert P.shape == (2, 2)
    eigvals = np.linalg.eigvalsh(P.real)
    assert set(np.round(eigvals).astype(int)) == {-1, 1}


def test_majorana_holographic_enhancement_positive():
    ads = AdsCft3(central_charge=1.0, ads_radius=1.1e-9)
    qubit = MajoranaQubit(100e-9, 1.1e-9, ads_system=ads)
    enh = qubit.holographic_enhancement_factor()
    assert enh > 1.0


def test_majorana_total_T2_exceeds_topological():
    ads = AdsCft3(central_charge=1.0, ads_radius=1.1e-9)
    qubit = MajoranaQubit(100e-9, 1.1e-9, ads_system=ads)
    tau0 = 1e-6
    T2_topo = tau0 * np.exp(100e-9 / 1.1e-9)
    T2_total = qubit.total_coherence_time(tau0)
    assert T2_total >= T2_topo


def test_majorana_gate_fidelity_lt_1():
    ads = AdsCft3(central_charge=1.0, ads_radius=1.1e-9)
    qubit = MajoranaQubit(100e-9, 1.1e-9, ads_system=ads)
    F = qubit.gate_fidelity(0.01)
    assert 0.0 < F < 1.0


def test_majorana_gate_fidelity_approaches_1_large_L():
    ads = AdsCft3(central_charge=2.0, ads_radius=1.1e-9)
    qubit_large = MajoranaQubit(1e-4, 1.1e-9, ads_system=ads)
    F = qubit_large.gate_fidelity(0.01)
    assert F > 0.999


def test_majorana_system_spectrum():
    sys = MajoranaFermionSystem(n_sites=6, t=1.0, delta=1.0, mu=0.0)
    spectrum = sys.energy_spectrum()
    assert len(spectrum) == 12
    assert np.all(np.diff(spectrum) >= -1e-10)


def test_majorana_system_topological_invariant():
    sys_topo = MajoranaFermionSystem(n_sites=6, t=1.0, delta=1.0, mu=0.0)
    sys_trivial = MajoranaFermionSystem(n_sites=6, t=1.0, delta=1.0, mu=3.0)
    assert sys_topo.topological_invariant() == 1
    assert sys_trivial.topological_invariant() == 0


def test_surface_code_distance():
    code = SurfaceCode(distance=5)
    assert code.distance == 5
    assert code.k == 1


def test_surface_code_holographic_distance_larger():
    code = SurfaceCode(distance=7)
    d_holo = code.holographic_code_distance(1.0, 1e-6, 1e-9)
    assert d_holo > code.distance


def test_holographic_code_threshold_larger():
    code = SurfaceCode(distance=7)
    p_std = 0.01
    p_holo = code.holographic_threshold(1.0, p_std)
    assert p_holo > p_std


def test_pentagon_code_encoding_rate():
    code = PentagonHaPPYCode(n_layers=2)
    rate = code.encoding_rate()
    assert 0 < rate < 1


def test_holographic_code_logical_error_below_threshold():
    code = HolographicCode(code_type="surface", central_charge=1.0)
    p_phys = 0.005
    err = code.logical_error_rate(p_phys, 1e-6, 1e-9)
    assert 0 <= err < p_phys


def test_holographic_code_above_threshold():
    code = HolographicCode(code_type="surface", central_charge=1.0)
    p_phys = 0.05
    err = code.logical_error_rate(p_phys, 1e-6, 1e-9)
    assert err >= 0.1