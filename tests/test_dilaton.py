import numpy as np
import pytest
from holographic_qc.core.dilaton import DilatonField, DilatonConfig


@pytest.fixture
def dilaton_massless():
    return DilatonField(DilatonConfig(ads_radius=1.1e-9, mass_sq_times_l_sq=0.0, fermi_velocity=5e5))


@pytest.fixture
def dilaton_massive():
    return DilatonField(DilatonConfig(ads_radius=1.1e-9, mass_sq_times_l_sq=2.0, fermi_velocity=5e5))


def test_nu_massless(dilaton_massless):
    assert dilaton_massless.nu == pytest.approx(0.5, abs=1e-10)


def test_delta_massless(dilaton_massless):
    assert dilaton_massless.delta == pytest.approx(1.5, abs=1e-10)


def test_nu_massive(dilaton_massive):
    expected = np.sqrt(0.25 + 2.0)
    assert dilaton_massive.nu == pytest.approx(expected, abs=1e-10)


def test_two_point_power_law(dilaton_massless):
    x, xp = 1e-8, 0.0
    result = dilaton_massless.two_point_function(x, xp)
    assert result > 0.0
    assert np.isfinite(result)


def test_two_point_coincident_raises(dilaton_massless):
    with pytest.raises(ValueError):
        dilaton_massless.two_point_function(1.0, 1.0)


def test_two_point_separation_scaling(dilaton_massless):
    delta = dilaton_massless.delta
    x1, x2 = 1e-8, 2e-8
    r1 = dilaton_massless.two_point_function(x1, 0.0)
    r2 = dilaton_massless.two_point_function(x2, 0.0)
    ratio = r1 / r2
    expected = (x2 / x1)**(2 * delta)
    assert ratio == pytest.approx(expected, rel=1e-5)


def test_holographic_log_correction_positive(dilaton_massless):
    xi = 1.1e-9
    x = 100e-9
    corr = dilaton_massless.holographic_correlation_with_log_correction(x, xi, c=1.0)
    base = 1.0 / x**2
    assert corr > base


def test_dsf_zero_below_threshold(dilaton_massless):
    q = 1e8
    omega_below = dilaton_massless.v_F * abs(q) * 0.5
    S = dilaton_massless.dynamical_structure_factor(q, omega_below, temperature=4.0)
    assert S == pytest.approx(0.0, abs=1e-30)


def test_dsf_positive_above_threshold(dilaton_massless):
    q = 1e8
    omega_above = dilaton_massless.v_F * abs(q) * 1.5
    S = dilaton_massless.dynamical_structure_factor(q, omega_above, temperature=4.0)
    assert S > 0.0


def test_dsf_increases_with_amplitude(dilaton_massless):
    q, omega = 1e8, dilaton_massless.v_F * 1e8 * 2.0
    S1 = dilaton_massless.dynamical_structure_factor(q, omega, 4.0, A=1.0)
    S2 = dilaton_massless.dynamical_structure_factor(q, omega, 4.0, A=2.0)
    assert S2 == pytest.approx(2.0 * S1, rel=1e-5)


def test_noise_screening_decreases_with_size(dilaton_massless):
    xi = 1.1e-9
    noise = 1.0
    s1 = dilaton_massless.noise_power_after_screening(1e-7, xi, 1.0, noise)
    s2 = dilaton_massless.noise_power_after_screening(1e-6, xi, 1.0, noise)
    s3 = dilaton_massless.noise_power_after_screening(1e-5, xi, 1.0, noise)
    assert s1 > s2 > s3


def test_bosonization_density(dilaton_massless):
    phi = np.sin(np.linspace(0, 2 * np.pi, 100))
    dx = 2 * np.pi / 100
    rho = dilaton_massless.bosonization_density(phi, dx)
    assert len(rho) == len(phi)
    assert np.all(np.isfinite(rho))