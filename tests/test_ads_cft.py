import numpy as np
import pytest
from holographic_qc.core.ads_cft import AdsCft3
from holographic_qc.core.ryu_takayanagi import RyuTakayanagi, RTConfig
from holographic_qc.core.christoffel import ChristoffelSymbols


@pytest.fixture
def ads_c1():
    return AdsCft3(central_charge=1.0, ads_radius=1.1e-9, fermi_velocity=5e5)


@pytest.fixture
def ads_c2():
    return AdsCft3(central_charge=2.0, ads_radius=19.8e-9, fermi_velocity=3e5)


@pytest.fixture
def rt_c1():
    config = RTConfig(
        central_charge=1.0,
        newton_constant_3d=1.1e-9 * 2.0 / 3.0,
        ads_radius=1.1e-9,
        uv_cutoff=0.3e-9,
    )
    return RyuTakayanagi(config)


def test_brown_henneaux_consistency(ads_c1):
    c_computed = ads_c1.brown_henneaux_central_charge()
    assert c_computed == pytest.approx(ads_c1.central_charge, rel=1e-6)


def test_scaling_dimension_massless(ads_c1):
    delta = ads_c1.scaling_dimension_from_mass(0.0)
    assert delta == pytest.approx(1.5, abs=1e-10)


def test_breitenlohner_freedman_bound(ads_c1):
    bf = ads_c1.breitenlohner_freedman_bound()
    assert bf == pytest.approx(-0.25, abs=1e-10)


def test_scaling_dim_below_bf_raises(ads_c1):
    with pytest.raises(ValueError):
        ads_c1.scaling_dimension_from_mass(-1.0)


def test_two_point_function_power_law(ads_c1):
    x1, x2 = 1e-8, 0.0
    delta = 1.0
    result = ads_c1.two_point_function(x1, x2, delta)
    expected = 0.5 / abs(x1)**2
    assert result == pytest.approx(expected, rel=1e-5)


def test_two_point_coincident_raises(ads_c1):
    with pytest.raises(ValueError):
        ads_c1.two_point_function(1.0, 1.0, 1.0)


def test_bulk_to_boundary_propagator_boundary_limit(ads_c1):
    z_small = 1e-12
    x, x_prime = 0.0, 1e-8
    delta = 1.0
    K = ads_c1.bulk_to_boundary_propagator(z_small, x, x_prime, delta)
    assert K > 0.0
    assert np.isfinite(K)


def test_bulk_to_boundary_propagator_decay(ads_c1):
    x, xp = 0.0, 1e-8
    delta = 1.0
    K_small = ads_c1.bulk_to_boundary_propagator(1e-11, x, xp, delta)
    K_large = ads_c1.bulk_to_boundary_propagator(1e-9, x, xp, delta)
    assert K_small > K_large


def test_optical_conductivity_dc(ads_c1):
    sigma = ads_c1.optical_conductivity_dc()
    e_sq_over_h = 3.87404e-5
    expected = e_sq_over_h * ads_c1.central_charge / 2.0
    assert sigma == pytest.approx(expected, rel=1e-5)


def test_optical_conductivity_dc_c2(ads_c2):
    sigma_c1 = AdsCft3(central_charge=1.0, ads_radius=1e-9).optical_conductivity_dc()
    sigma_c2 = ads_c2.optical_conductivity_dc()
    assert sigma_c2 == pytest.approx(2.0 * sigma_c1, rel=1e-5)


def test_wiedemann_franz_c1(ads_c1):
    ratio = ads_c1.wiedemann_franz_ratio()
    kB = 1.380649e-23
    e = 1.602176634e-19
    L0 = (np.pi**2 / 3.0) * kB**2 / e**2
    assert ratio == pytest.approx(0.0 * L0, abs=1e-40)


def test_wiedemann_franz_c_inf():
    ads_inf = AdsCft3(central_charge=1000.0, ads_radius=1e-9)
    ratio = ads_inf.wiedemann_franz_ratio()
    kB = 1.380649e-23
    e = 1.602176634e-19
    L0 = (np.pi**2 / 3.0) * kB**2 / e**2
    assert ratio == pytest.approx(L0, rel=1e-2)


def test_lyapunov_below_bound(ads_c1):
    T = 4.0
    kB = 1.380649e-23
    hbar = 1.054571817e-34
    bound = 2.0 * np.pi * kB * T / hbar
    lam = ads_c1.lyapunov_exponent(T)
    assert lam <= bound + 1e-10


def test_lyapunov_approaches_bound_large_c():
    T = 4.0
    kB = 1.380649e-23
    hbar = 1.054571817e-34
    bound = 2.0 * np.pi * kB * T / hbar
    ads_large = AdsCft3(central_charge=1000.0, ads_radius=1e-9)
    lam = ads_large.lyapunov_exponent(T)
    assert lam == pytest.approx(bound, rel=1e-2)


def test_coherence_enhancement_unity_at_ratio_1(ads_c1):
    enh = ads_c1.holographic_coherence_enhancement(1.0, 1.0)
    assert enh == pytest.approx(1.0, abs=1e-10)


def test_coherence_enhancement_scaling(ads_c1):
    L, xi = 1000.0, 1.0
    c = ads_c1.central_charge
    enh = ads_c1.holographic_coherence_enhancement(L, xi)
    expected = (L / xi)**(c / 6.0)
    assert enh == pytest.approx(expected, rel=1e-10)


def test_rt_entropy_formula(rt_c1):
    ell = 100e-9
    S = rt_c1.entanglement_entropy_central_charge(ell)
    expected = (rt_c1.c / 3.0) * np.log(ell / rt_c1.a)
    assert S == pytest.approx(expected, rel=1e-10)


def test_rt_mutual_information_non_negative(rt_c1):
    MI = rt_c1.mutual_information(50e-9, 50e-9, 10e-9)
    assert MI >= -1e-10


def test_rt_renyi_n1_equals_von_neumann(rt_c1):
    ell = 100e-9
    S_vN = rt_c1.entanglement_entropy_central_charge(ell)
    S_R1 = rt_c1.renyi_entropy(ell, 1)
    assert S_vN == pytest.approx(S_R1, rel=1e-10)


def test_rt_renyi_ordering(rt_c1):
    ell = 100e-9
    S2 = rt_c1.renyi_entropy(ell, 2)
    S3 = rt_c1.renyi_entropy(ell, 3)
    S_inf = rt_c1.renyi_entropy(ell, 100)
    assert S2 >= S3 >= S_inf - 1e-10


def test_christoffel_ads3_diagonal():
    cs = ChristoffelSymbols.from_ads3_poincare(1.1e-9, 1e-10)
    assert cs.g.shape == (3, 3)
    for i in range(3):
        for j in range(3):
            if i != j:
                assert cs.g[i, j] == pytest.approx(0.0, abs=1e-30)


def test_christoffel_geodesic_length(ads_c1):
    cs = ChristoffelSymbols.from_ads3_poincare(ads_c1.ads_radius, 1e-11)
    L_geo = cs.ads3_geodesic_length_boundary(100e-9, 0.3e-9, ads_c1.ads_radius)
    expected = 2.0 * ads_c1.ads_radius * np.log(100e-9 / 0.3e-9)
    assert L_geo == pytest.approx(expected, rel=1e-10)


def test_btz_entropy_positive(ads_c1):
    T = 4.0
    S = ads_c1.btz_entropy(T)
    assert S > 0.0


def test_scrambling_time_log_n(ads_c1):
    T = 4.0
    t1 = ads_c1.scrambling_time(T, 10)
    t2 = ads_c1.scrambling_time(T, 100)
    ratio = t2 / t1
    assert ratio == pytest.approx(np.log(100) / np.log(10), rel=1e-5)