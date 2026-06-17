import numpy as np
import pytest
from holographic_qc.core.christoffel import ChristoffelSymbols


@pytest.fixture
def ads3_cs():
    return ChristoffelSymbols.from_ads3_poincare(ads_radius=1.1e-9, z=1e-10)


@pytest.fixture
def sphere_cs():
    return ChristoffelSymbols.from_sphere(radius=1.0, theta=np.pi / 4)


def test_ads3_metric_diagonal(ads3_cs):
    for i in range(3):
        for j in range(3):
            if i != j:
                assert ads3_cs.g[i, j] == pytest.approx(0.0, abs=1e-30)


def test_ads3_metric_positive_spatial(ads3_cs):
    assert ads3_cs.g[0, 0] > 0
    assert ads3_cs.g[1, 1] > 0
    assert ads3_cs.g[2, 2] < 0


def test_ads3_metric_inverse_product():
    cs = ChristoffelSymbols.from_ads3_poincare(1.1e-9, 1e-10)
    product = cs.g @ cs.g_inv
    assert np.allclose(product, np.eye(3), atol=1e-10)


def test_sphere_metric_diagonal(sphere_cs):
    assert sphere_cs.g[0, 1] == pytest.approx(0.0, abs=1e-30)
    assert sphere_cs.g[1, 0] == pytest.approx(0.0, abs=1e-30)


def test_sphere_metric_values():
    cs = ChristoffelSymbols.from_sphere(radius=2.0, theta=np.pi / 3)
    assert cs.g[0, 0] == pytest.approx(4.0, rel=1e-6)
    assert cs.g[1, 1] == pytest.approx(4.0 * np.sin(np.pi / 3)**2, rel=1e-6)


def test_geodesic_length_symmetric():
    cs = ChristoffelSymbols.from_ads3_poincare(1.1e-9, 1e-11)
    L12 = cs.ads3_geodesic_length(0.0, 1e-11, 10e-9, 1e-11, 1.1e-9)
    L21 = cs.ads3_geodesic_length(10e-9, 1e-11, 0.0, 1e-11, 1.1e-9)
    assert L12 == pytest.approx(L21, rel=1e-10)


def test_geodesic_length_boundary_formula():
    L_ads = 1.1e-9
    cs = ChristoffelSymbols.from_ads3_poincare(L_ads, 1e-11)
    ell, a = 100e-9, 0.3e-9
    L_geo = cs.ads3_geodesic_length_boundary(ell, a, L_ads)
    expected = 2.0 * L_ads * np.log(ell / a)
    assert L_geo == pytest.approx(expected, rel=1e-10)


def test_geodesic_length_positive():
    cs = ChristoffelSymbols.from_ads3_poincare(1e-9, 1e-11)
    L = cs.ads3_geodesic_length(0.0, 1e-11, 50e-9, 1e-11, 1e-9)
    assert L > 0.0


def test_geodesic_length_increases_with_separation():
    cs = ChristoffelSymbols.from_ads3_poincare(1e-9, 1e-11)
    L1 = cs.ads3_geodesic_length(0.0, 1e-11, 10e-9, 1e-11, 1e-9)
    L2 = cs.ads3_geodesic_length(0.0, 1e-11, 50e-9, 1e-11, 1e-9)
    L3 = cs.ads3_geodesic_length(0.0, 1e-11, 100e-9, 1e-11, 1e-9)
    assert L1 < L2 < L3


def test_metric_inverse_correctness():
    g = np.array([[2.0, 0.5], [0.5, 3.0]])
    cs = ChristoffelSymbols(g)
    assert np.allclose(cs.g @ cs.g_inv, np.eye(2), atol=1e-10)


def test_singular_metric_raises():
    g_singular = np.array([[1.0, 1.0], [1.0, 1.0]])
    with pytest.raises(Exception):
        ChristoffelSymbols(g_singular)