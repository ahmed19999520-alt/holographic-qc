import numpy as np
from holographic_qc.core.virasoro import VirasoroAlgebra, VirasoroConfig
from holographic_qc.core.ads_cft import AdsCft3
from holographic_qc.core.ryu_takayanagi import RyuTakayanagi, RTConfig
from holographic_qc.materials.bi2se3 import Bi2Se3
from holographic_qc.materials.hgte import HgTeCdTe
from holographic_qc.materials.trapped_ions import TrappedIonChain
from holographic_qc.protection.decoherence import HolographicDecoherence
from holographic_qc.protection.majorana import MajoranaFermionSystem


def validate_virasoro_algebra():
    print("Validating Virasoro algebra...")
    alg = VirasoroAlgebra(VirasoroConfig(central_charge=1.0, max_mode=10))
    kB, hbar = 1.380649e-23, 1.054571817e-34
    for T in [0.1, 1.0, 4.0, 10.0, 300.0]:
        lam = alg.lyapunov_from_central_charge(T)
        bound = 2.0 * np.pi * kB * T / hbar
        assert lam <= bound + 1e-10, f"FAIL: lambda_L={lam} > bound={bound} at T={T}K"
    for m, n in [(3, -3), (4, -4), (5, -5)]:
        lin, cen = alg.commutator_scalar(m, n)
        expected_lin = float(m - n)
        expected_cen = (1.0 / 12.0) * m * (m * m - 1) if m + n == 0 else 0.0
        assert abs(lin - expected_lin) < 1e-10
        assert abs(cen - expected_cen) < 1e-10
    print("  Virasoro algebra: PASS")


def validate_brown_henneaux():
    print("Validating Brown-Henneaux formula...")
    for c in [0.5, 1.0, 2.0, 5.0]:
        ads = AdsCft3(central_charge=c, ads_radius=1.1e-9)
        c_computed = ads.brown_henneaux_central_charge()
        assert abs(c_computed - c) < 1e-6, f"FAIL: c={c}, computed={c_computed}"
    print("  Brown-Henneaux formula: PASS")


def validate_ryu_takayanagi():
    print("Validating Ryu-Takayanagi formula...")
    for c in [0.5, 1.0, 2.0]:
        ads = AdsCft3(central_charge=c, ads_radius=1.1e-9)
        rt = RyuTakayanagi(RTConfig(
            central_charge=c, newton_constant_3d=ads.newton_constant_3d,
            ads_radius=ads.ads_radius, uv_cutoff=0.3e-9
        ))
        for ell_nm in [10, 50, 100, 200]:
            ell = ell_nm * 1e-9
            S_formula = (c / 3.0) * np.log(ell / 0.3e-9)
            S_rt = rt.entanglement_entropy_central_charge(ell)
            assert abs(S_rt - S_formula) < 1e-10, f"FAIL: c={c}, ell={ell_nm}nm"
        S2 = rt.renyi_entropy(100e-9, 2)
        S3 = rt.renyi_entropy(100e-9, 3)
        S1 = rt.renyi_entropy(100e-9, 1)
        assert S1 >= S2 >= S3 - 1e-10, f"FAIL: Renyi ordering c={c}"
    print("  Ryu-Takayanagi formula: PASS")


def validate_holographic_decoherence():
    print("Validating holographic decoherence rates...")
    mat = Bi2Se3()
    ads = AdsCft3(central_charge=mat.central_charge, ads_radius=mat.xi)
    dec = HolographicDecoherence(ads, mat)
    for T in [0.1, 1.0, 4.0]:
        r1 = dec.standard_phonon_rate_2d(T)
        r2 = dec.standard_phonon_rate_2d(2.0 * T)
        assert abs(r2 / r1 - 4.0) < 1e-4, f"FAIL: quadratic scaling at T={T}"
    for L in [100e-9, 1e-6, 10e-6]:
        enh = dec.coherence_time_ratio(L, mat.xi)
        expected = (L / mat.xi)**(mat.central_charge / 6.0)
        assert abs(enh - expected) < 1e-8, f"FAIL: enhancement at L={L}"
        assert enh >= 1.0, f"FAIL: enhancement < 1 at L={L}"
    ads2 = AdsCft3(central_charge=2.0, ads_radius=mat.xi)
    dec2 = HolographicDecoherence(ads2, mat)
    L_test = 1e-6
    enh1 = dec.coherence_time_ratio(L_test, mat.xi)
    enh2 = dec2.coherence_time_ratio(L_test, mat.xi)
    assert enh2 > enh1, "FAIL: c=2 should give larger enhancement than c=1"
    print("  Holographic decoherence: PASS")


def validate_majorana_spectrum():
    print("Validating Majorana BdG spectrum...")
    for n in [4, 6, 8]:
        for mu in [0.0, 1.0, 3.0]:
            sys_m = MajoranaFermionSystem(n_sites=n, t=1.0, delta=1.0, mu=mu)
            spectrum = sys_m.energy_spectrum()
            H = sys_m.bdg_hamiltonian()
            assert np.allclose(H, H.conj().T, atol=1e-10), f"FAIL: H not Hermitian n={n}"
            assert len(spectrum) == 2 * n, f"FAIL: spectrum size n={n}"
            assert np.all(np.diff(spectrum) >= -1e-10), f"FAIL: spectrum not sorted n={n}"
            Z = sys_m.topological_invariant()
            if mu == 0.0:
                assert Z == 1, f"FAIL: topo_inv=0 at mu=0 n={n}"
    print("  Majorana spectrum: PASS")


def validate_optical_conductivity():
    print("Validating optical conductivity scaling...")
    e_sq_over_h = 3.87404e-5
    for c in [0.5, 1.0, 2.0, 5.0]:
        ads = AdsCft3(central_charge=c, ads_radius=1e-9)
        sigma = ads.optical_conductivity_dc()
        expected = e_sq_over_h * c / 2.0
        assert abs(sigma - expected) < 1e-12, f"FAIL: sigma c={c}"
    ads1 = AdsCft3(central_charge=1.0, ads_radius=1e-9)
    ads2 = AdsCft3(central_charge=2.0, ads_radius=1e-9)
    assert abs(ads2.optical_conductivity_dc() / ads1.optical_conductivity_dc() - 2.0) < 1e-8
    print("  Optical conductivity: PASS")


def validate_wiedemann_franz():
    print("Validating Wiedemann-Franz law violation...")
    kB, e = 1.380649e-23, 1.602176634e-19
    L0 = (np.pi**2 / 3.0) * kB**2 / e**2
    for c in [1.0, 2.0, 5.0, 10.0, 1000.0]:
        ads = AdsCft3(central_charge=c, ads_radius=1e-9)
        ratio = ads.wiedemann_franz_ratio()
        expected = L0 * (1.0 - 3.0 / c)
        assert abs(ratio - expected) < 1e-50, f"FAIL: WF c={c}"
    ads_large = AdsCft3(central_charge=1e6, ads_radius=1e-9)
    assert abs(ads_large.wiedemann_franz_ratio() - L0) < L0 * 1e-5
    print("  Wiedemann-Franz violation: PASS")


def validate_chaos_bound():
    print("Validating quantum chaos bound...")
    kB, hbar = 1.380649e-23, 1.054571817e-34
    for T in [0.01, 0.1, 1.0, 4.0, 300.0]:
        bound = 2.0 * np.pi * kB * T / hbar
        for c in [0.5, 1.0, 2.0, 5.0, 100.0]:
            ads = AdsCft3(central_charge=c, ads_radius=1e-9)
            lam = ads.lyapunov_exponent(T)
            assert lam <= bound + 1e-10, f"FAIL: lambda_L > bound at T={T}K, c={c}"
        ion = TrappedIonChain(n_ions=50)
        lam_ion = ion.lyapunov_exponent(T * 1e3)
        bound_ion = 2.0 * np.pi * kB * T / hbar
        assert lam_ion <= bound_ion + 1e-10
    print("  Quantum chaos bound: PASS")


def main():
    print("=" * 60)
    print("HolographicQC Physics Validation Suite")
    print("=" * 60)
    validate_virasoro_algebra()
    validate_brown_henneaux()
    validate_ryu_takayanagi()
    validate_holographic_decoherence()
    validate_majorana_spectrum()
    validate_optical_conductivity()
    validate_wiedemann_franz()
    validate_chaos_bound()
    print("\n" + "=" * 60)
    print("All physics validations passed.")
    print("=" * 60)


if __name__ == "__main__":
    main()