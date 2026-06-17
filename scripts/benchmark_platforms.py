import numpy as np
import pandas as pd
from holographic_qc.materials.bi2se3 import Bi2Se3
from holographic_qc.materials.hgte import HgTeCdTe
from holographic_qc.materials.trapped_ions import TrappedIonChain
from holographic_qc.core.ads_cft import AdsCft3
from holographic_qc.protection.decoherence import HolographicDecoherence
from holographic_qc.algorithms.shor import ShorAlgorithm
from holographic_qc.algorithms.grover import GroverAlgorithm


def benchmark_decoherence():
    print("=" * 70)
    print("DECOHERENCE SUPPRESSION BENCHMARK")
    print("=" * 70)
    temperatures = [4.0, 1.0, 0.1, 0.02]
    system_size = 1e-6

    for mat_cls, name in [(Bi2Se3, "Bi2Se3"), (HgTeCdTe, "HgTeCdTe")]:
        mat = mat_cls()
        ads = AdsCft3(central_charge=mat.central_charge, ads_radius=mat.xi, fermi_velocity=mat.fermi_velocity)
        dec = HolographicDecoherence(ads_system=ads, material=mat)
        print(f"\n{name} (c={mat.central_charge}, xi={mat.xi * 1e9:.2f} nm):")
        print(f"{'T [K]':>10} | {'T2_std [ns]':>14} | {'T2_holo [ns]':>14} | {'Enhancement':>12}")
        print("-" * 58)
        for T in temperatures:
            T2_std = mat.t2_standard_ns(T)
            T2_holo = mat.t2_holographic_ns(T, system_size)
            enh = T2_holo / T2_std
            print(f"{T:10.2f} | {T2_std:14.4f} | {T2_holo:14.4f} | {enh:12.4f}")

    print(f"\nTrapped Ions (Yb-171, N=50, c=0.5):")
    chain = TrappedIonChain()
    T_values = [1.0, 5.0, 10.0]
    print(f"{'T [mK]':>10} | {'lambda_L [s-1]':>16} | {'Chaos bound [s-1]':>18} | {'Ratio':>8}")
    print("-" * 60)
    for T_mK in T_values:
        lam = chain.lyapunov_exponent(T_mK)
        kB = 1.380649e-23
        hbar = 1.054571817e-34
        bound = 2.0 * np.pi * kB * T_mK * 1e-3 / hbar
        print(f"{T_mK:10.1f} | {lam:16.4e} | {bound:18.4e} | {lam / bound:8.4f}")


def benchmark_shor():
    print("\n" + "=" * 70)
    print("SHOR'S ALGORITHM BENCHMARK")
    print("=" * 70)
    test_numbers = [15, 21, 35, 77, 143]
    print(f"{'N':>6} | {'Factors':>16} | {'Verified':>10} | {'Resources'}") 
    print("-" * 70)
    for N in test_numbers:
        shor = ShorAlgorithm(N)
        result = shor.factor_classical_simulation()
        resources = shor.circuit_resource_estimate()
        if result:
            p, q = result
            verified = shor.verify_factorization(p, q)
            print(f"{N:6d} | {p:6d} x {q:6d}   | {str(verified):>10} | qubits={resources['n_logical_qubits']}, depth={resources['circuit_depth']}")
        else:
            print(f"{N:6d} | {'FAILED':>16} | {'False':>10} |")


def benchmark_grover():
    print("\n" + "=" * 70)
    print("GROVER'S ALGORITHM BENCHMARK")
    print("=" * 70)
    configs = [
        (6, [5]),
        (8, [42, 100]),
        (10, [300, 700, 900]),
    ]
    print(f"{'n_qubits':>10} | {'n_targets':>10} | {'Iterations':>12} | {'Succ_prob':>12} | {'Speedup':>10}")
    print("-" * 65)
    for n_qubits, targets in configs:
        grover = GroverAlgorithm(n_qubits, targets)
        result = grover.run_with_measurement(n_shots=500)
        resources = grover.circuit_resource_estimate()
        speedup_info = grover.verify_quadratic_speedup()
        print(f"{n_qubits:10d} | {len(targets):10d} | {result['n_iterations']:12d} | "
              f"{result['success_rate']:12.4f} | {speedup_info['speedup_factor']:10.2f}")


def benchmark_holographic_codes():
    print("\n" + "=" * 70)
    print("HOLOGRAPHIC ERROR CORRECTION BENCHMARK")
    print("=" * 70)
    from holographic_qc.protection.holographic_error_correction import HolographicCode
    from holographic_qc.core.ryu_takayanagi import RTConfig

    central_charges = [1.0, 2.0, 5.0]
    system_size = 1e-6
    lattice_spacing = 1e-9
    p_phys = 0.005

    print(f"{'c':>5} | {'d_std':>8} | {'d_holo':>10} | {'p_thresh_std':>14} | {'p_thresh_holo':>15} | {'logical_err':>12}")
    print("-" * 75)
    for c in central_charges:
        rt_cfg = RTConfig(central_charge=c, newton_constant_3d=3e-9 / (2 * c), ads_radius=1e-9)
        code = HolographicCode(code_type="surface", central_charge=c, rt_config=rt_cfg)
        d_holo = code.effective_distance(system_size, lattice_spacing)
        p_thresh_holo = code.error_threshold(p_std=0.01)
        log_err = code.logical_error_rate(p_phys, system_size, lattice_spacing)
        print(f"{c:5.1f} | {7:8d} | {d_holo:10.3f} | {0.01:14.4f} | {p_thresh_holo:15.6f} | {log_err:12.6e}")


if __name__ == "__main__":
    benchmark_decoherence()
    benchmark_shor()
    benchmark_grover()
    benchmark_holographic_codes()