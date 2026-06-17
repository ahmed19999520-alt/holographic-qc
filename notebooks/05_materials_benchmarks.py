import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from holographic_qc.materials.bi2se3 import Bi2Se3
from holographic_qc.materials.hgte import HgTeCdTe
from holographic_qc.materials.trapped_ions import TrappedIonChain
from holographic_qc.core.ads_cft import AdsCft3
from holographic_qc.core.ryu_takayanagi import RyuTakayanagi, RTConfig
from holographic_qc.protection.decoherence import HolographicDecoherence
from holographic_qc.protection.majorana import MajoranaFermionSystem
from holographic_qc.algorithms.vqe import VQE
from holographic_qc.utils.benchmarks import Benchmarker
from holographic_qc.utils.datasets import SyntheticDatasetGenerator
from holographic_qc.utils.visualization import HolographicVisualizer

Path("figures").mkdir(exist_ok=True)
viz = HolographicVisualizer(output_dir="figures")

print("=" * 65)
print("Notebook 05: Materials Benchmarks and VQE")
print("=" * 65)

print("\n--- Section 1: Material Parameter Summary ---")
materials = [Bi2Se3(), HgTeCdTe()]
for mat in materials:
    params = mat.material_parameters_dict()
    name = mat.__class__.__name__
    print(f"\n{name}:")
    for k, v in params.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4e}")
        else:
            print(f"  {k}: {v}")

print("\n--- Section 2: Coherence Enhancement vs System Size ---")
L_values = np.array([100, 200, 500, 1000, 2000, 5000, 10000]) * 1e-9
bi = Bi2Se3()
hg = HgTeCdTe()
ads_bi = AdsCft3(central_charge=bi.central_charge, ads_radius=bi.xi)
ads_hg = AdsCft3(central_charge=hg.central_charge, ads_radius=hg.xi)
dec_bi = HolographicDecoherence(ads_bi, bi)
dec_hg = HolographicDecoherence(ads_hg, hg)

print(f"\n{'L [um]':>10} | {'Bi2Se3 enh':>12} | {'HgTe enh':>12} | {'5-layer Bi enh':>16}")
print("-" * 56)
for L in L_values:
    enh_bi = dec_bi.coherence_time_ratio(L, bi.xi)
    enh_hg = dec_hg.coherence_time_ratio(L, hg.xi)
    ads_5l = AdsCft3(central_charge=5.0, ads_radius=bi.xi)
    dec_5l = HolographicDecoherence(ads_5l, bi)
    enh_5l = dec_5l.coherence_time_ratio(L, bi.xi)
    print(f"{L*1e6:10.3f} | {enh_bi:12.4f} | {enh_hg:12.4f} | {enh_5l:16.4f}")

print("\n--- Section 3: Transport Coefficients ---")
T_list = [0.1, 1.0, 4.0, 10.0, 77.0, 300.0]
print(f"\nHgTe transport coefficients:")
print(f"{'T [K]':>8} | {'sigma_dc [S]':>14} | {'L_ratio':>10} | {'WF_violation':>14}")
print("-" * 52)
for T in T_list:
    coeffs = hg.transport_coefficients(T)
    print(
        f"{T:8.1f} | {coeffs['sigma_dc_S']:14.4e} | "
        f"{coeffs['lorenz_ratio']:10.4e} | "
        f"{coeffs['WF_violation_fraction']:14.4f}"
    )

print("\n--- Section 4: Trapped Ion OTOC and Scrambling ---")
chain = TrappedIonChain(n_ions=50)
kB = 1.380649e-23
hbar = 1.054571817e-34

print(f"\nIon chain parameters: {chain.material_parameters_dict()}")
print(f"\n{'T [mK]':>10} | {'lambda_L [s-1]':>16} | {'bound [s-1]':>14} | {'t_scramble [ms]':>16}")
print("-" * 62)
for T_mK in [0.1, 0.5, 1.0, 5.0, 10.0]:
    lam = chain.lyapunov_exponent(T_mK)
    bound = 2.0 * np.pi * kB * T_mK * 1e-3 / hbar
    t_s = chain.scrambling_time_ms(T_mK)
    print(f"{T_mK:10.2f} | {lam:16.4e} | {bound:14.4e} | {t_s:16.4f}")

print("\n--- Section 5: Entanglement Entropy vs Subsystem Size ---")
print(f"\n{'n_sites':>9} | {'EE (c=0.5)':>12} | {'S_2 (Renyi)':>14} | {'slope':>8}")
print("-" * 50)
rt_config = RTConfig(
    central_charge=chain.central_charge,
    newton_constant_3d=1e-9,
    ads_radius=1e-9,
    uv_cutoff=1.0,
)
rt = RyuTakayanagi(rt_config)
prev_EE = None
for n_sub in [2, 4, 8, 12, 16, 20, 25]:
    EE = chain.entanglement_entropy_critical(n_sub)
    S2 = rt.renyi_entropy(float(n_sub), 2)
    slope = (EE - prev_EE) / np.log(2) if prev_EE else float("nan")
    print(f"{n_sub:9d} | {EE:12.6f} | {S2:14.6f} | {slope:8.4f}")
    prev_EE = EE

print("\n--- Section 6: VQE for Critical Ising Model ---")
n_vqe = 4
H_ising = VQE.ising_hamiltonian(n_vqe, J=1.0, h=1.0)
print(f"\nTransverse-field Ising Hamiltonian (n={n_vqe} sites, J=h=1 [critical]):")
eigvals_exact = np.linalg.eigvalsh(H_ising)
print(f"  Exact eigenvalues: {eigvals_exact[:4]}")
print(f"  Ground state energy: {eigvals_exact[0]:.8f}")

for depth in [1, 2, 3, 4]:
    vqe = VQE(H_ising, n_vqe, depth=depth, max_iter=200)
    result = vqe.run(seed=42)
    print(
        f"  Depth {depth}: "
        f"E_VQE={result['optimal_energy']:.6f}, "
        f"E_exact={result['exact_ground_energy']:.6f}, "
        f"err={result['energy_error']:.2e}, "
        f"converged={result['converged']}, "
        f"iters={result['n_iterations']}"
    )

print("\n--- Section 7: VQE Heisenberg Model ---")
n_heis = 4
H_heis = VQE.heisenberg_hamiltonian(n_heis, J=1.0)
eigvals_heis = np.linalg.eigvalsh(H_heis)
print(f"\nHeisenberg chain (n={n_heis}): E_0={eigvals_heis[0]:.6f}")
vqe_heis = VQE(H_heis, n_heis, depth=3, max_iter=300)
res_heis = vqe_heis.run(seed=123)
print(
    f"  VQE: E={res_heis['optimal_energy']:.6f}, "
    f"error={res_heis['energy_error']:.4e}, "
    f"variance={vqe_heis.variance(res_heis['optimal_params']):.4e}"
)

print("\n--- Section 8: VQE Figure ---")
vqe_fig = VQE(H_ising, n_vqe, depth=3, max_iter=300)
res_fig = vqe_fig.run(seed=7)
viz.plot_vqe_convergence(res_fig["energy_history"], res_fig["exact_ground_energy"])
print("VQE convergence figure saved.")

print("\n--- Section 9: Synthetic Dataset Statistics ---")
gen = SyntheticDatasetGenerator(seed=0)
df_bi = gen.generate_bi2se3_dataset(n_samples=1000)
print(f"\nBi2Se3 dataset statistics (n=1000):")
print(df_bi[["temperature_K", "ratio_L_xi", "enhancement_factor", "t2_holographic_ns"]].describe().to_string())

df_ee = gen.generate_entanglement_entropy_dataset(n_samples=800)
print(f"\nEntanglement entropy dataset statistics (n=800):")
print(df_ee[["central_charge", "interval_length", "entropy_theory", "entropy_measured"]].describe().to_string())

print("\n--- Section 10: Performance Benchmarks ---")
bench = Benchmarker()
bench.run_all()

print("\nDone. Materials benchmarks notebook complete.")