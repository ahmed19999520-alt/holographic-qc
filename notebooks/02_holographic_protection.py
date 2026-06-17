import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from holographic_qc.core.ads_cft import AdsCft3
from holographic_qc.core.ryu_takayanagi import RyuTakayanagi, RTConfig
from holographic_qc.core.dilaton import DilatonField, DilatonConfig
from holographic_qc.protection.decoherence import HolographicDecoherence, LindbladEvolution
from holographic_qc.protection.majorana import MajoranaQubit, MajoranaFermionSystem
from holographic_qc.materials.bi2se3 import Bi2Se3
from holographic_qc.materials.hgte import HgTeCdTe
from holographic_qc.materials.trapped_ions import TrappedIonChain
from holographic_qc.utils.visualization import HolographicVisualizer

Path("figures").mkdir(exist_ok=True)
viz = HolographicVisualizer(output_dir="figures")

print("=" * 65)
print("Notebook 02: Holographic Protection Mechanism")
print("=" * 65)

print("\n--- Section 1: AdS3/CFT2 Dictionary ---")
mat_bi = Bi2Se3()
ads_bi = AdsCft3(
    central_charge=mat_bi.central_charge,
    ads_radius=mat_bi.xi,
    fermi_velocity=mat_bi.fermi_velocity,
)
print(f"Bi2Se3 parameters:")
for k, v in mat_bi.material_parameters_dict().items():
    print(f"  {k}: {v:.4e}")
print(f"  ads_radius (= xi): {ads_bi.ads_radius:.4e} m")
print(f"  G_3 (Newton const): {ads_bi.newton_constant_3d:.4e}")
print(f"  c (Brown-Henneaux): {ads_bi.brown_henneaux_central_charge():.4f}")

print("\n--- Section 2: Ryu-Takayanagi Formula ---")
rt_config = RTConfig(
    central_charge=mat_bi.central_charge,
    newton_constant_3d=ads_bi.newton_constant_3d,
    ads_radius=ads_bi.ads_radius,
    uv_cutoff=mat_bi.a_lattice,
)
rt = RyuTakayanagi(rt_config)

ell_values = np.array([5, 10, 20, 50, 100, 200, 500]) * 1e-9
print(f"{'ell [nm]':>10} | {'S_A (geodesic)':>16} | {'S_A (c/3 log)':>15} | {'S_2 (Renyi)':>13}")
print("-" * 60)
for ell in ell_values:
    S_geo = rt.entanglement_entropy(ell)
    S_cc = rt.entanglement_entropy_central_charge(ell)
    S2 = rt.renyi_entropy(ell, 2)
    print(f"{ell*1e9:10.1f} | {S_geo:16.6f} | {S_cc:15.6f} | {S2:13.6f}")

print("\nBulk-to-boundary propagator K(z, x; x') at x-x'=10 nm:")
r = 10e-9
print(f"{'z [nm]':>10} | {'K(z) Delta=1':>15} | {'K(z) Delta=1.5':>16}")
print("-" * 46)
for z_nm in [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]:
    z = z_nm * 1e-9
    K1 = ads_bi.bulk_to_boundary_propagator(z, 0.0, r, 1.0)
    K15 = ads_bi.bulk_to_boundary_propagator(z, 0.0, r, 1.5)
    print(f"{z_nm:10.2f} | {K1:15.6e} | {K15:16.6e}")

print("\n--- Section 3: Holographic Decoherence Suppression ---")
dec_bi = HolographicDecoherence(ads_bi, mat_bi)

temperatures = [0.02, 0.1, 1.0, 4.0, 10.0]
system_sizes = [100e-9, 1e-6, 10e-6]

print("\nBi2Se3: Enhancement factor (L/xi)^(c/6):")
print(f"{'T [K]':>8} | {'L=100nm':>10} | {'L=1um':>10} | {'L=10um':>10}")
print("-" * 46)
for T in temperatures:
    row = f"{T:8.2f} |"
    for L in system_sizes:
        enh = dec_bi.coherence_time_ratio(L, mat_bi.xi)
        row += f" {enh:10.4f} |"
    print(row)

print("\nCoherence times at T=4K, L=1um:")
T_test = 4.0
L_test = 1e-6
T2_std = dec_bi.coherence_time_standard(T_test)
T2_holo = dec_bi.coherence_time_holographic(T_test, L_test)
print(f"  T2_standard  = {T2_std * 1e9:.4f} ns")
print(f"  T2_holo      = {T2_holo * 1e9:.4f} ns")
print(f"  Ratio        = {T2_holo / T2_std:.4f}")

mat_hg = HgTeCdTe()
ads_hg = AdsCft3(
    central_charge=mat_hg.central_charge,
    ads_radius=mat_hg.xi,
    fermi_velocity=mat_hg.fermi_velocity,
)
dec_hg = HolographicDecoherence(ads_hg, mat_hg)

print("\nHgTe/CdTe: Enhancement factor (L/xi)^(c/6):")
print(f"{'T [K]':>8} | {'L=100nm':>10} | {'L=1um':>10} | {'L=10um':>10}")
print("-" * 46)
for T in temperatures:
    row = f"{T:8.2f} |"
    for L in system_sizes:
        enh = dec_hg.coherence_time_ratio(L, mat_hg.xi)
        row += f" {enh:10.4f} |"
    print(row)

print("\n--- Section 4: Lindblad Evolution ---")
H_qubit = 0.5 * np.array([[1, 0], [0, -1]], dtype=np.complex128)
sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
gamma_std = dec_bi.standard_phonon_rate_2d(4.0)
gamma_holo = dec_bi.holographic_decoherence_rate(4.0, 1e-6, mat_bi.xi)

print(f"  gamma_std  = {gamma_std:.4e} s^-1")
print(f"  gamma_holo = {gamma_holo:.4e} s^-1")
print(f"  Suppression factor = {gamma_holo / gamma_std:.6e}")

ev_std = LindbladEvolution(H_qubit, [sigma_z], [gamma_std])
ev_holo = LindbladEvolution(H_qubit, [sigma_z], [gamma_holo])

rho0 = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.complex128)
t_max = 2.0 / gamma_std
times = np.linspace(0, t_max, 200)
coh_std = ev_std.coherence_decay(rho0, times)
coh_holo = ev_holo.coherence_decay(rho0, times)
T2_fit_std = ev_std.T2_from_decay(coh_std, times)
T2_fit_holo = ev_holo.T2_from_decay(coh_holo, times)
print(f"\n  T2 from Lindblad fit (standard): {T2_fit_std * 1e9:.4f} ns")
print(f"  T2 from Lindblad fit (holo):     {T2_fit_holo * 1e9:.4f} ns")
print(f"  Ratio: {T2_fit_holo / T2_fit_std:.4f}")

print("\n--- Section 5: Majorana Qubit ---")
for L_nm, c in [(20, 1.0), (50, 1.0), (100, 1.0), (100, 2.0)]:
    ads_m = AdsCft3(central_charge=c, ads_radius=mat_bi.xi)
    qubit = MajoranaQubit(L_nm * 1e-9, mat_bi.xi, mat_bi.fermi_velocity, ads_m)
    tau0 = 1e-6
    T2_total = qubit.total_coherence_time(tau0)
    F = qubit.gate_fidelity(0.01)
    phi_berry = qubit.holographic_berry_phase()
    split = qubit.energy_splitting()
    print(
        f"  L={L_nm:4d}nm, c={c}: "
        f"T2={T2_total:.2e}s, "
        f"F={F:.6f}, "
        f"phi_Berry={phi_berry:.6f} rad, "
        f"dE={split:.2e} J"
    )

print("\n--- Section 6: BdG Spectrum (Majorana system, n=6 sites) ---")
for mu_val in [0.0, 1.0, 2.0, 3.0]:
    sys = MajoranaFermionSystem(n_sites=6, t=1.0, delta=1.0, mu=mu_val)
    spectrum = sys.energy_spectrum()
    Z = sys.topological_invariant()
    gap = spectrum[len(spectrum)//2] - spectrum[len(spectrum)//2 - 1]
    print(f"  mu={mu_val:.1f}: topo_inv={Z}, gap={gap:.4f}, min_E={spectrum.min():.4f}")

print("\n--- Section 7: Temperature Scan ---")
T_arr = np.array([0.02, 0.05, 0.1, 0.5, 1.0, 4.0])
enh_T = dec_bi.temperature_dependence_enhancement(T_arr, 1e-6)
print(f"{'T [K]':>10} | {'Enhancement':>14}")
print("-" * 28)
for T, e in zip(T_arr, enh_T):
    print(f"{T:10.3f} | {e:14.6f}")

print("\n--- Generating Figures ---")
L_xi = np.linspace(1.0, 50.0, 300)
viz.plot_coherence_enhancement(L_xi, [0.5, 1.0, 2.0, 5.0])

T_range = np.logspace(-2, 1, 150)
viz.plot_materials_comparison(T_range)

t_range = np.linspace(0.01, 5.0, 300)
F_holo = np.exp(-2.0 * t_range)
F_int = 0.5 + 0.5 * np.cos(2.0 * t_range)
F_gen = np.exp(-1.2 * t_range)
viz.plot_otoc(t_range, F_holo, F_int, F_gen)

ell_plot = np.array([2, 5, 10, 20, 50, 100])
S_meas = (1.0 / 3.0) * np.log(ell_plot) + 0.05 * np.random.randn(len(ell_plot))
viz.plot_entanglement_entropy_scaling(
    ell_plot.astype(float), S_meas, [0.5, 1.0, 2.0]
)

x_vals = np.linspace(2e-9, 200e-9, 200)
C_cft = 1.0 / x_vals**2
C_holo = C_cft * (1.0 + (1.0 / (12.0 * np.pi**2)) * np.log(x_vals / mat_bi.xi))
noise = 0.03 * np.random.randn(len(x_vals))
C_meas = C_holo * (1.0 + noise)
viz.plot_density_correlations(x_vals * 1e9, C_meas, C_cft, C_holo)

print("All figures saved to figures/")
print("\nDone. Holographic protection notebook complete.")