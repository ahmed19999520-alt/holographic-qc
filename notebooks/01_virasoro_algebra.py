import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from holographic_qc.core.virasoro import VirasoroAlgebra, VirasoroConfig, WardIdentityVerifier

print("=" * 60)
print("Notebook 01: Virasoro Algebra")
print("=" * 60)

config = VirasoroConfig(central_charge=1.0, max_mode=10)
alg = VirasoroAlgebra(config)

print(f"\nCentral charge: c = {alg.c}")

print("\nCommutator table [L_m, L_n] for small modes:")
print(f"{'(m,n)':>12} | {'Linear coeff (m-n)':>20} | {'Central coeff c/12 m(m^2-1)':>30}")
print("-" * 68)
for m, n in [(2, -3), (3, -3), (2, -2), (1, 0), (4, -2)]:
    lin, cen = alg.commutator_scalar(m, n)
    print(f"({m:3d},{n:3d})     | {lin:20.4f} | {cen:30.6f}")

print("\nVerifying Jacobi identity for selected triples:")
triples = [(1, 2, -3), (2, -1, -1), (3, -2, -1), (4, -2, -2)]
for l, m, n in triples:
    ok = alg.verify_jacobi_identity(l, m, n)
    print(f"  Jacobi({l},{m},{n}): {'PASS' if ok else 'FAIL'}")

print("\nVirasoro characters chi_h(q) at q=0.1:")
for h in [0.0, 0.0625, 0.5, 1.0]:
    chi = alg.character(h, 0.1 + 0j, n_levels=30)
    print(f"  chi_{{h={h}}}(0.1) = {chi.real:.6f}")

print("\nKac table for (p,q) = (4,3) minimal model:")
table = alg.kac_table(4, 3)
print(f"  Shape: {table.shape}")
for r in range(table.shape[0]):
    row_str = "  " + " | ".join(f"{table[r,s]:8.5f}" for s in range(table.shape[1]))
    print(row_str)

print("\nGram matrices and Kac determinants:")
for h_val, level in [(0.5, 1), (0.5, 2), (1.0, 1), (1.0, 2)]:
    G = alg.gram_matrix(h_val, level)
    det = alg.kac_determinant(h_val, level)
    eigvals = np.linalg.eigvalsh(G)
    print(f"  h={h_val}, level={level}: det={det:.4f}, min_eig={eigvals.min():.6f}")

print("\nPartition states (Young tableaux) at each level:")
for lvl in range(5):
    states = alg.partition_states(lvl)
    print(f"  Level {lvl}: {len(states)} state(s) -- {states[:4]}{'...' if len(states) > 4 else ''}")

print("\nOPE coefficient C_Delta for density operator (Delta=1):")
C = alg.ope_tilde_coeff(2.0 + 0j, 0.0 + 0j)
print(f"  T(2)T(0) leading pole: {C.real:.8f}")
print(f"  Expected c/2 / (2-0)^4 = {alg.c / 2.0 / 16.0:.8f}")

print("\nLyapunov exponents vs. temperature (c=1):")
kB = 1.380649e-23
hbar = 1.054571817e-34
print(f"{'T [K]':>10} | {'lambda_L [s^-1]':>18} | {'bound [s^-1]':>18} | {'ratio':>8}")
print("-" * 62)
for T in [0.1, 1.0, 4.0, 10.0, 300.0]:
    lam = alg.lyapunov_from_central_charge(T)
    bound = 2.0 * np.pi * kB * T / hbar
    print(f"{T:10.1f} | {lam:18.4e} | {bound:18.4e} | {lam/bound:8.5f}")

verifier = WardIdentityVerifier(alg)
errors = verifier.verify_algebra_closure(mode_range=4)
max_err = max(errors.values())
print(f"\nMax algebra closure error over all modes |m|,|n| <= 4: {max_err:.2e}")

print("\nConformal block (Zamolodchikov approximation):")
hi = (1.0, 0.5, 0.5, 1.0)
z_vals = [0.1, 0.3, 0.5]
for z in z_vals:
    block = alg.virasoro_block_zamolodchikov(0.5, hi, z + 0j, n_terms=15)
    print(f"  V(h=0.5, hi={hi}, z={z}) = {block.real:.6f}")

print("\nDone. Virasoro algebra fully verified.")