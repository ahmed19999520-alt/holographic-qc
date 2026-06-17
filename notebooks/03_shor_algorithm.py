import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from holographic_qc.algorithms.shor import ShorAlgorithm, PeriodFinder, ModularExponentiator
from holographic_qc.algorithms.qft import QuantumFourierTransform
from holographic_qc.utils.visualization import HolographicVisualizer

Path("figures").mkdir(exist_ok=True)
viz = HolographicVisualizer(output_dir="figures")

print("=" * 65)
print("Notebook 03: Shor's Algorithm")
print("=" * 65)

print("\n--- Section 1: Quantum Fourier Transform ---")
for n in [2, 3, 4, 5, 6, 8]:
    qft = QuantumFourierTransform(n)
    N = 2**n
    is_unitary = qft.verify_unitarity()
    is_inverse_ok = qft.verify_inverse()
    depth = qft.circuit_depth()
    gates = qft.gate_count()
    print(
        f"  n={n}: N={N:4d}, "
        f"depth={depth:4d}, "
        f"gates={gates:4d}, "
        f"unitary={is_unitary}, "
        f"inverse_ok={is_inverse_ok}"
    )

print("\n--- Section 2: QFT on Specific States ---")
qft6 = QuantumFourierTransform(6)
state = np.zeros(64, dtype=complex)
state[0] = 1.0
F_state = qft6.apply(state)
print(f"  QFT|0>: all amplitudes = {np.abs(F_state[0]):.6f} (expected 1/sqrt(64) = {1/8:.6f})")

state_comp = np.zeros(64, dtype=complex)
for k in range(64):
    state_comp[k] = np.exp(2j * np.pi * k * 5 / 64)
state_comp /= np.linalg.norm(state_comp)
F_comp = qft6.apply_inverse(state_comp)
peak = int(np.argmax(np.abs(F_comp)**2))
print(f"  IFT of phase-5 state: peak at k={peak} (expected 5)")

print("\n--- Section 3: Modular Exponentiation ---")
test_cases = [(7, 15), (11, 21), (7, 35)]
print(f"{'a':>5} | {'N':>6} | {'a^0 mod N':>12} | {'a^1 mod N':>12} | {'a^4 mod N':>12}")
print("-" * 55)
for a, N in test_cases:
    me = ModularExponentiator(a, N)
    print(f"{a:5d} | {N:6d} | {me.compute(0):12d} | {me.compute(1):12d} | {me.compute(4):12d}")

print("\n--- Section 4: Period Finding ---")
test_cases_period = [(7, 15, 8), (11, 21, 8), (2, 15, 8)]
print(f"{'a':>5} | {'N':>5} | {'True period r':>15} | {'Found r':>10} | {'Correct':>10}")
print("-" * 55)
for a, N, n_prec in test_cases_period:
    true_r = 1
    x = a % N
    while x != 1 and true_r < N:
        x = (x * a) % N
        true_r += 1
    finder = PeriodFinder(a, N, n_prec)
    found_r = finder.run(n_trials=20)
    correct = found_r == true_r if found_r else False
    print(f"{a:5d} | {N:5d} | {true_r:15d} | {str(found_r):>10} | {str(correct):>10}")

print("\n--- Section 5: Full Factoring ---")
test_numbers = [6, 15, 21, 35, 51, 77, 91, 143]
print(f"{'N':>6} | {'Factors':>18} | {'Verified':>10} | {'n_qubits':>10} | {'circuit_depth':>14}")
print("-" * 68)
for N in test_numbers:
    shor = ShorAlgorithm(N)
    result = shor.factor_classical_simulation(max_attempts=100)
    resources = shor.circuit_resource_estimate()
    if result:
        p, q = result
        ok = shor.verify_factorization(p, q)
        print(
            f"{N:6d} | {p:6d} x {q:8d}   | {str(ok):>10} | "
            f"{resources['n_logical_qubits']:>10d} | {resources['circuit_depth']:>14d}"
        )
    else:
        print(f"{N:6d} | {'FAILED':>18} | {'False':>10} | {'---':>10} | {'---':>14}")

print("\n--- Section 6: Success Probability Analysis ---")
N_test = 15
print(f"Success probability analysis for N={N_test}:")
shor15 = ShorAlgorithm(N_test)
print(f"{'a':>5} | {'gcd(a,N)':>10} | {'Period r':>10} | {'P_success':>12}")
print("-" * 44)
for a in range(2, N_test):
    g = math.gcd(a, N_test)
    if g > 1:
        p_succ = 0.0
        period_str = "N/A (gcd>1)"
    else:
        r = 1
        x = a % N_test
        while x != 1 and r < N_test:
            x = (x * a) % N_test
            r += 1
        period_str = str(r) if r < N_test else "None"
        p_succ = shor15.success_probability_estimate(a)
    print(f"{a:5d} | {g:10d} | {period_str:>10} | {p_succ:12.4f}")

print("\n--- Section 7: Resource Scaling ---")
print(f"{'N_bits':>8} | {'n_qubits':>10} | {'qft_gates':>12} | {'mod_exp_gates':>15} | {'total_depth':>12}")
print("-" * 65)
for n_bits in [3, 4, 5, 6, 7, 8, 10, 12]:
    N_sample = 2**n_bits - 1
    shor_s = ShorAlgorithm(N_sample)
    r = shor_s.circuit_resource_estimate()
    print(
        f"{n_bits:8d} | {r['n_logical_qubits']:10d} | "
        f"{r['n_qft_gates']:12d} | {r['n_modular_exp_gates']:15d} | "
        f"{r['circuit_depth']:12d}"
    )

print("\n--- Section 8: QFT Spectrum Visualization ---")
viz.plot_shor_qft_spectrum(7, 15, n_precision=8)
viz.plot_shor_qft_spectrum(11, 21, n_precision=8)
print("QFT spectrum figures saved.")

print("\nDone. Shor's algorithm notebook complete.")