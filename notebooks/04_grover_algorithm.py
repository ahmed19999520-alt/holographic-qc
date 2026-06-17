import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from holographic_qc.algorithms.grover import GroverAlgorithm, GroverOracle, DiffusionOperator
from holographic_qc.algorithms.qft import QuantumFourierTransform
from holographic_qc.utils.visualization import HolographicVisualizer

Path("figures").mkdir(exist_ok=True)
viz = HolographicVisualizer(output_dir="figures")

print("=" * 65)
print("Notebook 04: Grover's Algorithm")
print("=" * 65)

print("\n--- Section 1: Oracle and Diffusion Operators ---")
for n in [3, 4, 5]:
    oracle = GroverOracle(n, [5])
    diff = DiffusionOperator(n)
    U_oracle = oracle.matrix()
    U_diff = diff.matrix()
    unit_oracle = np.allclose(U_oracle @ U_oracle.conj().T, np.eye(2**n), atol=1e-10)
    unit_diff = np.allclose(U_diff @ U_diff.conj().T, np.eye(2**n), atol=1e-10)
    herm_diff = np.allclose(U_diff, U_diff.conj().T, atol=1e-10)
    print(
        f"  n={n}: oracle_unitary={unit_oracle}, "
        f"diffusion_unitary={unit_diff}, "
        f"diffusion_hermitian={herm_diff}"
    )

print("\n--- Section 2: Optimal Iterations and Success Probabilities ---")
print(f"{'n':>5} | {'M':>5} | {'N':>8} | {'k_opt':>8} | {'P_opt':>10} | {'speedup':>10}")
print("-" * 55)
configs = [
    (4, 1), (4, 2), (5, 1), (6, 1), (6, 4),
    (8, 1), (8, 8), (10, 1), (10, 16), (12, 1)
]
for n, M in configs:
    targets = list(range(M))
    grover = GroverAlgorithm(n, targets)
    k = grover.optimal_iterations()
    p = grover.success_probability(k)
    speedup = grover.verify_quadratic_speedup()
    sf = speedup["speedup_factor"]
    print(f"{n:5d} | {M:5d} | {2**n:8d} | {k:8d} | {p:10.6f} | {sf:10.3f}")

print("\n--- Section 3: Amplitude Evolution ---")
n_demo = 6
targets_demo = [42]
grover_demo = GroverAlgorithm(n_demo, targets_demo)
k_opt = grover_demo.optimal_iterations()

print(f"n={n_demo}, target={targets_demo[0]}, k_opt={k_opt}")
print(f"\n{'k':>5} | {'target_amp':>14} | {'nontarget_amp':>16} | {'P_success':>12}")
print("-" * 55)
for k in range(k_opt + 3):
    t_amp, nt_amp = grover_demo.amplitude_at_iteration(k)
    p = grover_demo.success_probability(k)
    print(f"{k:5d} | {t_amp:14.8f} | {nt_amp:16.10f} | {p:12.8f}")

print("\n--- Section 4: Statistical Measurement Results ---")
n_shots = 2000
configs_shot = [
    (6, [5]),
    (6, [5, 10, 20]),
    (8, [42]),
    (8, [100, 150, 200, 50]),
    (10, [512]),
]
print(f"{'n':>4} | {'targets':>22} | {'k_opt':>7} | {'P_theory':>10} | {'P_measured':>12} | {'success':>8}")
print("-" * 75)
for n, targets in configs_shot:
    g = GroverAlgorithm(n, targets)
    result = g.run_with_measurement(n_shots=n_shots)
    p_th = result["theoretical_success_prob"]
    p_meas = result["success_rate"]
    k = result["n_iterations"]
    targets_str = str(targets[:2]) + ("..." if len(targets) > 2 else "")
    print(
        f"{n:4d} | {targets_str:>22} | {k:7d} | {p_th:10.6f} | "
        f"{p_meas:12.6f} | {'OK' if abs(p_meas - p_th) < 0.05 else 'WARN':>8}"
    )

print("\n--- Section 5: Quantum Counting ---")
print(f"{'n':>5} | {'M_true':>8} | {'M_estimated':>14} | {'error %':>10}")
print("-" * 45)
for n, M in [(6, 1), (6, 4), (8, 1), (8, 16), (10, 4)]:
    targets = list(range(M))
    g = GroverAlgorithm(n, targets)
    M_est = g.quantum_counting_estimate(n_estimation_qubits=8)
    err_pct = abs(M_est - M) / M * 100
    print(f"{n:5d} | {M:8d} | {M_est:14.3f} | {err_pct:10.3f}")

print("\n--- Section 6: Amplitude Amplification (general) ---")
n_aa = 6
grover_aa = GroverAlgorithm(n_aa, [5, 10])
state0 = grover_aa.uniform_superposition()
amplified = grover_aa.amplitude_amplification(state0, None, n_iterations=8)
probs = np.abs(amplified)**2
target_prob = sum(probs[t] for t in [5, 10])
print(f"  Target probability after 8 iterations: {target_prob:.6f}")
print(f"  Best output state: index={np.argmax(probs)}, prob={np.max(probs):.6f}")

print("\n--- Section 7: Resource Comparison ---")
print("Grover vs classical search resource comparison:")
print(f"{'n':>5} | {'N=2^n':>12} | {'Classical calls':>17} | {'Grover calls':>14} | {'Speedup':>10}")
print("-" * 65)
for n in [4, 6, 8, 10, 12, 14, 16, 20]:
    g = GroverAlgorithm(n, [0])
    N = 2**n
    k = g.optimal_iterations()
    classical = N
    speedup = classical / max(k, 1)
    print(f"{n:5d} | {N:12d} | {classical:17d} | {k:14d} | {speedup:10.2f}")

print("\n--- Section 8: Generating Figures ---")
for n_fig, targets_fig in [(6, [5]), (8, [42, 100])]:
    viz.plot_grover_amplitudes(n_fig, targets_fig, n_iterations_range=range(0, 20))
print("Grover figures saved.")

print("\nDone. Grover's algorithm notebook complete.")