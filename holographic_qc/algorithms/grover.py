from __future__ import annotations

import math
import numpy as np
from typing import Callable, List, Optional, Tuple


class GroverOracle:
    def __init__(self, n_qubits: int, target_states: List[int]):
        self.n = n_qubits
        self.N = 2**n_qubits
        self.targets = set(target_states)

    def matrix(self) -> np.ndarray:
        U = np.eye(self.N, dtype=np.complex128)
        for t in self.targets:
            if 0 <= t < self.N:
                U[t, t] = -1.0
        return U

    def apply(self, state: np.ndarray) -> np.ndarray:
        result = state.copy()
        for t in self.targets:
            result[t] *= -1.0
        return result

    def phase_kickback_oracle(self, state: np.ndarray) -> np.ndarray:
        return self.apply(state)

    def mark_state(self, index: int, state: np.ndarray) -> np.ndarray:
        result = state.copy()
        result[index] *= -1.0
        return result


class DiffusionOperator:
    def __init__(self, n_qubits: int):
        self.n = n_qubits
        self.N = 2**n_qubits
        self._uniform = np.ones(self.N, dtype=np.complex128) / np.sqrt(self.N)

    def matrix(self) -> np.ndarray:
        s = self._uniform.reshape(-1, 1)
        return 2.0 * (s @ s.conj().T) - np.eye(self.N, dtype=np.complex128)

    def apply(self, state: np.ndarray) -> np.ndarray:
        mean = np.dot(self._uniform.conj(), state)
        return 2.0 * mean * self._uniform - state

    def apply_explicit(self, state: np.ndarray) -> np.ndarray:
        mean_amp = np.mean(state)
        return 2.0 * mean_amp * np.ones(self.N, dtype=np.complex128) - state


class GroverAlgorithm:
    def __init__(self, n_qubits: int, target_states: List[int]):
        self.n = n_qubits
        self.N = 2**n_qubits
        self.oracle = GroverOracle(n_qubits, target_states)
        self.diffusion = DiffusionOperator(n_qubits)
        self.targets = target_states
        self.n_targets = len(target_states)

    def uniform_superposition(self) -> np.ndarray:
        return np.ones(self.N, dtype=np.complex128) / np.sqrt(self.N)

    def optimal_iterations(self) -> int:
        if self.n_targets == 0 or self.n_targets >= self.N:
            return 0
        theta = np.arcsin(np.sqrt(self.n_targets / self.N))
        return int(np.floor(np.pi / (4.0 * theta)))

    def success_probability(self, n_iterations: int) -> float:
        if self.n_targets == 0:
            return 0.0
        theta = np.arcsin(np.sqrt(self.n_targets / self.N))
        angle = (2 * n_iterations + 1) * theta
        return float(np.sin(angle)**2)

    def amplitude_at_iteration(self, k: int) -> Tuple[float, float]:
        if self.n_targets == 0 or self.n_targets >= self.N:
            return 0.0, 0.0
        theta = np.arcsin(np.sqrt(self.n_targets / self.N))
        angle = (2 * k + 1) * theta
        target_amp = np.sin(angle) / np.sqrt(self.n_targets)
        nontarget_amp = np.cos(angle) / np.sqrt(self.N - self.n_targets)
        return float(target_amp), float(nontarget_amp)

    def run(self, n_iterations: Optional[int] = None) -> Tuple[np.ndarray, int]:
        if n_iterations is None:
            n_iterations = self.optimal_iterations()
        state = self.uniform_superposition()
        for _ in range(n_iterations):
            state = self.oracle.apply(state)
            state = self.diffusion.apply(state)
        return state, n_iterations

    def run_with_measurement(
        self, n_shots: int = 1000, n_iterations: Optional[int] = None
    ) -> dict:
        state, n_iter = self.run(n_iterations)
        probs = np.abs(state)**2
        probs /= probs.sum()
        measurements = np.random.choice(self.N, size=n_shots, p=probs)
        counts = {}
        for m in measurements:
            counts[int(m)] = counts.get(int(m), 0) + 1
        success_count = sum(counts.get(t, 0) for t in self.targets)
        return {
            "state": state,
            "probabilities": probs,
            "measurements": counts,
            "success_rate": success_count / n_shots,
            "n_iterations": n_iter,
            "theoretical_success_prob": self.success_probability(n_iter),
        }

    def amplitude_amplification(
        self, initial_state: np.ndarray,
        target_oracle: np.ndarray,
        n_iterations: Optional[int] = None
    ) -> np.ndarray:
        if n_iterations is None:
            n_iterations = self.optimal_iterations()
        state = initial_state.copy()
        reflect_target = np.eye(self.N, dtype=np.complex128)
        for t in self.targets:
            reflect_target[t, t] = -1.0
        mean_state = initial_state.reshape(-1, 1)
        reflect_initial = 2.0 * (mean_state @ mean_state.conj().T) - np.eye(self.N, dtype=np.complex128)
        for _ in range(n_iterations):
            state = reflect_target @ state
            state = reflect_initial @ state
        return state

    def quantum_counting_estimate(self, n_estimation_qubits: int = 8) -> float:
        from holographic_qc.algorithms.qft import QuantumFourierTransform
        n_est = n_estimation_qubits
        qft = QuantumFourierTransform(n_est)
        dim_est = 2**n_est
        state = np.ones(dim_est, dtype=np.complex128) / np.sqrt(dim_est)
        theta = np.arcsin(np.sqrt(self.n_targets / self.N))
        phases = np.array([np.exp(2j * (2 * k + 1) * theta) for k in range(dim_est)])
        state = state * phases
        freq_state = qft.apply_inverse(state)
        probs = np.abs(freq_state)**2
        best_k = int(np.argmax(probs))
        theta_est = np.pi * best_k / dim_est
        M_est = self.N * np.sin(theta_est)**2
        return float(M_est)

    def circuit_resource_estimate(self) -> dict:
        n_iter = self.optimal_iterations()
        n_oracle_calls = n_iter
        n_diffusion_gates = n_iter * (self.n + 1)
        return {
            "n_qubits": self.n,
            "optimal_iterations": n_iter,
            "oracle_calls": n_oracle_calls,
            "total_gate_count": n_oracle_calls + n_diffusion_gates,
            "classical_search_calls": math.ceil(self.N / 2),
            "speedup_factor": math.ceil(self.N / 2) / max(1, n_oracle_calls),
        }

    def verify_quadratic_speedup(self) -> dict:
        classical_calls = self.N / self.n_targets if self.n_targets > 0 else self.N
        quantum_calls = self.optimal_iterations()
        speedup = classical_calls / max(1, quantum_calls)
        theoretical = np.sqrt(self.N / self.n_targets) if self.n_targets > 0 else np.sqrt(self.N)
        return {
            "classical_expected_calls": classical_calls,
            "quantum_iterations": quantum_calls,
            "speedup_factor": speedup,
            "theoretical_speedup": theoretical,
            "quadratic_speedup_confirmed": abs(speedup - theoretical) / theoretical < 0.5,
        }