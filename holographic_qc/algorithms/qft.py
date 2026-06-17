from __future__ import annotations

import numpy as np
from typing import List, Optional


class QuantumFourierTransform:
    def __init__(self, n_qubits: int):
        self.n = n_qubits
        self.N = 2**n_qubits
        self._matrix = None

    def matrix(self) -> np.ndarray:
        if self._matrix is not None:
            return self._matrix
        N = self.N
        omega = np.exp(2j * np.pi / N)
        F = np.zeros((N, N), dtype=np.complex128)
        for j in range(N):
            for k in range(N):
                F[j, k] = omega**(j * k)
        F /= np.sqrt(N)
        self._matrix = F
        return F

    def inverse_matrix(self) -> np.ndarray:
        return self.matrix().conj().T

    def apply(self, state: np.ndarray) -> np.ndarray:
        if len(state) != self.N:
            raise ValueError(f"State must have length {self.N}")
        return self.matrix() @ state

    def apply_inverse(self, state: np.ndarray) -> np.ndarray:
        if len(state) != self.N:
            raise ValueError(f"State must have length {self.N}")
        return self.inverse_matrix() @ state

    def apply_fft(self, state: np.ndarray) -> np.ndarray:
        return np.fft.fft(state) / np.sqrt(len(state))

    def apply_ifft(self, state: np.ndarray) -> np.ndarray:
        return np.fft.ifft(state) * np.sqrt(len(state))

    def rotation_gate(self, k: int) -> np.ndarray:
        angle = 2.0 * np.pi / (2**k)
        return np.array([[1, 0], [0, np.exp(1j * angle)]], dtype=np.complex128)

    def hadamard(self) -> np.ndarray:
        return np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)

    def circuit_depth(self) -> int:
        return self.n * (self.n + 1) // 2

    def gate_count(self) -> int:
        return self.n + self.n * (self.n - 1) // 2

    def apply_to_register(self, state: np.ndarray, qubit_indices: List[int]) -> np.ndarray:
        F = self.matrix()
        return F @ state

    def phase_estimation_state(self, eigenvalue_phase: float) -> np.ndarray:
        state = np.zeros(self.N, dtype=np.complex128)
        for k in range(self.N):
            state[k] = np.exp(2j * np.pi * k * eigenvalue_phase)
        state /= np.sqrt(self.N)
        return state

    def estimate_phase(self, state: np.ndarray) -> float:
        amplitudes = np.abs(self.apply_inverse(state))**2
        best_k = int(np.argmax(amplitudes))
        return best_k / self.N

    def approximate_qft_matrix(self, n_terms: int) -> np.ndarray:
        N = self.N
        F = np.zeros((N, N), dtype=np.complex128)
        for j in range(N):
            for k in range(min(n_terms, N)):
                omega = np.exp(2j * np.pi * j * k / N)
                F[j, k] = omega
        F /= np.sqrt(N)
        return F

    def verify_unitarity(self, tol: float = 1e-10) -> bool:
        F = self.matrix()
        product = F @ F.conj().T
        return np.allclose(product, np.eye(self.N), atol=tol)

    def verify_inverse(self, tol: float = 1e-10) -> bool:
        F = self.matrix()
        F_inv = self.inverse_matrix()
        return np.allclose(F @ F_inv, np.eye(self.N), atol=tol)

    def bit_reversal_permutation(self, state: np.ndarray) -> np.ndarray:
        N = self.N
        n = self.n
        result = np.zeros(N, dtype=np.complex128)
        for i in range(N):
            reversed_i = int(bin(i)[2:].zfill(n)[::-1], 2)
            result[reversed_i] = state[i]
        return result