from __future__ import annotations

import numpy as np
from typing import List, Optional, Tuple
from holographic_qc.core.ads_cft import AdsCft3


class MajoranaMode:
    def __init__(self, position: float, xi: float):
        self.position = position
        self.xi = xi

    def wavefunction(self, x: np.ndarray) -> np.ndarray:
        return np.exp(-np.abs(x - self.position) / self.xi) / np.sqrt(self.xi)

    def overlap(self, other: MajoranaMode) -> float:
        separation = abs(self.position - other.position)
        return np.exp(-separation / self.xi)


class MajoranaQubit:
    SIGMA_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    I2 = np.eye(2, dtype=np.complex128)

    def __init__(
        self, wire_length: float, coherence_length: float,
        fermi_velocity: float = 5e5, ads_system: Optional[AdsCft3] = None
    ):
        self.L = wire_length
        self.xi = coherence_length
        self.v_F = fermi_velocity
        self.ads = ads_system
        self.gamma1 = MajoranaMode(0.0, coherence_length)
        self.gamma2 = MajoranaMode(wire_length, coherence_length)
        self._state = np.array([1.0, 0.0], dtype=np.complex128)

    def parity_operator(self) -> np.ndarray:
        return 1j * self.SIGMA_Z

    def qubit_state(self, alpha: complex, beta: complex) -> np.ndarray:
        state = np.array([alpha, beta], dtype=np.complex128)
        norm = np.linalg.norm(state)
        return state / norm

    def topological_gap(self, pairing_potential: float) -> float:
        hbar = 1.054571817e-34
        return pairing_potential

    def topological_coherence_time(
        self, tau0: float, pairing_potential: float, temperature: float
    ) -> float:
        hbar = 1.054571817e-34
        kB = 1.380649e-23
        ratio = self.L / self.xi
        topo_factor = np.exp(ratio)
        return tau0 * topo_factor

    def holographic_enhancement_factor(self) -> float:
        if self.ads is None:
            return 1.0
        ratio = self.L / self.xi
        c = self.ads.central_charge
        return ratio**(c / 6.0)

    def total_coherence_time(self, tau0: float) -> float:
        ratio = self.L / self.xi
        topo = np.exp(ratio)
        holo = self.holographic_enhancement_factor()
        return tau0 * topo * holo

    def braiding_unitary(self, i: int, j: int) -> np.ndarray:
        angle = np.pi / 4.0
        if (i, j) == (0, 1) or (i, j) == (1, 0):
            return np.array([
                [np.cos(angle), -1j * np.sin(angle)],
                [-1j * np.sin(angle), np.cos(angle)]
            ], dtype=np.complex128)
        return self.I2.copy()

    def holographic_berry_phase(self) -> float:
        base = np.pi / 4.0
        correction = (self.xi / self.L) * np.log(self.L / self.xi)
        return base * (1.0 + correction)

    def gate_fidelity(self, epsilon0: float) -> float:
        ratio = self.L / self.xi
        c = self.ads.central_charge if self.ads else 1.0
        return 1.0 - epsilon0 * ratio**(-c / 6.0)

    def measurement_operator(self) -> np.ndarray:
        return self.SIGMA_Z.copy()

    def qnd_measurement_backaction(self, delta_m: float, mass_phi: float) -> float:
        if mass_phi <= 0:
            return delta_m
        return delta_m * np.exp(-mass_phi * self.xi)

    def density_matrix(self) -> np.ndarray:
        s = self._state.reshape(-1, 1)
        return s @ s.conj().T

    def apply_gate(self, U: np.ndarray):
        self._state = U @ self._state

    def fidelity_with(self, target_state: np.ndarray) -> float:
        return float(abs(np.dot(self._state.conj(), target_state))**2)

    def pairing_induced_gap(
        self, delta_sc: float, mu: float
    ) -> float:
        E_plus = np.sqrt((self.v_F * np.pi / self.L)**2 + delta_sc**2)
        return float(E_plus)

    def energy_splitting(self) -> float:
        hbar = 1.054571817e-34
        overlap = self.gamma1.overlap(self.gamma2)
        return hbar * self.v_F * overlap / self.xi

    def protection_diagram_points(self, n_points: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        L_over_xi = np.linspace(1, 50, n_points)
        coherence_times = np.exp(L_over_xi)
        return L_over_xi, coherence_times


class MajoranaFermionSystem:
    def __init__(self, n_sites: int, t: float = 1.0, delta: float = 1.0, mu: float = 0.0):
        self.n = n_sites
        self.t = t
        self.delta = delta
        self.mu = mu

    def bdg_hamiltonian(self) -> np.ndarray:
        n = self.n
        H = np.zeros((2 * n, 2 * n), dtype=np.complex128)
        for i in range(n):
            H[i, i] = -self.mu
            H[i + n, i + n] = self.mu
        for i in range(n - 1):
            H[i, i + 1] = -self.t
            H[i + 1, i] = -self.t
            H[i, i + 1 + n] = self.delta
            H[i + 1 + n, i] = np.conj(self.delta)
            H[i + n, i + 1] = -np.conj(self.delta)
            H[i + 1, i + n] = -self.delta
            H[i + n, i + 1 + n] = self.t
            H[i + 1 + n, i + n] = self.t
        return H

    def energy_spectrum(self) -> np.ndarray:
        H = self.bdg_hamiltonian()
        eigvals = np.linalg.eigvalsh(H)
        return np.sort(eigvals)

    def topological_invariant(self) -> int:
        if self.mu == 0:
            return 1
        ratio = abs(self.mu) / (2.0 * abs(self.t))
        return 1 if ratio < 1.0 else 0

    def majorana_operators(self) -> Tuple[np.ndarray, np.ndarray]:
        n = self.n
        gamma_A = np.zeros((2 * n, 2 * n), dtype=np.complex128)
        gamma_B = np.zeros((2 * n, 2 * n), dtype=np.complex128)
        for i in range(n):
            gamma_A[i, i + n] = 1.0
            gamma_A[i + n, i] = 1.0
            gamma_B[i, i + n] = -1j
            gamma_B[i + n, i] = 1j
        return gamma_A, gamma_B

    def zero_mode_wavefunction(self) -> np.ndarray:
        H = self.bdg_hamiltonian()
        eigvals, eigvecs = np.linalg.eigh(H)
        zero_idx = np.argmin(np.abs(eigvals))
        return eigvecs[:, zero_idx]