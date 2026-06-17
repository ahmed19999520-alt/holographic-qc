from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class TrappedIonChain:
    n_ions: int = 50
    axial_frequency_Hz: float = 1e6
    ion_spacing_um: float = 3.0
    J_coupling_Hz: float = 100.0
    transverse_field_ratio: float = 1.0
    species: str = "Yb171"
    T2_s: float = 10.0

    def __post_init__(self):
        self.central_charge = 0.5
        self.at_criticality = abs(self.transverse_field_ratio - 1.0) < 0.01

    def ising_hamiltonian(self, J: Optional[float] = None, h: Optional[float] = None) -> np.ndarray:
        J = J or self.J_coupling_Hz
        h = h or J * self.transverse_field_ratio
        n = self.n_ions
        H = np.zeros((2**n, 2**n), dtype=np.complex128) if n <= 12 else None
        if H is None:
            raise ValueError("Too many ions for exact diagonalization (n > 12)")
        sigma_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
        I2 = np.eye(2, dtype=np.complex128)

        def kron_op(op, site, n_sites):
            result = np.eye(1, dtype=np.complex128)
            for s in range(n_sites):
                result = np.kron(result, op if s == site else I2)
            return result

        for i in range(n - 1):
            Xi = kron_op(sigma_x, i, n)
            Xi1 = kron_op(sigma_x, i + 1, n)
            H -= J * (Xi @ Xi1)
        for i in range(n):
            Zi = kron_op(sigma_z, i, n)
            H -= h * Zi
        return H

    def ms_gate_unitary(self, theta: float, ion_pairs: list) -> np.ndarray:
        n = len(ion_pairs) * 2
        J_mat = np.zeros((n, n))
        for idx, (i, j) in enumerate(ion_pairs):
            J_mat[2 * idx, 2 * idx + 1] = 1.0
            J_mat[2 * idx + 1, 2 * idx] = 1.0
        phase_mat = np.zeros((2**n, 2**n), dtype=np.complex128)
        sigma_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        I2 = np.eye(2, dtype=np.complex128)
        SxSx_total = np.zeros((2**n, 2**n), dtype=np.complex128)
        for k in range(n // 2):
            for l in range(n // 2):
                ops = [I2] * n
                ops[2 * k] = sigma_x
                ops[2 * l] = sigma_x
                result = np.eye(1, dtype=np.complex128)
                for op in ops:
                    result = np.kron(result, op)
                SxSx_total += J_mat[2 * k, 2 * l] * result
        from scipy.linalg import expm
        return expm(-1j * theta / 2.0 * SxSx_total)

    def lyapunov_exponent(self, temperature_mK: float) -> float:
        kB = 1.380649e-23
        hbar = 1.054571817e-34
        T = temperature_mK * 1e-3
        bound = 2.0 * np.pi * kB * T / hbar
        correction = 6.0 / self.central_charge**2
        return bound * (1.0 - correction)

    def scrambling_time_ms(self, temperature_mK: float) -> float:
        kB = 1.380649e-23
        hbar = 1.054571817e-34
        T = temperature_mK * 1e-3
        beta = hbar / (kB * T)
        return (beta / (2.0 * np.pi)) * np.log(self.n_ions) * 1e3

    def entanglement_entropy_critical(self, subsystem_size: int) -> float:
        a_lat = 1
        return (self.central_charge / 3.0) * np.log(subsystem_size / a_lat)

    def holographic_decoherence_rate(self, temperature_mK: float) -> float:
        spontaneous_emission = 1e-4
        motional_heating = 1e-2
        base_rate = spontaneous_emission + motional_heating
        enhancement = self.n_ions**(-self.central_charge / 6.0)
        return base_rate * enhancement

    def otoc_protocol_steps(self) -> list:
        return [
            "Prepare product state |psi_0> = |up down up down ...>",
            "Apply local sigma_x at site i0 = n_ions // 2 (operator V)",
            "Evolve under H_Ising for time t",
            "Apply local sigma_x at site i (operator W)",
            "Evolve backward under H_Ising for time t",
            "Apply V^dagger at i0",
            "Measure overlap with initial state via Ramsey interferometry",
            "Repeat for different t and i to map scrambling front",
        ]

    def material_parameters_dict(self) -> dict:
        return {
            "n_ions": self.n_ions,
            "J_coupling_Hz": self.J_coupling_Hz,
            "central_charge": self.central_charge,
            "at_criticality": self.at_criticality,
            "T2_s": self.T2_s,
            "scrambling_time_ms_at_1mK": self.scrambling_time_ms(1.0),
        }