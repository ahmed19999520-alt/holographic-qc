from __future__ import annotations

import numpy as np
import scipy.integrate as integrate
from dataclasses import dataclass
from typing import Optional, Tuple
from holographic_qc.core.ads_cft import AdsCft3


@dataclass
class DecoherenceConfig:
    temperature: float
    phonon_coupling: float = 1e-3
    charge_noise_amplitude: float = 1e-6
    magnetic_impurity_density: float = 1e-4


class LindbladEvolution:
    def __init__(self, H: np.ndarray, L_ops: list, gamma_rates: list):
        self.H = H
        self.L_ops = L_ops
        self.gamma_rates = gamma_rates
        self.dim = H.shape[0]

    def liouvillian(self) -> np.ndarray:
        d = self.dim
        L_total = np.zeros((d**2, d**2), dtype=np.complex128)
        I = np.eye(d, dtype=np.complex128)
        H_commutator = -1j * (np.kron(self.H, I) - np.kron(I, self.H.T))
        L_total += H_commutator
        for L, gamma in zip(self.L_ops, self.gamma_rates):
            L_dag = L.conj().T
            L_dag_L = L_dag @ L
            dissipator = gamma * (
                np.kron(L, L.conj())
                - 0.5 * np.kron(L_dag_L, I)
                - 0.5 * np.kron(I, L_dag_L.T)
            )
            L_total += dissipator
        return L_total

    def evolve(self, rho0: np.ndarray, t: float) -> np.ndarray:
        from scipy.linalg import expm
        L = self.liouvillian()
        rho_vec = rho0.flatten()
        evolved_vec = expm(L * t) @ rho_vec
        return evolved_vec.reshape(self.dim, self.dim)

    def coherence_decay(self, rho0: np.ndarray, times: np.ndarray) -> np.ndarray:
        coherences = np.zeros(len(times), dtype=np.complex128)
        for i, t in enumerate(times):
            rho_t = self.evolve(rho0, t)
            coherences[i] = rho_t[0, 1]
        return coherences

    def T2_from_decay(self, coherences: np.ndarray, times: np.ndarray) -> float:
        log_coherences = np.log(np.abs(coherences) + 1e-30)
        coeffs = np.polyfit(times, log_coherences, 1)
        return -1.0 / coeffs[0] if coeffs[0] < 0 else np.inf

    def fidelity(self, rho: np.ndarray, sigma: np.ndarray) -> float:
        from scipy.linalg import sqrtm
        sqrt_rho = sqrtm(rho)
        product = sqrt_rho @ sigma @ sqrt_rho
        return float(np.real(np.trace(sqrtm(product)))**2)


class HolographicDecoherence:
    def __init__(self, ads_system: AdsCft3, material, config: Optional[DecoherenceConfig] = None):
        self.ads = ads_system
        self.material = material
        self.config = config

    def standard_phonon_rate_2d(self, temperature: float) -> float:
        kB = 1.380649e-23
        hbar = 1.054571817e-34
        alpha = getattr(self.material, 'phonon_coupling', 1.0)
        rate = alpha * (kB * temperature / hbar)**2
        return float(rate)

    def holographic_decoherence_rate(
        self, temperature: float, system_size: float, coherence_length: float,
        delta_n: float = 1.0
    ) -> float:
        gamma_std = self.standard_phonon_rate_2d(temperature)
        c = self.ads.central_charge
        exponent = -4.0 * delta_n
        screening = (coherence_length / system_size)**(-exponent)
        return gamma_std * screening

    def coherence_time_ratio(
        self, system_size: float, coherence_length: Optional[float] = None
    ) -> float:
        if coherence_length is None:
            coherence_length = getattr(self.material, 'xi', self.ads.ads_radius)
        ratio = system_size / coherence_length
        c = self.ads.central_charge
        return ratio**(c / 6.0)

    def coherence_time_standard(self, temperature: float) -> float:
        gamma = self.standard_phonon_rate_2d(temperature)
        return 1.0 / gamma if gamma > 0 else np.inf

    def coherence_time_holographic(
        self, temperature: float, system_size: float, coherence_length: Optional[float] = None
    ) -> float:
        T2_std = self.coherence_time_standard(temperature)
        ratio = self.coherence_time_ratio(system_size, coherence_length)
        return T2_std * ratio

    def noise_spectral_density(self, omega: float, noise_type: str = "phonon") -> float:
        kB = 1.380649e-23
        hbar = 1.054571817e-34
        T = getattr(self.config, 'temperature', 4.0) if self.config else 4.0
        if noise_type == "phonon":
            bose = 1.0 / (np.exp(hbar * omega / (kB * T)) - 1.0 + 1e-30)
            return hbar * omega * (bose + 0.5) * 1e-3
        elif noise_type == "charge":
            A = getattr(self.config, 'charge_noise_amplitude', 1e-6) if self.config else 1e-6
            return A / max(abs(omega), 1.0)
        elif noise_type == "white":
            return 1e-4
        return 0.0

    def holographic_noise_after_screening(
        self, omega: float, system_size: float, coherence_length: float,
        delta_n: float = 1.0, noise_type: str = "phonon"
    ) -> float:
        S_bare = self.noise_spectral_density(omega, noise_type)
        screening = (coherence_length / system_size)**(4.0 * delta_n) / 5.0
        return S_bare * screening

    def dephasing_rate_from_noise_spectrum(
        self, omega_max: float, system_size: float, coherence_length: float,
        n_points: int = 500
    ) -> float:
        omega = np.linspace(0.01, omega_max, n_points)
        S = np.array([
            self.holographic_noise_after_screening(w, system_size, coherence_length)
            for w in omega
        ])
        return float(integrate.trapezoid(S, omega))

    def qubit_purity_evolution(
        self, times: np.ndarray, T2: float
    ) -> np.ndarray:
        rho_01 = np.exp(-times / T2)
        purity = 0.5 * (1.0 + rho_01**2)
        return purity

    def quantum_fisher_information(
        self, system_size: float, coherence_length: float
    ) -> float:
        F_std = 1.0
        c = self.ads.central_charge
        correction = (c / 6.0) * np.log(system_size / coherence_length)
        return F_std * (1.0 + correction)

    def temperature_dependence_enhancement(
        self, temperatures: np.ndarray, system_size: float
    ) -> np.ndarray:
        hbar = 1.054571817e-34
        kB = 1.380649e-23
        v_F = self.ads.fermi_velocity
        enhancements = np.zeros(len(temperatures))
        c = self.ads.central_charge
        for i, T in enumerate(temperatures):
            xi_T = hbar * v_F / (kB * T)
            xi_eff = max(getattr(self.material, 'xi', 1e-9), xi_T)
            ratio = system_size / xi_eff
            enhancements[i] = ratio**(c / 6.0) if ratio > 1 else 1.0
        return enhancements

    def combined_topological_holographic_T2(
        self, T2_bare: float, system_size: float, coherence_length: float
    ) -> float:
        topo_factor = np.exp(system_size / coherence_length)
        holo_factor = (system_size / coherence_length)**(self.ads.central_charge / 6.0)
        return T2_bare * topo_factor * holo_factor