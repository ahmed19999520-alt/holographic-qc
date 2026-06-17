from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class Bi2Se3:
    fermi_velocity: float = 5.0e5
    bulk_gap_eV: float = 0.30
    lattice_constant_nm: float = 0.413
    dielectric_constant: float = 100.0
    density_kg_m3: float = 6510.0
    debye_temperature_K: float = 185.0
    phonon_coupling_alpha: float = 2.1e-3
    n_edge_channels: int = 1
    g_factor: float = 20.0

    def __post_init__(self):
        hbar = 1.054571817e-34
        eV = 1.602176634e-19
        self.xi = hbar * self.fermi_velocity / (self.bulk_gap_eV * eV)
        self.central_charge = float(self.n_edge_channels)
        self.a_lattice = self.lattice_constant_nm * 1e-9

    @property
    def xi_nm(self) -> float:
        return self.xi * 1e9

    def t2_standard_ns(self, temperature_K: float) -> float:
        hbar = 1.054571817e-34
        kB = 1.380649e-23
        rate = self.phonon_coupling_alpha * (kB * temperature_K / hbar)**2
        return 1.0e9 / rate

    def t2_holographic_ns(self, temperature_K: float, system_size_m: float) -> float:
        T2_std = self.t2_standard_ns(temperature_K) * 1e-9
        ratio = system_size_m / self.xi
        enhancement = ratio**(self.central_charge / 6.0)
        return T2_std * enhancement * 1e9

    def coherence_length_at_T(self, temperature_K: float) -> float:
        hbar = 1.054571817e-34
        kB = 1.380649e-23
        return hbar * self.fermi_velocity / (kB * temperature_K)

    def arpes_spectrum(self, k_values: np.ndarray, E_values: np.ndarray) -> np.ndarray:
        K, E = np.meshgrid(k_values, E_values)
        E_dirac = self.fermi_velocity * 1.054571817e-34 * np.abs(K)
        sigma_E = 0.005 * 1.602176634e-19
        intensity = np.exp(-0.5 * ((E - E_dirac) / sigma_E)**2)
        return intensity

    def stm_ldos(self, x_m: float, E_eV: float, delta: float = 1.0) -> float:
        hbar = 1.054571817e-34
        eV = 1.602176634e-19
        E_nat = abs(E_eV) * eV
        x_over_xi = abs(x_m) / self.xi
        if x_over_xi < 1e-10:
            return 0.0
        power_law = E_nat**(delta - 1) / (abs(x_m)**(2 * delta) + 1e-30)
        holographic_corr = 1.0 + (self.central_charge / (6.0 * np.pi)) * np.log(x_over_xi)
        return float(power_law * holographic_corr)

    def wiedemann_franz_ratio(self) -> float:
        kB = 1.380649e-23
        e = 1.602176634e-19
        L0 = (np.pi**2 / 3.0) * kB**2 / e**2
        return L0 * (1.0 - 3.0 / self.central_charge)

    def noise_spectral_density(self, omega: float, temperature_K: float) -> float:
        kB = 1.380649e-23
        hbar = 1.054571817e-34
        bose = 1.0 / (np.exp(hbar * omega / (kB * temperature_K)) - 1.0 + 1e-30)
        return self.phonon_coupling_alpha * hbar * omega * (bose + 0.5)

    def material_parameters_dict(self) -> dict:
        return {
            "fermi_velocity_m_s": self.fermi_velocity,
            "bulk_gap_eV": self.bulk_gap_eV,
            "coherence_length_nm": self.xi * 1e9,
            "central_charge": self.central_charge,
            "lattice_constant_nm": self.lattice_constant_nm,
            "phonon_coupling": self.phonon_coupling_alpha,
        }